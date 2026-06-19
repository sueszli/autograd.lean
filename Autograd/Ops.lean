namespace Autograd

/-! Numeric kernels — all operate on flat `Array Float` storage with explicit
dimension arguments, matching `autograd.c`'s `float32_t*` + `shape` style.
Config + cache types live here because Tensor.lean's `GradFn` references them
and must import this module. -/

structure Config where
  nLayer : Nat
  nEmbed : Nat
  blockSize : Nat
  nHead : Nat
  vocabSize : Nat
  numSteps : Nat
  epsilon : Float := 1e-5
  lr0 : Float := 0.01
  beta1 : Float := 0.85
  beta2 : Float := 0.99
  weightDecay : Float := 0.0
  maskValue : Float := -1.0e9
  deriving Inhabited

structure AttnCache where
  xPre : Array Float
  xPreRows : Nat
  xPreCols : Nat
  xn : Array Float
  rms : Array Float
  q : Array (Array Float)
  k : Array (Array Float)
  v : Array (Array Float)
  attnW : Array (Array Float)
  outFlat : Array Float
  deriving Inhabited

structure MlpCache where
  xPre : Array Float
  xn : Array Float
  rows : Nat
  cols : Nat
  rms : Array Float
  hPre : Array Float
  h : Array Float
  hidden : Nat
  deriving Inhabited

-- helper: transpose (r × c) → (c × r) into a fresh flat array
def transposeFlat (x : Array Float) (r c : Nat) : Array Float :=
  (Array.range (c * r)).map fun k => x[(k % r) * c + (k / r)]!

-- x (n × k) @ W (k × m) → (n × m). Pre-materializes products into an array,
-- then sums via foldl. The pre-materialization forces each `x*W` to round to
-- fp64 before any add, blocking the C backend from contracting into FMA.
-- CPython doesn't FMA, so without this we diverge by tens of ULPs per dot.
def linearFwd (x : Array Float) (n k : Nat) (W : Array Float) (m : Nat) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (n * m) 0.0
    for i in [0:n] do
      for j in [0:m] do
        let prods : Array Float := (Array.range k).map fun kk =>
          x[i * k + kk]! * W[kk * m + j]!
        out := out.set! (i * m + j) (prods.foldl (init := 0.0) (· + ·))
    return out

-- dout (n × m) @ Wᵀ (m × k) → (n × k)
def linearBwdX (dout : Array Float) (n m : Nat) (W : Array Float) (k : Nat) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (n * k) 0.0
    for i in [0:n] do
      for kk in [0:k] do
        let mut s : Float := 0.0
        for j in [0:m] do s := s + dout[i * m + j]! * W[kk * m + j]!
        out := out.set! (i * k + kk) s
    return out

-- xᵀ (k × n) @ dout (n × m) → (k × m)
def linearBwdW (dout : Array Float) (n m : Nat) (x : Array Float) (k : Nat) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (k * m) 0.0
    for kk in [0:k] do
      for j in [0:m] do
        let mut s : Float := 0.0
        for i in [0:n] do s := s + x[i * k + kk]! * dout[i * m + j]!
        out := out.set! (kk * m + j) s
    return out

-- elementwise add
def maddFlat (a b : Array Float) : Array Float :=
  (Array.range a.size).map fun i => a[i]! + b[i]!

def reluFlat (x : Array Float) : Array Float := x.map fun z => if z > 0.0 then z else 0.0

def reluBwdFlat (dout hPre : Array Float) : Array Float :=
  (Array.range dout.size).map fun i => if hPre[i]! > 0.0 then dout[i]! else 0.0

-- numerically stable softmax (1-D vector). The `e * (1/s)` matches Value's
-- __truediv__ instead of a direct division (`e / s`).
def softmaxFlat (v : Array Float) : Array Float :=
  if v.size = 0 then v
  else
    let mx := v.foldl (init := v[0]!) fun acc x => if x > acc then x else acc
    let exps := v.map fun x => Float.exp (x - mx)
    let invS := 1.0 / exps.foldl (init := 0.0) (· + ·)
    exps.map fun e => e * invS

-- row-wise softmax over a flat (rows × cols) buffer
def softmaxRows (x : Array Float) (rows cols : Nat) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (rows * cols) 0.0
    for i in [0:rows] do
      let row : Array Float := (Array.range cols).map fun j => x[i * cols + j]!
      let sm := softmaxFlat row
      for j in [0:cols] do out := out.set! (i * cols + j) sm[j]!
    return out

-- aw, daw : (rows × cols). returns scale · aw_ij · (daw_ij − ⟨aw_i, daw_i⟩).
def softmaxRowsBwd (aw daw : Array Float) (rows cols : Nat) (scale : Float) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (rows * cols) 0.0
    for i in [0:rows] do
      let mut dot : Float := 0.0
      for j in [0:cols] do dot := dot + aw[i * cols + j]! * daw[i * cols + j]!
      for j in [0:cols] do
        let g := scale * aw[i * cols + j]! * (daw[i * cols + j]! - dot)
        out := out.set! (i * cols + j) g
    return out

-- split (n × dModel) into nHead flat slices each (n × headDim)
def splitHeadsFlat (x : Array Float) (n dModel nHead : Nat) : Array (Array Float) :=
  let headDim := dModel / nHead
  (Array.range nHead).map fun h =>
    Id.run do
      let mut acc : Array Float := Array.replicate (n * headDim) 0.0
      for i in [0:n] do
        for j in [0:headDim] do acc := acc.set! (i * headDim + j) x[i * dModel + h * headDim + j]!
      return acc

-- concat nHead flat (n × headDim) → (n × dModel)
def mergeHeadsFlat (xs : Array (Array Float)) (n nHead headDim : Nat) : Array Float :=
  let dModel := nHead * headDim
  Id.run do
    let mut out : Array Float := Array.replicate (n * dModel) 0.0
    for i in [0:n] do
      for col in [0:dModel] do
        let h := col / headDim; let j := col % headDim
        out := out.set! (i * dModel + col) xs[h]![i * headDim + j]!
    return out

-- accumulate row gradients into a zeros table by index (embedding backward)
def scatterAddFlat (rows cols : Nat) (grad : Array Float) (ids : Array Nat) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (rows * cols) 0.0
    for i in [0:ids.size] do
      let id := ids[i]!
      for j in [0:cols] do
        out := out.set! (id * cols + j) (out[id * cols + j]! + grad[i * cols + j]!)
    return out

-- masked cross-entropy on already-softmaxed `probs` (rows × cols).
-- Mirrors Python's `(1/n) * sum(losses)` order: reciprocal first, then multiply.
def maskedCrossEntropy (probs : Array Float) (rows cols : Nat) (targetIds : Array Nat)
    (mask : Array Float) (sumMask : Float) : Float :=
  let total := (Array.range rows).foldl (init := 0.0) fun acc i =>
    let p := probs[i * cols + targetIds[i]!]!
    let pClamp := if p < 1e-30 then 1e-30 else p
    acc - mask[i]! * Float.log pClamp
  if sumMask == 0.0 then 0.0 else (1.0 / sumMask) * total

-- fused softmax+CE backward: d_logits[i,c] = mask_i · (probs[i,c] − [c == t_i]) / sumMask
def maskedCrossEntropyBwd (probs : Array Float) (rows cols : Nat) (targetIds : Array Nat)
    (mask : Array Float) (sumMask : Float) : Array Float :=
  let inv := if sumMask == 0.0 then 0.0 else 1.0 / sumMask
  Id.run do
    let mut out : Array Float := Array.replicate (rows * cols) 0.0
    for i in [0:rows] do
      let t := targetIds[i]!
      let m := mask[i]!
      for c in [0:cols] do
        let onehot : Float := if c = t then 1.0 else 0.0
        out := out.set! (i * cols + c) (m * (probs[i * cols + c]! - onehot) * inv)
    return out

/-! ## Layer kernels -/

-- Mirrors Python's order to keep bit-equivalence:
--   ms = sum_of_sq * (1.0/d)          -- Value.__truediv__(d) ⇒ self * d**-1
--   scale = (ms + eps) ** (-0.5)      -- Value.__pow__ stores pow(...,-0.5)
--   y_ij = x_ij * scale               -- Value.__mul__
-- We return (y, rms-per-row) where rms = sqrt(ms+eps) for use in the analytic
-- rmsnormBwd. The `scale = 1/rms` reciprocal is reconstructed in backward.
def rmsnormFwd (x : Array Float) (rows cols : Nat) (eps : Float) : Array Float × Array Float :=
  let invD : Float := 1.0 / cols.toFloat
  let scales : Array Float := (Array.range rows).map fun i =>
    let sumSq := (Array.range cols).foldl (init := 0.0) fun acc j =>
      let v := x[i * cols + j]!
      acc + v * v
    let ms := sumSq * invD
    Float.pow (ms + eps) (-0.5)
  let y : Array Float := Id.run do
    let mut out : Array Float := Array.replicate (rows * cols) 0.0
    for i in [0:rows] do
      let s := scales[i]!
      for j in [0:cols] do out := out.set! (i * cols + j) (x[i * cols + j]! * s)
    return out
  (y, scales)

-- Cache now stores `scale_i = 1/√(ms_i + ε)` (not `√(ms_i + ε)`). With s = scale:
--   dx_ik = s · (dy_ik − x_ik · ⟨dy_i, x_i⟩ · s² / d)
def rmsnormBwd (dy x : Array Float) (scale : Array Float) (rows cols : Nat) : Array Float :=
  let dF := cols.toFloat
  Id.run do
    let mut out : Array Float := Array.replicate (rows * cols) 0.0
    for i in [0:rows] do
      let s := scale[i]!
      let mut dot : Float := 0.0
      for j in [0:cols] do dot := dot + dy[i * cols + j]! * x[i * cols + j]!
      for j in [0:cols] do
        let g := s * (dy[i * cols + j]! - x[i * cols + j]! * dot * s * s / dF)
        out := out.set! (i * cols + j) g
    return out

def attnFwd (cfg : Config) (xPre : Array Float) (rows : Nat) (wq wk wv wo : Array Float)
    : Array Float × AttnCache :=
  let cols := cfg.nEmbed
  let (xn, rms) := rmsnormFwd xPre rows cols cfg.epsilon
  let qFlat := linearFwd xn rows cols wq cols
  let kFlat := linearFwd xn rows cols wk cols
  let vFlat := linearFwd xn rows cols wv cols
  let qs := splitHeadsFlat qFlat rows cols cfg.nHead
  let ks := splitHeadsFlat kFlat rows cols cfg.nHead
  let vs := splitHeadsFlat vFlat rows cols cfg.nHead
  let headDim := cols / cfg.nHead
  -- Mirror Value.__truediv__: precompute `1/√d` then multiply per element.
  let invScale := 1.0 / Float.pow headDim.toFloat 0.5
  let (aws, outs) : Array (Array Float) × Array (Array Float) := Id.run do
    let mut aws : Array (Array Float) := #[]
    let mut outs : Array (Array Float) := #[]
    for h in [0:cfg.nHead] do
      let q := qs[h]!; let k := ks[h]!; let v := vs[h]!
      let kT := transposeFlat k rows headDim
      let scores := linearFwd q rows headDim kT rows
      let scaled : Array Float := scores.map (· * invScale)
      let masked : Array Float := Id.run do
        let mut acc : Array Float := Array.replicate (rows * rows) 0.0
        for i in [0:rows] do
          for j in [0:rows] do
            let v := if j ≤ i then scaled[i * rows + j]! else cfg.maskValue
            acc := acc.set! (i * rows + j) v
        return acc
      let aw := softmaxRows masked rows rows
      aws := aws.push aw
      outs := outs.push (linearFwd aw rows rows v headDim)
    (aws, outs)
  let merged := mergeHeadsFlat outs rows cfg.nHead headDim
  let outFlat := linearFwd merged rows cols wo cols
  let outRes := maddFlat xPre outFlat
  (outRes, { xPre := xPre, xPreRows := rows, xPreCols := cols, xn := xn, rms := rms,
             q := qs, k := ks, v := vs, attnW := aws, outFlat := merged })

def attnBwd (cfg : Config) (dout : Array Float) (rows : Nat) (wq wk wv wo : Array Float)
    (c : AttnCache) : Array Float × (Array Float × Array Float × Array Float × Array Float) :=
  let cols := cfg.nEmbed
  let dMerged := linearBwdX dout rows cols wo cols
  let dWo := linearBwdW dout rows cols c.outFlat cols
  let headDim := cols / cfg.nHead
  let dOutHeads := splitHeadsFlat dMerged rows cols cfg.nHead
  -- backward uses 1/√d as the scale factor folded into softmaxRowsBwd
  let invSqrt := 1.0 / Float.pow headDim.toFloat 0.5
  let (dqs, dks, dvs) : Array (Array Float) × Array (Array Float) × Array (Array Float) := Id.run do
    let mut dqs : Array (Array Float) := #[]
    let mut dks : Array (Array Float) := #[]
    let mut dvs : Array (Array Float) := #[]
    for h in [0:cfg.nHead] do
      let aw := c.attnW[h]!
      let q := c.q[h]!; let k := c.k[h]!; let v := c.v[h]!
      let dHead := dOutHeads[h]!
      -- out = aw @ v ⇒ daw = dHead @ vᵀ ; dv = awᵀ @ dHead
      -- linearBwdX(dout=dHead, n=rows, m=headDim, W=v, k=rows) gives dHead @ vᵀ with shape (rows × rows)
      let dawH := linearBwdX dHead rows headDim v rows
      let dvH := linearBwdW dHead rows headDim aw rows
      let dScaledH := softmaxRowsBwd aw dawH rows rows invSqrt
      let dScaledMasked : Array Float := Id.run do
        let mut acc := dScaledH
        for i in [0:rows] do
          for j in [0:rows] do
            if j > i then acc := acc.set! (i * rows + j) 0.0
        return acc
      -- scores = q @ kᵀ ⇒ dq = dScaled @ k ; dk = dScaledᵀ @ q (both forward matmuls)
      let dqH := linearFwd dScaledMasked rows rows k headDim
      let dkH := linearFwd (transposeFlat dScaledMasked rows rows) rows rows q headDim
      dqs := dqs.push dqH
      dks := dks.push dkH
      dvs := dvs.push dvH
    (dqs, dks, dvs)
  let dQflat := mergeHeadsFlat dqs rows cfg.nHead headDim
  let dKflat := mergeHeadsFlat dks rows cfg.nHead headDim
  let dVflat := mergeHeadsFlat dvs rows cfg.nHead headDim
  let dXnQ := linearBwdX dQflat rows cols wq cols
  let dXnK := linearBwdX dKflat rows cols wk cols
  let dXnV := linearBwdX dVflat rows cols wv cols
  let dXn := maddFlat (maddFlat dXnQ dXnK) dXnV
  let dWq := linearBwdW dQflat rows cols c.xn cols
  let dWk := linearBwdW dKflat rows cols c.xn cols
  let dWv := linearBwdW dVflat rows cols c.xn cols
  let dxPre := maddFlat dout (rmsnormBwd dXn c.xPre c.rms rows cols)
  (dxPre, (dWq, dWk, dWv, dWo))

def mlpFwd (cfg : Config) (xPre : Array Float) (rows : Nat) (fc1 fc2 : Array Float)
    : Array Float × MlpCache :=
  let cols := cfg.nEmbed
  let hidden := 4 * cols
  let (xn, rms) := rmsnormFwd xPre rows cols cfg.epsilon
  let hPre := linearFwd xn rows cols fc1 hidden
  let h := reluFlat hPre
  let y := linearFwd h rows hidden fc2 cols
  (maddFlat xPre y, { xPre := xPre, xn := xn, rows := rows, cols := cols, rms := rms,
                      hPre := hPre, h := h, hidden := hidden })

def mlpBwd (dout : Array Float) (fc1 fc2 : Array Float) (c : MlpCache)
    : Array Float × (Array Float × Array Float) :=
  let dh := linearBwdX dout c.rows c.cols fc2 c.hidden
  let dfc2 := linearBwdW dout c.rows c.cols c.h c.hidden
  let dhPre := reluBwdFlat dh c.hPre
  let dxn := linearBwdX dhPre c.rows c.hidden fc1 c.cols
  let dfc1 := linearBwdW dhPre c.rows c.hidden c.xn c.cols
  (maddFlat dout (rmsnormBwd dxn c.xPre c.rms c.rows c.cols), (dfc1, dfc2))

end Autograd
