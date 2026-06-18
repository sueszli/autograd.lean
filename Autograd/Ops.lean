import Autograd.Matrix

namespace Autograd

-- linear: x (n × k) @ W (k × m) → (n × m)
def linearFwd (x W : Matrix) : Matrix :=
  let n := x.size
  let m := if W.size = 0 then 0 else W[0]!.size
  let k := W.size
  Matrix.ofFn n m fun i j =>
    (Array.range k).foldl (init := 0.0) fun acc kk => acc + x[i]![kk]! * W[kk]![j]!

def linearBwdX (dout W : Matrix) : Matrix :=
  let n := dout.size
  let m := if dout.size = 0 then 0 else dout[0]!.size
  let k := W.size
  Matrix.ofFn n k fun i kk =>
    (Array.range m).foldl (init := 0.0) fun acc j => acc + dout[i]![j]! * W[kk]![j]!

def linearBwdW (dout x : Matrix) : Matrix :=
  let n := x.size
  let k := if x.size = 0 then 0 else x[0]!.size
  let m := if dout.size = 0 then 0 else dout[0]!.size
  Matrix.ofFn k m fun kk j =>
    (Array.range n).foldl (init := 0.0) fun acc i => acc + x[i]![kk]! * dout[i]![j]!

def madd (a b : Matrix) : Matrix :=
  (Array.range a.size).map fun i =>
    (Array.range a[i]!.size).map fun j => a[i]![j]! + b[i]![j]!

def relu (m : Matrix) : Matrix :=
  m.map fun row => row.map fun x => if x > 0.0 then x else 0.0

def reluBwd (dout hPre : Matrix) : Matrix :=
  (Array.range dout.size).map fun i =>
    (Array.range dout[i]!.size).map fun j =>
      if hPre[i]![j]! > 0.0 then dout[i]![j]! else 0.0

-- numerically stable softmax (vector)
def softmax (v : Array Float) : Array Float :=
  if v.size = 0 then v
  else
    let mx := v.foldl (init := v[0]!) fun acc x => if x > acc then x else acc
    let exps := v.map fun x => Float.exp (x - mx)
    let s := exps.foldl (init := 0.0) (· + ·)
    exps.map fun e => e / s

-- row-wise softmax-Jacobian × upstream, with scale folded in
def softmaxBwd (aw daw : Matrix) (scale : Float) : Matrix :=
  (Array.range aw.size).map fun i =>
    let row := aw[i]!
    let drow := daw[i]!
    let dot := (Array.range row.size).foldl (init := 0.0) fun acc j =>
      acc + row[j]! * drow[j]!
    (Array.range row.size).map fun j =>
      scale * row[j]! * (drow[j]! - dot)

-- split (n × dModel) into nHead matrices (n × headDim)
def splitHeads (x : Matrix) (nHead : Nat) : Array Matrix :=
  let n := x.size
  let dModel := if n = 0 then 0 else x[0]!.size
  let headDim := dModel / nHead
  (Array.range nHead).map fun h =>
    (Array.range n).map fun i =>
      (Array.range headDim).map fun j => x[i]![h * headDim + j]!

def mergeHeads (xs : Array Matrix) : Matrix :=
  let nHead := xs.size
  if nHead = 0 then #[]
  else
    let n := xs[0]!.size
    let headDim := if n = 0 then 0 else xs[0]![0]!.size
    (Array.range n).map fun i =>
      (Array.range (nHead * headDim)).map fun col =>
        xs[col / headDim]![i]![col % headDim]!

-- loss = − Σ_i mask_i · log(probs[i, target_i]) / sumMask
def maskedCrossEntropy (probs : Matrix) (targetIds : Array Nat)
    (mask : Array Float) (sumMask : Float) : Float :=
  let total := (Array.range probs.size).foldl (init := 0.0) fun acc i =>
    let p := probs[i]![targetIds[i]!]!
    let pClamp := if p < 1e-30 then 1e-30 else p
    acc - mask[i]! * Float.log pClamp
  if sumMask == 0.0 then 0.0 else total / sumMask

-- fused softmax+CE backward
def maskedCrossEntropyBwd (probs : Matrix) (targetIds : Array Nat)
    (mask : Array Float) (sumMask : Float) : Matrix :=
  let inv := if sumMask == 0.0 then 0.0 else 1.0 / sumMask
  (Array.range probs.size).map fun i =>
    let row := probs[i]!
    let t := targetIds[i]!
    let m := mask[i]!
    (Array.range row.size).map fun c =>
      let onehot : Float := if c = t then 1.0 else 0.0
      m * (row[c]! - onehot) * inv

/-! ## Layer kernels (numeric only; Tensor.lean wraps them with autograd) -/

-- caches mirror the Python reference's namedtuples; they're the "ctx" carried
-- by GradFn variants so backward can run without re-doing the forward.
structure AttnCache where
  xPre : Matrix
  xn : Matrix
  rms : Array Float
  q : Array Matrix
  k : Array Matrix
  v : Array Matrix
  attnW : Array Matrix
  outFlat : Matrix
  deriving Inhabited

structure MlpCache where
  xPre : Matrix
  xn : Matrix
  rms : Array Float
  hPre : Matrix
  h : Matrix
  deriving Inhabited

-- y[i,j] = x[i,j] / sqrt(mean(x[i,:]²) + ε)
def rmsnormFwd (x : Matrix) (eps : Float) : Matrix × Array Float :=
  let n := x.size
  let d := if n = 0 then 0 else x[0]!.size
  let rms : Array Float := (Array.range n).map fun i =>
    let row := x[i]!
    let ms := (Array.range d).foldl (init := 0.0) fun acc j => acc + row[j]! * row[j]!
    Float.sqrt (ms / d.toFloat + eps)
  let y : Matrix := (Array.range n).map fun i =>
    let r := rms[i]!
    (Array.range d).map fun j => x[i]![j]! / r
  (y, rms)

-- dx_ik = (dy_ik − x_ik · ⟨dy_i, x_i⟩ / (d · rms_i²)) / rms_i
def rmsnormBwd (dy x : Matrix) (rms : Array Float) : Matrix :=
  let n := x.size
  let d := if n = 0 then 0 else x[0]!.size
  let dF := d.toFloat
  (Array.range n).map fun i =>
    let row := x[i]!
    let drow := dy[i]!
    let r := rms[i]!
    let dot := (Array.range d).foldl (init := 0.0) fun acc j => acc + drow[j]! * row[j]!
    (Array.range d).map fun j =>
      (drow[j]! - row[j]! * dot / (dF * r * r)) / r

def attnFwd (cfg : Config) (xPre : Matrix) (wq wk wv wo : Matrix) : Matrix × AttnCache :=
  let (xn, rms) := rmsnormFwd xPre cfg.epsilon
  let qs := splitHeads (linearFwd xn wq) cfg.nHead
  let ks := splitHeads (linearFwd xn wk) cfg.nHead
  let vs := splitHeads (linearFwd xn wv) cfg.nHead
  let headDim := cfg.nEmbed / cfg.nHead
  let invSqrt := 1.0 / Float.sqrt headDim.toFloat
  let (aws, outs) : Array Matrix × Array Matrix := Id.run do
    let mut aws : Array Matrix := #[]
    let mut outs : Array Matrix := #[]
    for h in [0:cfg.nHead] do
      let q := qs[h]!; let k := ks[h]!; let v := vs[h]!
      let scaled := (linearFwd q (Matrix.transpose k)).map (·.map (invSqrt * ·))
      let masked : Matrix := (Array.range scaled.size).map fun i =>
        (Array.range scaled[i]!.size).map fun j =>
          if j ≤ i then scaled[i]![j]! else cfg.maskValue
      let aw := masked.map softmax
      aws := aws.push aw
      outs := outs.push (linearFwd aw v)
    (aws, outs)
  let merged := mergeHeads outs
  let outRes := madd xPre (linearFwd merged wo)
  (outRes, { xPre := xPre, xn := xn, rms := rms, q := qs, k := ks, v := vs,
             attnW := aws, outFlat := merged })

def attnBwd (cfg : Config) (dout : Matrix) (wq wk wv wo : Matrix) (c : AttnCache)
    : Matrix × (Matrix × Matrix × Matrix × Matrix) :=
  let dMerged := linearBwdX dout wo
  let dWo := linearBwdW dout c.outFlat
  let dOutHeads := splitHeads dMerged cfg.nHead
  let headDim := cfg.nEmbed / cfg.nHead
  let invSqrt := 1.0 / Float.sqrt headDim.toFloat
  let (dqs, dks, dvs) : Array Matrix × Array Matrix × Array Matrix := Id.run do
    let mut dqs : Array Matrix := #[]
    let mut dks : Array Matrix := #[]
    let mut dvs : Array Matrix := #[]
    for h in [0:cfg.nHead] do
      let aw := c.attnW[h]!
      let q := c.q[h]!; let k := c.k[h]!; let v := c.v[h]!
      let dHead := dOutHeads[h]!
      let dawH := linearBwdX dHead v
      let dvH := linearBwdW dHead aw
      let dScaledH := softmaxBwd aw dawH invSqrt
      let dScaledMasked : Matrix := (Array.range dScaledH.size).map fun i =>
        (Array.range dScaledH[i]!.size).map fun j =>
          if j ≤ i then dScaledH[i]![j]! else 0.0
      dqs := dqs.push (linearBwdX dScaledMasked (Matrix.transpose k))
      dks := dks.push (Matrix.transpose (linearBwdW dScaledMasked q))
      dvs := dvs.push dvH
    (dqs, dks, dvs)
  let dQflat := mergeHeads dqs
  let dKflat := mergeHeads dks
  let dVflat := mergeHeads dvs
  let dXn := madd (madd (linearBwdX dQflat wq) (linearBwdX dKflat wk)) (linearBwdX dVflat wv)
  let dWq := linearBwdW dQflat c.xn
  let dWk := linearBwdW dKflat c.xn
  let dWv := linearBwdW dVflat c.xn
  let dxPre := madd dout (rmsnormBwd dXn c.xPre c.rms)
  (dxPre, (dWq, dWk, dWv, dWo))

def mlpFwd (cfg : Config) (xPre : Matrix) (fc1 fc2 : Matrix) : Matrix × MlpCache :=
  let (xn, rms) := rmsnormFwd xPre cfg.epsilon
  let hPre := linearFwd xn fc1
  let h := relu hPre
  (madd xPre (linearFwd h fc2),
   { xPre := xPre, xn := xn, rms := rms, hPre := hPre, h := h })

def mlpBwd (dout : Matrix) (fc1 fc2 : Matrix) (c : MlpCache)
    : Matrix × (Matrix × Matrix) :=
  let dh := linearBwdX dout fc2
  let dfc2 := linearBwdW dout c.h
  let dhPre := reluBwd dh c.hPre
  let dxn := linearBwdX dhPre fc1
  let dfc1 := linearBwdW dhPre c.xn
  (madd dout (rmsnormBwd dxn c.xPre c.rms), (dfc1, dfc2))

-- accumulate row gradients into a zeros table by index (used in embedding backward)
def scatterAdd (rows cols : Nat) (grad : Matrix) (ids : Array Nat) : Matrix :=
  (Array.range ids.size).foldl (init := Matrix.zeros rows cols) fun A i =>
    let id := ids[i]!
    let drow := grad[i]!
    A.set! id ((Array.range cols).map fun j => A[id]![j]! + drow[j]!)

-- linearFwd (verified by hand)
example : allcloseM
    (linearFwd #[#[1.0, 2.0], #[3.0, 4.0]] #[#[0.5, -0.5], #[1.0, 0.0]])
    #[#[2.5, -0.5], #[5.5, -1.5]] := by native_decide

example :
  let x : Matrix := #[#[1.0, 2.0], #[3.0, 4.0]]
  let W : Matrix := #[#[0.5, -0.5], #[1.0, 0.0]]
  let dout : Matrix := #[#[1.0, 1.0], #[1.0, 1.0]]
  allcloseM (linearBwdX dout W) (fdGradMat (fun X => matSum (linearFwd X W)) x) := by native_decide

example :
  let x : Matrix := #[#[1.0, 2.0], #[3.0, 4.0]]
  let W : Matrix := #[#[0.5, -0.5], #[1.0, 0.0]]
  let dout : Matrix := #[#[1.0, 1.0], #[1.0, 1.0]]
  allcloseM (linearBwdW dout x) (fdGradMat (fun Wp => matSum (linearFwd x Wp)) W) := by native_decide

example : allcloseM (relu #[#[1.0, -2.0], #[0.0, 3.0]]) #[#[1.0, 0.0], #[0.0, 3.0]] := by native_decide
example :
  let x : Matrix := #[#[1.0, -2.0], #[0.5, -0.5]]
  let dout : Matrix := #[#[1.0, 1.0], #[1.0, 1.0]]
  allcloseM (reluBwd dout x) (fdGradMat (fun X => matSum (relu X)) x) := by native_decide

example : (((softmax #[1.0, 2.0, 3.0]).foldl (init := 0.0) (· + ·)) - 1.0).abs < 1e-6 := by native_decide
example : allcloseV (softmax #[1.0, 2.0, 3.0]) (softmax #[101.0, 102.0, 103.0]) 1e-6 := by native_decide

example :
  let x : Matrix := #[#[0.1, 0.5, -0.3]]
  let daw : Matrix := #[#[1.0, -1.0, 0.5]]
  let aw := #[softmax x[0]!]
  let loss : Matrix → Float := fun X =>
    (Array.range (softmax X[0]!).size).foldl (init := 0.0) fun acc j =>
      acc + daw[0]![j]! * (softmax X[0]!)[j]!
  allcloseM (softmaxBwd aw daw 1.0) (fdGradMat loss x) := by native_decide

example :
  let x : Matrix := #[#[1.0, 2.0, 3.0, 4.0], #[5.0, 6.0, 7.0, 8.0]]
  allcloseM (mergeHeads (splitHeads x 2)) x := by native_decide

example :
  let v := 4
  let probs : Matrix := #[Array.replicate v (1.0 / v.toFloat),
                          Array.replicate v (1.0 / v.toFloat)]
  ((maskedCrossEntropy probs #[0, 1] #[1.0, 1.0] 2.0) - Float.log v.toFloat).abs < 1e-6 := by native_decide

example :
  let logits : Matrix := #[#[0.1, 0.4, -0.2, 0.5], #[1.0, -1.0, 0.3, 0.0]]
  let targets : Array Nat := #[2, 0]
  let mask : Array Float := #[1.0, 1.0]
  let probs : Matrix := logits.map softmax
  let bwd := maskedCrossEntropyBwd probs targets mask 2.0
  let loss : Matrix → Float := fun L => maskedCrossEntropy (L.map softmax) targets mask 2.0
  allcloseM bwd (fdGradMat loss logits) := by native_decide

example :
  let x := #[1.5, -0.5, 2.0]
  allcloseV (Array.replicate x.size 1.0)
            (fdGradVec (fun v => v.foldl (init := 0.0) (· + ·)) x) := by native_decide
example :
  let x := #[1.5, -0.5, 2.0]
  allcloseV (Array.replicate x.size (1.0 / x.size.toFloat))
            (fdGradVec (fun v => if v.size = 0 then 0.0
                                 else (v.foldl (init := 0.0) (· + ·)) / v.size.toFloat) x) := by native_decide

-- RMSNorm backward via FD
example :
  let x : Matrix := #[#[1.0, 2.0, -1.0, 0.5], #[0.3, -0.4, 1.2, 0.1]]
  let dy : Matrix := #[#[0.5, -1.0, 0.2, 0.3], #[1.0, 0.0, -0.5, 0.7]]
  let (_, rms) := rmsnormFwd x 1e-5
  allcloseM (rmsnormBwd dy x rms)
            (fdGradMat (fun X => wLoss dy (rmsnormFwd X 1e-5).1) x) 5e-3 := by native_decide

-- MLP backward via FD
example :
  let cfg : Config := { nLayer := 1, nEmbed := 4, blockSize := 2, nHead := 2, vocabSize := 0, numSteps := 0 }
  let x : Matrix := #[#[1.0, 2.0, -1.0, 0.5], #[0.3, -0.4, 1.2, 0.1]]
  let fc1 : Matrix := #[#[0.1, -0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8],
                        #[0.2, 0.1, -0.3, 0.4, -0.5, 0.6, 0.7, -0.8],
                        #[-0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8],
                        #[0.1, -0.2, -0.3, 0.4, 0.5, 0.6, 0.7, -0.8]]
  let fc2 : Matrix := #[#[0.1, 0.2, 0.3, -0.4], #[-0.5, 0.6, -0.7, 0.8],
                        #[0.1, -0.2, 0.3, 0.4], #[0.5, 0.6, 0.7, 0.8],
                        #[-0.1, 0.2, -0.3, 0.4], #[0.5, -0.6, 0.7, -0.8],
                        #[0.1, 0.2, -0.3, -0.4], #[-0.5, -0.6, 0.7, 0.8]]
  let dout : Matrix := #[#[0.5, -1.0, 0.2, 0.3], #[1.0, 0.0, -0.5, 0.7]]
  let (_, cache) := mlpFwd cfg x fc1 fc2
  let (dxPre, _) := mlpBwd dout fc1 fc2 cache
  allcloseM dxPre (fdGradMat (fun X => wLoss dout (mlpFwd cfg X fc1 fc2).1) x) 5e-3 := by native_decide

-- Attention backward via FD (single head)
example :
  let cfg : Config := { nLayer := 1, nEmbed := 2, blockSize := 2, nHead := 1, vocabSize := 0, numSteps := 0 }
  let x : Matrix := #[#[1.0, 2.0], #[0.3, -0.4]]
  let wq : Matrix := #[#[0.1, -0.2], #[0.3, 0.4]]
  let wk : Matrix := #[#[0.5, 0.1], #[-0.2, 0.3]]
  let wv : Matrix := #[#[0.2, -0.5], #[0.4, 0.1]]
  let wo : Matrix := #[#[-0.3, 0.4], #[0.5, -0.1]]
  let dout : Matrix := #[#[1.0, 0.5], #[-0.3, 0.7]]
  let (_, cache) := attnFwd cfg x wq wk wv wo
  let (dxPre, _) := attnBwd cfg dout wq wk wv wo cache
  allcloseM dxPre (fdGradMat (fun X => wLoss dout (attnFwd cfg X wq wk wv wo).1) x) 5e-3 := by native_decide

end Autograd
