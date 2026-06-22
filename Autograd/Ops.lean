import Autograd.Utils

namespace Autograd

/-!
===--------------------------------------------------------------------------===
Size-invariant proof helpers

`#guard` pins output sizes at fixed inputs. The `_size` theorems below prove them
for every input: the invariant the flat-buffer `i * cols + j` indexing relies on.
`foldl_size_pres`: a `List.foldl` whose step preserves `.size` preserves it overall.
`replicate_loop2_size`: a double `Id.run` loop that only `setIfInBounds` into a
`replicate` keeps that buffer's size.
===--------------------------------------------------------------------------===
-/

theorem foldl_size_pres {α : Type} (l : List α) (init : Array Float) (f : Array Float → α → Array Float) (hf : ∀ b a, (f b a).size = b.size) : (l.foldl f init).size = init.size := by
  induction l generalizing init with
  | nil => rfl
  | cons x xs ih => rw [List.foldl_cons, ih, hf]

theorem replicate_loop2_size {α : Type} {β : Type} (outer : List α) (inner : α → List β) (n : Nat) (v : Float) (g : Array Float → α → β → Nat) (h : Array Float → α → β → Float) : (outer.foldl (fun b a => (inner a).foldl (fun b' c => b'.setIfInBounds (g b' a c) (h b' a c)) b) (Array.replicate n v)).size = n :=
  (foldl_size_pres _ _ _ (fun _ _ => foldl_size_pres _ _ _ (fun _ _ => Array.size_setIfInBounds ..))).trans (Array.size_replicate ..)

/-!
===--------------------------------------------------------------------------===
Matmul
===--------------------------------------------------------------------------===
-/

def transposeFlat (x : Array Float) (r c : Nat) : Array Float :=
  (Array.range (c * r)).map fun k => x[(k % r) * c + (k / r)]!

-- pre-materialize `x*W` products so the C backend can't contract them into FMA. CPython doesn't FMA, so without this we diverge by tens of ULPs per dot.
def matmulFwd (x : Array Float) (n k : Nat) (W : Array Float) (m : Nat) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (n * m) 0.0
    for i in [0:n] do
      for j in [0:m] do
        let prods : Array Float := (Array.range k).map fun kk => x[i * k + kk]! * W[kk * m + j]!
        out := out.set! (i * m + j) (prods.foldl (init := 0.0) (· + ·))
    return out

def matmulBwdX (dout : Array Float) (n m : Nat) (W : Array Float) (k : Nat) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (n * k) 0.0
    for i in [0:n] do
      for kk in [0:k] do
        let mut s : Float := 0.0
        for j in [0:m] do s := s + dout[i * m + j]! * W[kk * m + j]!
        out := out.set! (i * k + kk) s
    return out

def matmulBwdW (dout : Array Float) (n m : Nat) (x : Array Float) (k : Nat) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (k * m) 0.0
    for kk in [0:k] do
      for j in [0:m] do
        let mut s : Float := 0.0
        for i in [0:n] do s := s + x[i * k + kk]! * dout[i * m + j]!
        out := out.set! (kk * m + j) s
    return out

theorem transposeFlat_size (x : Array Float) (r : Nat) (c : Nat) : (transposeFlat x r c).size = c * r := by simp [transposeFlat]
theorem matmulFwd_size (x : Array Float) (n : Nat) (k : Nat) (W : Array Float) (m : Nat) : (matmulFwd x n k W m).size = n * m := by simp [matmulFwd, Id.run]; exact replicate_loop2_size ..
theorem matmulBwdX_size (dout : Array Float) (n : Nat) (m : Nat) (W : Array Float) (k : Nat) : (matmulBwdX dout n m W k).size = n * k := by simp [matmulBwdX, Id.run]; exact replicate_loop2_size ..
theorem matmulBwdW_size (dout : Array Float) (n : Nat) (m : Nat) (x : Array Float) (k : Nat) : (matmulBwdW dout n m x k).size = k * m := by simp [matmulBwdW, Id.run]; exact replicate_loop2_size ..

#guard arrApproxEq (transposeFlat #[1, 2, 3, 4, 5, 6] 2 3) #[1, 4, 2, 5, 3, 6]
#guard arrApproxEq (transposeFlat (transposeFlat #[1, 2, 3, 4, 5, 6] 2 3) 3 2) #[1, 2, 3, 4, 5, 6]  -- transpose is its own inverse
#guard arrApproxEq (matmulFwd #[1, 2, 3, 4] 2 2 #[1, 2, 3, 4] 2) #[7, 10, 15, 22]
#guard arrApproxEq (matmulFwd #[1, 2, 3, 4, 5, 6] 2 3 #[1, 2, 3, 4, 5, 6] 2) #[22, 28, 49, 64]  -- rectangular `2×3 @ 3×2`, exercises `n ≠ m`
-- with an identity operand, `matmulBwdX`/`matmulBwdW` pass `dout` straight through
#guard arrApproxEq (matmulBwdX #[1, 2, 3, 4] 2 2 #[1, 0, 0, 1] 2) #[1, 2, 3, 4]
#guard arrApproxEq (matmulBwdW #[1, 2, 3, 4] 2 2 #[1, 0, 0, 1] 2) #[1, 2, 3, 4]

/-!
===--------------------------------------------------------------------------===
Element-wise
===--------------------------------------------------------------------------===
-/

def maddFlat (a b : Array Float) : Array Float :=
  (Array.range a.size).map fun i => a[i]! + b[i]!

def reluFlat (x : Array Float) : Array Float := x.map fun z => if z > 0.0 then z else 0.0

def reluBwdFlat (dout hPre : Array Float) : Array Float :=
  (Array.range dout.size).map fun i => if hPre[i]! > 0.0 then dout[i]! else 0.0

theorem maddFlat_size (a : Array Float) (b : Array Float) : (maddFlat a b).size = a.size := by simp [maddFlat]
theorem reluFlat_size (x : Array Float) : (reluFlat x).size = x.size := by simp [reluFlat]
theorem reluBwdFlat_size (dout : Array Float) (hPre : Array Float) : (reluBwdFlat dout hPre).size = dout.size := by simp [reluBwdFlat]

#guard arrApproxEq (maddFlat #[1, 2, 3] #[3, 4, 5]) #[4, 6, 8]
#guard arrApproxEq (reluFlat #[-1, 0, 2, -3]) #[0, 0, 2, 0]                  -- clamps negatives and exactly-zero
#guard arrApproxEq (reluBwdFlat #[1, 1, 1, 1] #[-1, 0, 2, -3]) #[0, 0, 1, 0]  -- gradient gated by the pre-activation sign

/-!
===--------------------------------------------------------------------------===
Softmax
===--------------------------------------------------------------------------===
-/

-- `e * (1/s)` (not `e / s`) mirrors Python's `Value.__truediv__` for bit-exact parity.
def softmaxFlat (v : Array Float) : Array Float :=
  if v.size = 0 then v
  else
    let mx := v.foldl (init := v[0]!) fun acc x => if x > acc then x else acc
    let exps := v.map fun x => Float.exp (x - mx)
    let invS := 1.0 / exps.foldl (init := 0.0) (· + ·)
    exps.map fun e => e * invS

def softmaxRows (x : Array Float) (rows cols : Nat) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (rows * cols) 0.0
    for i in [0:rows] do
      let row : Array Float := (Array.range cols).map fun j => x[i * cols + j]!
      let sm := softmaxFlat row
      for j in [0:cols] do out := out.set! (i * cols + j) sm[j]!
    return out

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

theorem softmaxFlat_size (v : Array Float) : (softmaxFlat v).size = v.size := by unfold softmaxFlat; split <;> simp
theorem softmaxRows_size (x : Array Float) (rows : Nat) (cols : Nat) : (softmaxRows x rows cols).size = rows * cols := by simp [softmaxRows, Id.run]; exact replicate_loop2_size ..
theorem softmaxRowsBwd_size (aw : Array Float) (daw : Array Float) (rows : Nat) (cols : Nat) (scale : Float) : (softmaxRowsBwd aw daw rows cols scale).size = rows * cols := by simp [softmaxRowsBwd, Id.run]; exact replicate_loop2_size ..

#guard arrApproxEq (softmaxFlat #[0, 0, 0]) #[1.0 / 3, 1.0 / 3, 1.0 / 3]
#guard (softmaxFlat #[]).size == 0                                                -- empty-input branch returns the input untouched
#guard approxEq ((softmaxFlat #[1, 2, 3]).foldl (· + ·) 0.0) 1.0                  -- normalized
#guard arrApproxEq (softmaxFlat #[1, 2, 3]) (softmaxFlat #[-4, -3, -2])           -- shift-invariant (max-subtraction)
#guard let sm := softmaxRows #[1, 2, 1, 0] 2 2; approxEq (sm[0]! + sm[1]!) 1.0 && approxEq (sm[2]! + sm[3]!) 1.0
-- a row-constant upstream gradient maps to zero: the softmax Jacobian kills the common-mode component
#guard arrApproxEq (softmaxRowsBwd (softmaxRows #[1, 2, 3, 4] 1 4) #[5, 5, 5, 5] 1 4 1.0) #[0, 0, 0, 0]

/-!
===--------------------------------------------------------------------------===
Multi-head
===--------------------------------------------------------------------------===
-/

def splitHeadsFlat (x : Array Float) (n dModel nHead : Nat) : Array (Array Float) :=
  let headDim := dModel / nHead
  (Array.range nHead).map fun h =>
    Id.run do
      let mut acc : Array Float := Array.replicate (n * headDim) 0.0
      for i in [0:n] do
        for j in [0:headDim] do acc := acc.set! (i * headDim + j) x[i * dModel + h * headDim + j]!
      return acc

def mergeHeadsFlat (xs : Array (Array Float)) (n nHead headDim : Nat) : Array Float :=
  let dModel := nHead * headDim
  Id.run do
    let mut out : Array Float := Array.replicate (n * dModel) 0.0
    for i in [0:n] do
      for col in [0:dModel] do
        let h := col / headDim; let j := col % headDim
        out := out.set! (i * dModel + col) xs[h]![i * headDim + j]!
    return out

theorem splitHeadsFlat_count (x : Array Float) (n : Nat) (dModel : Nat) (nHead : Nat) : (splitHeadsFlat x n dModel nHead).size = nHead := by simp [splitHeadsFlat]
theorem mergeHeadsFlat_size (xs : Array (Array Float)) (n : Nat) (nHead : Nat) (headDim : Nat) : (mergeHeadsFlat xs n nHead headDim).size = n * (nHead * headDim) := by simp [mergeHeadsFlat, Id.run]; exact replicate_loop2_size ..

-- one row of `d_model = 4` splits into 2 heads of width 2: first half, second half
#guard let hs := splitHeadsFlat #[1, 2, 3, 4] 1 4 2; hs.size == 2 && arrApproxEq hs[0]! #[1, 2] && arrApproxEq hs[1]! #[3, 4]
-- merge undoes split for a 2-row, 2-head buffer
#guard arrApproxEq (mergeHeadsFlat (splitHeadsFlat #[1, 2, 3, 4, 5, 6, 7, 8] 2 4 2) 2 2 2) #[1, 2, 3, 4, 5, 6, 7, 8]

/-!
===--------------------------------------------------------------------------===
Embedding scatter
===--------------------------------------------------------------------------===
-/

-- data is flat row-major, so decode output index `k` to 2D via `/ cols` and `% cols`,
-- then re-encode the source cell as `srcRow * cols + col`. duplicate ids are fine
def gatherFlat (table : Array Float) (cols : Nat) (ids : Array Nat) : Array Float :=
  (Array.range (ids.size * cols)).map fun k => table[ids[k / cols]! * cols + k % cols]!

def scatterAddFlat (rows cols : Nat) (grad : Array Float) (ids : Array Nat) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (rows * cols) 0.0
    for i in [0:ids.size] do
      let id := ids[i]!
      for j in [0:cols] do
        out := out.set! (id * cols + j) (out[id * cols + j]! + grad[i * cols + j]!)
    return out

theorem gatherFlat_size (table : Array Float) (cols : Nat) (ids : Array Nat) : (gatherFlat table cols ids).size = ids.size * cols := by simp [gatherFlat]
theorem scatterAddFlat_size (rows : Nat) (cols : Nat) (grad : Array Float) (ids : Array Nat) : (scatterAddFlat rows cols grad ids).size = rows * cols := by simp [scatterAddFlat, Id.run]; exact replicate_loop2_size ..

-- ids [2, 0] select table rows 2 then 0
#guard arrApproxEq (gatherFlat #[10, 11, 20, 21, 30, 31] 2 #[2, 0]) #[30, 31, 10, 11]
-- ids [0, 0, 2] over 3 rows: row 0 accumulates both grads, row 1 stays zero, row 2 gets the third
#guard arrApproxEq (scatterAddFlat 3 2 #[1, 1, 2, 2, 3, 3] #[0, 0, 2]) #[3, 3, 0, 0, 3, 3]

/-!
===--------------------------------------------------------------------------===
Cross-entropy
===--------------------------------------------------------------------------===
-/

-- `(1/n) * sum(losses)` order (reciprocal first) mirrors Python for bit-exact parity.
def maskedCrossEntropy (probs : Array Float) (rows cols : Nat) (targetIds : Array Nat) (mask : Array Float) (sumMask : Float) : Float :=
  let total := (Array.range rows).foldl (init := 0.0) fun acc i =>
    let p := probs[i * cols + targetIds[i]!]!
    let pClamp := if p < 1e-30 then 1e-30 else p
    acc - mask[i]! * Float.log pClamp
  if sumMask == 0.0 then 0.0 else (1.0 / sumMask) * total

def maskedCrossEntropyBwd (probs : Array Float) (rows cols : Nat) (targetIds : Array Nat) (mask : Array Float) (sumMask : Float) : Array Float :=
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

theorem maskedCrossEntropyBwd_size (probs : Array Float) (rows : Nat) (cols : Nat) (t : Array Nat) (mask : Array Float) (sumMask : Float) : (maskedCrossEntropyBwd probs rows cols t mask sumMask).size = rows * cols := by simp [maskedCrossEntropyBwd, Id.run]; exact replicate_loop2_size ..

-- a 50/50 prediction on a 2-class target costs `-log 0.5`
#guard approxEq (maskedCrossEntropy #[0.5, 0.5] 1 2 #[0] #[1] 1.0) (-Float.log 0.5)
#guard approxEq (maskedCrossEntropy #[0.5, 0.5] 1 2 #[0] #[0] 0.0) 0.0          -- zero mask sum short-circuits to 0
#guard approxEq (maskedCrossEntropy #[0.5, 0.5, 0.5, 0.5] 2 2 #[0, 1] #[1, 0] 1.0) (-Float.log 0.5)  -- per-token `mask=0` drops the second row from the sum
-- gradient is `probs - onehot` (scaled); it must sum to zero across the row
#guard arrApproxEq (maskedCrossEntropyBwd #[0.5, 0.5] 1 2 #[0] #[1] 1.0) #[-0.5, 0.5]
#guard arrApproxEq (maskedCrossEntropyBwd #[0.5, 0.5, 0.5, 0.5] 2 2 #[0, 1] #[1, 0] 1.0) #[-0.5, 0.5, 0, 0]  -- masked row has zero gradient

/-!
===--------------------------------------------------------------------------===
RMS norm
===--------------------------------------------------------------------------===
-/

-- `ms`/`scale`/`y` order matches Python's `Value.__truediv__/__pow__/__mul__` for bit-exact parity. Cache stores `scale = 1/√(ms+ε)`, backward reconstructs from there.
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

theorem rmsnormFwd_size (x : Array Float) (rows : Nat) (cols : Nat) (eps : Float) : (rmsnormFwd x rows cols eps).1.size = rows * cols := by simp [rmsnormFwd, Id.run]; exact replicate_loop2_size ..
theorem rmsnormBwd_size (dy : Array Float) (x : Array Float) (scale : Array Float) (rows : Nat) (cols : Nat) : (rmsnormBwd dy x scale rows cols).size = rows * cols := by simp [rmsnormBwd, Id.run]; exact replicate_loop2_size ..

-- with `eps = 0` the normalized row has unit mean-square, and the scale is `(ms)^(-1/2)`
#guard let (y, _) := rmsnormFwd #[3, 4] 1 2 0.0; approxEq ((y[0]! * y[0]! + y[1]! * y[1]!) / 2.0) 1.0
#guard let (_, s) := rmsnormFwd #[3, 4] 1 2 0.0; approxEq s[0]! (Float.pow 12.5 (-0.5))
-- rmsnorm is scale-invariant, so the gradient along the input direction (`dy = x`, `eps = 0`) vanishes
#guard let (_, rms) := rmsnormFwd #[3, 4] 1 2 0.0; arrApproxEq (rmsnormBwd #[3, 4] #[3, 4] rms 1 2) #[0, 0]

/-!
===--------------------------------------------------------------------------===
Attention
===--------------------------------------------------------------------------===
-/

structure AttnConfig where
  nEmbed : Nat
  nHead : Nat
  epsilon : Float := 1e-5
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

def attnFwd (cfg : AttnConfig) (xPre : Array Float) (rows : Nat) (wq wk wv wo : Array Float) : Array Float × AttnCache :=
  let cols := cfg.nEmbed
  let (xn, rms) := rmsnormFwd xPre rows cols cfg.epsilon
  let qFlat := matmulFwd xn rows cols wq cols
  let kFlat := matmulFwd xn rows cols wk cols
  let vFlat := matmulFwd xn rows cols wv cols
  let qs := splitHeadsFlat qFlat rows cols cfg.nHead
  let ks := splitHeadsFlat kFlat rows cols cfg.nHead
  let vs := splitHeadsFlat vFlat rows cols cfg.nHead
  let headDim := cols / cfg.nHead
  let invScale := 1.0 / Float.pow headDim.toFloat 0.5
  let (aws, outs) : Array (Array Float) × Array (Array Float) := Id.run do
    let mut aws : Array (Array Float) := #[]
    let mut outs : Array (Array Float) := #[]
    for h in [0:cfg.nHead] do
      let q := qs[h]!; let k := ks[h]!; let v := vs[h]!
      let kT := transposeFlat k rows headDim
      let scores := matmulFwd q rows headDim kT rows
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
      outs := outs.push (matmulFwd aw rows rows v headDim)
    (aws, outs)
  let merged := mergeHeadsFlat outs rows cfg.nHead headDim
  let outFlat := matmulFwd merged rows cols wo cols
  let outRes := maddFlat xPre outFlat
  (outRes, { xPre := xPre, xPreRows := rows, xPreCols := cols, xn := xn, rms := rms, q := qs, k := ks, v := vs, attnW := aws, outFlat := merged })

def attnBwd (cfg : AttnConfig) (dout : Array Float) (rows : Nat) (wq wk wv wo : Array Float) (c : AttnCache) : Array Float × (Array Float × Array Float × Array Float × Array Float) :=
  let cols := cfg.nEmbed
  let dMerged := matmulBwdX dout rows cols wo cols
  let dWo := matmulBwdW dout rows cols c.outFlat cols
  let headDim := cols / cfg.nHead
  let dOutHeads := splitHeadsFlat dMerged rows cols cfg.nHead
  let invSqrt := 1.0 / Float.pow headDim.toFloat 0.5
  let (dqs, dks, dvs) : Array (Array Float) × Array (Array Float) × Array (Array Float) := Id.run do
    let mut dqs : Array (Array Float) := #[]
    let mut dks : Array (Array Float) := #[]
    let mut dvs : Array (Array Float) := #[]
    for h in [0:cfg.nHead] do
      let aw := c.attnW[h]!
      let q := c.q[h]!; let k := c.k[h]!; let v := c.v[h]!
      let dHead := dOutHeads[h]!
      let dawH := matmulBwdX dHead rows headDim v rows
      let dvH := matmulBwdW dHead rows headDim aw rows
      let dScaledH := softmaxRowsBwd aw dawH rows rows invSqrt
      let dScaledMasked : Array Float := Id.run do
        let mut acc := dScaledH
        for i in [0:rows] do
          for j in [0:rows] do
            if j > i then acc := acc.set! (i * rows + j) 0.0
        return acc
      let dqH := matmulFwd dScaledMasked rows rows k headDim
      let dkH := matmulFwd (transposeFlat dScaledMasked rows rows) rows rows q headDim
      dqs := dqs.push dqH
      dks := dks.push dkH
      dvs := dvs.push dvH
    (dqs, dks, dvs)
  let dQflat := mergeHeadsFlat dqs rows cfg.nHead headDim
  let dKflat := mergeHeadsFlat dks rows cfg.nHead headDim
  let dVflat := mergeHeadsFlat dvs rows cfg.nHead headDim
  let dXnQ := matmulBwdX dQflat rows cols wq cols
  let dXnK := matmulBwdX dKflat rows cols wk cols
  let dXnV := matmulBwdX dVflat rows cols wv cols
  let dXn := maddFlat (maddFlat dXnQ dXnK) dXnV
  let dWq := matmulBwdW dQflat rows cols c.xn cols
  let dWk := matmulBwdW dKflat rows cols c.xn cols
  let dWv := matmulBwdW dVflat rows cols c.xn cols
  let dxPre := maddFlat dout (rmsnormBwd dXn c.xPre c.rms rows cols)
  (dxPre, (dWq, dWk, dWv, dWo))

-- zero `wo` zeroes the attention contribution, so the residual passes the input through unchanged
#guard let cfg : AttnConfig := { nEmbed := 4, nHead := 2 }
       let x : Array Float := #[1, 2, 3, 4, 5, 6, 7, 8]
       let z : Array Float := Array.replicate 16 0.0
       let (out, _) := attnFwd cfg x 2 z z z z
       arrApproxEq out x
-- with zero weights the only surviving gradient path is the residual: `dxPre = dout`, weight grads are zero-sized-correct
#guard let cfg : AttnConfig := { nEmbed := 4, nHead := 2 }
       let x : Array Float := #[1, 2, 3, 4, 5, 6, 7, 8]
       let z : Array Float := Array.replicate 16 0.0
       let dout : Array Float := #[1, 1, 1, 1, 1, 1, 1, 1]
       let (_, c) := attnFwd cfg x 2 z z z z
       let (dxPre, (dWq, dWk, dWv, dWo)) := attnBwd cfg dout 2 z z z z c
       arrApproxEq dxPre dout && dWq.size == 16 && dWk.size == 16 && dWv.size == 16 && dWo.size == 16

/-!
===--------------------------------------------------------------------------===
MLP
===--------------------------------------------------------------------------===
-/

structure MlpConfig where
  nEmbed : Nat
  epsilon : Float := 1e-5
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

def mlpFwd (cfg : MlpConfig) (xPre : Array Float) (rows : Nat) (fc1 fc2 : Array Float) : Array Float × MlpCache :=
  let cols := cfg.nEmbed
  let hidden := 4 * cols
  let (xn, rms) := rmsnormFwd xPre rows cols cfg.epsilon
  let hPre := matmulFwd xn rows cols fc1 hidden
  let h := reluFlat hPre
  let y := matmulFwd h rows hidden fc2 cols
  (maddFlat xPre y, { xPre := xPre, xn := xn, rows := rows, cols := cols, rms := rms, hPre := hPre, h := h, hidden := hidden })

def mlpBwd (dout : Array Float) (fc1 fc2 : Array Float) (c : MlpCache) : Array Float × (Array Float × Array Float) :=
  let dh := matmulBwdX dout c.rows c.cols fc2 c.hidden
  let dfc2 := matmulBwdW dout c.rows c.cols c.h c.hidden
  let dhPre := reluBwdFlat dh c.hPre
  let dxn := matmulBwdX dhPre c.rows c.hidden fc1 c.cols
  let dfc1 := matmulBwdW dhPre c.rows c.hidden c.xn c.cols
  (maddFlat dout (rmsnormBwd dxn c.xPre c.rms c.rows c.cols), (dfc1, dfc2))

-- zero `fc2` zeroes the MLP branch, so the residual passes the input through unchanged
#guard let cfg : MlpConfig := { nEmbed := 4 }
       let x : Array Float := #[1, 2, 3, 4, 5, 6, 7, 8]
       let z : Array Float := Array.replicate 64 0.0
       let (out, _) := mlpFwd cfg x 2 z z
       arrApproxEq out x
-- zero weights leave only the residual gradient path: `dxPre = dout`
#guard let cfg : MlpConfig := { nEmbed := 4 }
       let x : Array Float := #[1, 2, 3, 4, 5, 6, 7, 8]
       let z : Array Float := Array.replicate 64 0.0
       let dout : Array Float := #[1, 1, 1, 1, 1, 1, 1, 1]
       let (_, c) := mlpFwd cfg x 2 z z
       let (dxPre, (df1, df2)) := mlpBwd dout z z c
       arrApproxEq dxPre dout && df1.size == 64 && df2.size == 64

end Autograd
