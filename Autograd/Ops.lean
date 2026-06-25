import Autograd.Utils

namespace Autograd

/-!
===--------------------------------------------------------------------------===
Matmul
===--------------------------------------------------------------------------===
-/

-- generic over `K`: runs on `Float`, proves on exact `ℚ` (Lean can't reason about floats).
def transposeFlat {K : Type} [Inhabited K] (x : Array K) (r : Nat) (c : Nat) : Array K :=
  (Array.range (c * r)).map fun k => x[(k % r) * c + (k / r)]!

-- forward: Y = X·W
def matmulFwd {K : Type} [Add K] [Mul K] [Zero K] [Inhabited K] (x : Array K) (n : Nat) (k : Nat) (W : Array K) (m : Nat) : Array K :=
  (Array.range (n * m)).map fun idx => (Array.range k).foldl (fun s kk => s + x[(idx / m) * k + kk]! * W[kk * m + (idx % m)]!) (0 : K)

-- backward to the input: dout·Wᵀ
def matmulBwdX {K : Type} [Add K] [Mul K] [Zero K] [Inhabited K] (dout : Array K) (n : Nat) (m : Nat) (W : Array K) (k : Nat) : Array K :=
  (Array.range (n * k)).map fun idx => (Array.range m).foldl (fun s j => s + dout[(idx / k) * m + j]! * W[(idx % k) * m + j]!) (0 : K)

-- backward to the weights: Xᵀ·dout
def matmulBwdW {K : Type} [Add K] [Mul K] [Zero K] [Inhabited K] (dout : Array K) (n : Nat) (m : Nat) (x : Array K) (k : Nat) : Array K :=
  (Array.range (k * m)).map fun idx => (Array.range n).foldl (fun s i => s + x[i * k + (idx / m)]! * dout[i * m + (idx % m)]!) (0 : K)

-- reading a freshly built `(range n).map f` at `k` gives `f k`
theorem map_range_getElem! {K : Type} [Inhabited K] (f : Nat → K) (n : Nat) (k : Nat) (hk : k < n) : ((Array.range n).map f)[k]! = f k := by rw [getElem!_pos _ k (by simp [hk]), Array.getElem_map, Array.getElem_range]

-- result entry (j,i) is input entry (i,j)
theorem transposeFlat_get (x : Array Float) (r : Nat) (c : Nat) (i : Nat) (j : Nat) (hi : i < r) (hj : j < c) : (transposeFlat x r c)[j * r + i]! = x[i * c + j]! := by
  have hr : 0 < r := Nat.lt_of_le_of_lt (Nat.zero_le i) hi
  have hbound : j * r + i < c * r := by
    have h2 : j * r + r = (j + 1) * r := by rw [Nat.add_mul, Nat.one_mul]
    have h3 : (j + 1) * r ≤ c * r := Nat.mul_le_mul_right r hj
    omega
  unfold transposeFlat
  rw [map_range_getElem! _ _ _ hbound]
  rw [show (j * r + i) % r = i by rw [Nat.mul_add_mod', Nat.mod_eq_of_lt hi], show (j * r + i) / r = j by rw [Nat.mul_comm j r, Nat.mul_add_div hr, Nat.div_eq_of_lt hi, Nat.add_zero]]

-- transpose is its own inverse
theorem transposeFlat_involution (x : Array Float) (r : Nat) (c : Nat) (h : x.size = r * c) : transposeFlat (transposeFlat x r c) c r = x := by
  apply Array.ext
  · simp [transposeFlat, h]
  · intro k hk1 hk2
    have hkrc : k < r * c := by simpa [transposeFlat] using hk1
    have hc : 0 < c := by rcases Nat.eq_zero_or_pos c with rfl | hc; simp at hkrc; exact hc
    have hkc : k % c < c := Nat.mod_lt _ hc
    have hkdc : k / c < r := by apply Nat.div_lt_of_lt_mul; rwa [Nat.mul_comm] at hkrc
    rw [← getElem!_pos _ k hk1, ← getElem!_pos x k hk2]
    rw [show k = (k / c) * c + (k % c) by rw [Nat.mul_comm (k / c) c, Nat.div_add_mod]]
    rw [transposeFlat_get _ c r (k % c) (k / c) hkc hkdc]
    rw [transposeFlat_get x r c (k / c) (k % c) hkdc hkc]

-- tests
theorem transposeFlat_size {K : Type} [Inhabited K] (x : Array K) (r : Nat) (c : Nat) : (transposeFlat x r c).size = c * r := by simp [transposeFlat]
theorem matmulFwd_size {K : Type} [Add K] [Mul K] [Zero K] [Inhabited K] (x : Array K) (n : Nat) (k : Nat) (W : Array K) (m : Nat) : (matmulFwd x n k W m).size = n * m := by simp [matmulFwd]
theorem matmulBwdX_size {K : Type} [Add K] [Mul K] [Zero K] [Inhabited K] (dout : Array K) (n : Nat) (m : Nat) (W : Array K) (k : Nat) : (matmulBwdX dout n m W k).size = n * k := by simp [matmulBwdX]
theorem matmulBwdW_size {K : Type} [Add K] [Mul K] [Zero K] [Inhabited K] (dout : Array K) (n : Nat) (m : Nat) (x : Array K) (k : Nat) : (matmulBwdW dout n m x k).size = k * m := by simp [matmulBwdW]
#guard arrApproxEq (transposeFlat #[1, 2, 3, 4, 5, 6] 2 3) #[1, 4, 2, 5, 3, 6]
#guard arrApproxEq (transposeFlat (transposeFlat #[1, 2, 3, 4, 5, 6] 2 3) 3 2) #[1, 2, 3, 4, 5, 6]
#guard arrApproxEq (matmulFwd #[1, 2, 3, 4] 2 2 #[1, 2, 3, 4] 2) #[7, 10, 15, 22]
#guard arrApproxEq (matmulFwd #[1, 2, 3, 4, 5, 6] 2 3 #[1, 2, 3, 4, 5, 6] 2) #[22, 28, 49, 64]
#guard arrApproxEq (matmulBwdX #[1, 2, 3, 4] 2 2 #[1, 0, 0, 1] 2) #[1, 2, 3, 4]
#guard arrApproxEq (matmulBwdW #[1, 2, 3, 4] 2 2 #[1, 0, 0, 1] 2) #[1, 2, 3, 4]

/-!
===--------------------------------------------------------------------------===
Element-wise
===--------------------------------------------------------------------------===
-/

def maddFlat {K : Type} [Add K] [Inhabited K] (a : Array K) (b : Array K) : Array K :=
  (Array.range a.size).map fun i => a[i]! + b[i]!

private def reluFlat (x : Array Float) : Array Float := x.map fun z => if z > 0.0 then z else 0.0

private def reluBwdFlat (dout hPre : Array Float) : Array Float :=
  (Array.range dout.size).map fun i => if hPre[i]! > 0.0 then dout[i]! else 0.0

-- tests
theorem maddFlat_size {K : Type} [Add K] [Inhabited K] (a : Array K) (b : Array K) : (maddFlat a b).size = a.size := by simp [maddFlat]
theorem reluFlat_size (x : Array Float) : (reluFlat x).size = x.size := by simp [reluFlat]
theorem reluBwdFlat_size (dout : Array Float) (hPre : Array Float) : (reluBwdFlat dout hPre).size = dout.size := by simp [reluBwdFlat]
#guard arrApproxEq (maddFlat #[1, 2, 3] #[3, 4, 5]) #[4, 6, 8]
#guard arrApproxEq (reluFlat #[-1, 0, 2, -3]) #[0, 0, 2, 0]
#guard arrApproxEq (reluBwdFlat #[1, 1, 1, 1] #[-1, 0, 2, -3]) #[0, 0, 1, 0]

/-!
===--------------------------------------------------------------------------===
Softmax
===--------------------------------------------------------------------------===
-/

-- `e * (1/s)` not `e / s`: matches Python `__truediv__`.
private def softmaxFlat (v : Array Float) : Array Float :=
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

private def softmaxRowsBwd (aw daw : Array Float) (rows cols : Nat) (scale : Float) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.replicate (rows * cols) 0.0
    for i in [0:rows] do
      let mut dot : Float := 0.0
      for j in [0:cols] do dot := dot + aw[i * cols + j]! * daw[i * cols + j]!
      for j in [0:cols] do
        let g := scale * aw[i * cols + j]! * (daw[i * cols + j]! - dot)
        out := out.set! (i * cols + j) g
    return out

theorem foldl_size_pres {α : Type} {K : Type} (l : List α) (init : Array K) (f : Array K → α → Array K) (hf : ∀ b a, (f b a).size = b.size) : (l.foldl f init).size = init.size := by
  induction l generalizing init with
  | nil => rfl
  | cons x xs ih => rw [List.foldl_cons, ih, hf]

theorem replicate_loop2_size {α : Type} {β : Type} {K : Type} (outer : List α) (inner : α → List β) (n : Nat) (v : K) (g : Array K → α → β → Nat) (h : Array K → α → β → K) : (outer.foldl (fun b a => (inner a).foldl (fun b' c => b'.setIfInBounds (g b' a c) (h b' a c)) b) (Array.replicate n v)).size = n := (foldl_size_pres _ _ _ (fun _ _ => foldl_size_pres _ _ _ (fun _ _ => Array.size_setIfInBounds ..))).trans (Array.size_replicate ..)

-- tests
theorem softmaxFlat_size (v : Array Float) : (softmaxFlat v).size = v.size := by unfold softmaxFlat; split <;> simp
theorem softmaxRows_size (x : Array Float) (rows : Nat) (cols : Nat) : (softmaxRows x rows cols).size = rows * cols := by simp [softmaxRows, Id.run]; exact replicate_loop2_size ..
theorem softmaxRowsBwd_size (aw : Array Float) (daw : Array Float) (rows : Nat) (cols : Nat) (scale : Float) : (softmaxRowsBwd aw daw rows cols scale).size = rows * cols := by simp [softmaxRowsBwd, Id.run]; exact replicate_loop2_size ..
#guard arrApproxEq (softmaxFlat #[0, 0, 0]) #[1.0 / 3, 1.0 / 3, 1.0 / 3]
#guard approxEq ((softmaxFlat #[1, 2, 3]).foldl (· + ·) 0.0) 1.0
#guard arrApproxEq (softmaxFlat #[1, 2, 3]) (softmaxFlat #[-4, -3, -2])
#guard let sm := softmaxRows #[1, 2, 1, 0] 2 2; approxEq (sm[0]! + sm[1]!) 1.0 && approxEq (sm[2]! + sm[3]!) 1.0
#guard arrApproxEq (softmaxRowsBwd (softmaxRows #[1, 2, 3, 4] 1 4) #[5, 5, 5, 5] 1 4 1.0) #[0, 0, 0, 0]

/-!
===--------------------------------------------------------------------------===
Multi-head
===--------------------------------------------------------------------------===
-/

private def splitHeadsFlat (x : Array Float) (n dModel nHead : Nat) : Array (Array Float) :=
  let headDim := dModel / nHead
  (Array.range nHead).map fun h =>
    Id.run do
      let mut acc : Array Float := Array.replicate (n * headDim) 0.0
      for i in [0:n] do
        for j in [0:headDim] do acc := acc.set! (i * headDim + j) x[i * dModel + h * headDim + j]!
      return acc

private def mergeHeadsFlat (xs : Array (Array Float)) (n nHead headDim : Nat) : Array Float :=
  let dModel := nHead * headDim
  Id.run do
    let mut out : Array Float := Array.replicate (n * dModel) 0.0
    for i in [0:n] do
      for col in [0:dModel] do
        let h := col / headDim; let j := col % headDim
        out := out.set! (i * dModel + col) xs[h]![i * headDim + j]!
    return out

-- tests
theorem splitHeadsFlat_count (x : Array Float) (n : Nat) (dModel : Nat) (nHead : Nat) : (splitHeadsFlat x n dModel nHead).size = nHead := by simp [splitHeadsFlat]
theorem mergeHeadsFlat_size (xs : Array (Array Float)) (n : Nat) (nHead : Nat) (headDim : Nat) : (mergeHeadsFlat xs n nHead headDim).size = n * (nHead * headDim) := by simp [mergeHeadsFlat, Id.run]; exact replicate_loop2_size ..
#guard let hs := splitHeadsFlat #[1, 2, 3, 4] 1 4 2; hs.size == 2 && arrApproxEq hs[0]! #[1, 2] && arrApproxEq hs[1]! #[3, 4]
#guard arrApproxEq (mergeHeadsFlat (splitHeadsFlat #[1, 2, 3, 4, 5, 6, 7, 8] 2 4 2) 2 2 2) #[1, 2, 3, 4, 5, 6, 7, 8]

/-!
===--------------------------------------------------------------------------===
Embedding scatter
===--------------------------------------------------------------------------===
-/

def gatherFlat {K : Type} [Inhabited K] (table : Array K) (cols : Nat) (ids : Array Nat) : Array K :=
  (Array.range (ids.size * cols)).map fun k => table[ids[k / cols]! * cols + k % cols]!

def scatterAddFlat {K : Type} [Add K] [Zero K] [Inhabited K] (rows : Nat) (cols : Nat) (grad : Array K) (ids : Array Nat) : Array K :=
  Id.run do
    let mut out : Array K := Array.replicate (rows * cols) (0 : K)
    for i in [0:ids.size] do
      let id := ids[i]!
      for j in [0:cols] do
        out := out.set! (id * cols + j) (out[id * cols + j]! + grad[i * cols + j]!)
    return out

theorem nat_range_getElem! (n : Nat) (i : Nat) (hi : i < n) : (Array.range n)[i]! = i := by rw [getElem!_pos _ i (by simp [hi]), Array.getElem_range]

theorem gatherFlat_get (table : Array Float) (cols : Nat) (ids : Array Nat) (row : Nat) (col : Nat) (hrow : row < ids.size) (hcol : col < cols) : (gatherFlat table cols ids)[row * cols + col]! = table[ids[row]! * cols + col]! := by
  have hbound : row * cols + col < ids.size * cols := by
    have h2 : row * cols + cols = (row + 1) * cols := by rw [Nat.add_mul, Nat.one_mul]
    have h3 : (row + 1) * cols ≤ ids.size * cols := Nat.mul_le_mul_right cols hrow
    omega
  unfold gatherFlat
  rw [map_range_getElem! _ _ _ hbound]
  rw [show (row * cols + col) % cols = col by rw [Nat.mul_add_mod', Nat.mod_eq_of_lt hcol], show (row * cols + col) / cols = row by rw [Nat.mul_comm row cols, Nat.mul_add_div (Nat.lt_of_le_of_lt (Nat.zero_le col) hcol), Nat.div_eq_of_lt hcol, Nat.add_zero]]

theorem gatherFlat_identity (table : Array Float) (rows : Nat) (cols : Nat) (h : table.size = rows * cols) : gatherFlat table cols (Array.range rows) = table := by
  apply Array.ext
  · simp [gatherFlat, h]
  · intro k hk1 hk2
    have hkrc : k < rows * cols := by simpa [gatherFlat] using hk1
    have hc : 0 < cols := by rcases Nat.eq_zero_or_pos cols with rfl | hc; simp at hkrc; exact hc
    have hkdc : k / cols < rows := by apply Nat.div_lt_of_lt_mul; rwa [Nat.mul_comm] at hkrc
    have hkc : k % cols < cols := Nat.mod_lt _ hc
    rw [← getElem!_pos _ k hk1, ← getElem!_pos table k hk2]
    rw [show k = (k / cols) * cols + (k % cols) by rw [Nat.mul_comm (k / cols) cols, Nat.div_add_mod]]
    rw [gatherFlat_get table cols (Array.range rows) (k / cols) (k % cols) (by rw [Array.size_range]; exact hkdc) hkc]
    rw [nat_range_getElem! _ _ hkdc]

-- tests
theorem gatherFlat_size (table : Array Float) (cols : Nat) (ids : Array Nat) : (gatherFlat table cols ids).size = ids.size * cols := by simp [gatherFlat]
theorem scatterAddFlat_size (rows : Nat) (cols : Nat) (grad : Array Float) (ids : Array Nat) : (scatterAddFlat rows cols grad ids).size = rows * cols := by simp [scatterAddFlat, Id.run]; exact replicate_loop2_size ..
#guard arrApproxEq (gatherFlat #[10, 11, 20, 21, 30, 31] 2 #[2, 0]) #[30, 31, 10, 11]
#guard arrApproxEq (scatterAddFlat 3 2 #[1, 1, 2, 2, 3, 3] #[0, 0, 2]) #[3, 3, 0, 0, 3, 3]

/-!
===--------------------------------------------------------------------------===
Cross-entropy
===--------------------------------------------------------------------------===
-/

-- reciprocal-first `(1/n) * sum(losses)` matches Python.
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

-- tests
theorem maskedCrossEntropyBwd_size (probs : Array Float) (rows : Nat) (cols : Nat) (t : Array Nat) (mask : Array Float) (sumMask : Float) : (maskedCrossEntropyBwd probs rows cols t mask sumMask).size = rows * cols := by simp [maskedCrossEntropyBwd, Id.run]; exact replicate_loop2_size ..
#guard approxEq (maskedCrossEntropy #[0.5, 0.5] 1 2 #[0] #[1] 1.0) (-Float.log 0.5)
#guard approxEq (maskedCrossEntropy #[0.5, 0.5] 1 2 #[0] #[0] 0.0) 0.0
#guard approxEq (maskedCrossEntropy #[0.5, 0.5, 0.5, 0.5] 2 2 #[0, 1] #[1, 0] 1.0) (-Float.log 0.5)
#guard arrApproxEq (maskedCrossEntropyBwd #[0.5, 0.5] 1 2 #[0] #[1] 1.0) #[-0.5, 0.5]
#guard arrApproxEq (maskedCrossEntropyBwd #[0.5, 0.5, 0.5, 0.5] 2 2 #[0, 1] #[1, 0] 1.0) #[-0.5, 0.5, 0, 0]

/-!
===--------------------------------------------------------------------------===
RMS norm
===--------------------------------------------------------------------------===
-/

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

-- tests
theorem rmsnormFwd_size (x : Array Float) (rows : Nat) (cols : Nat) (eps : Float) : (rmsnormFwd x rows cols eps).1.size = rows * cols := by simp [rmsnormFwd, Id.run]; exact replicate_loop2_size ..
theorem rmsnormBwd_size (dy : Array Float) (x : Array Float) (scale : Array Float) (rows : Nat) (cols : Nat) : (rmsnormBwd dy x scale rows cols).size = rows * cols := by simp [rmsnormBwd, Id.run]; exact replicate_loop2_size ..
#guard let (y, _) := rmsnormFwd #[3, 4] 1 2 0.0; approxEq ((y[0]! * y[0]! + y[1]! * y[1]!) / 2.0) 1.0
#guard let (_, s) := rmsnormFwd #[3, 4] 1 2 0.0; approxEq s[0]! (Float.pow 12.5 (-0.5))
#guard let (_, rms) := rmsnormFwd #[3, 4] 1 2 0.0; arrApproxEq (rmsnormBwd #[3, 4] #[3, 4] rms 1 2) #[0, 0]

/-!
===--------------------------------------------------------------------------===
Attention
===--------------------------------------------------------------------------===
-/

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

def attnFwd (nEmbed nHead : Nat) (xPre : Array Float) (rows : Nat) (wq wk wv wo : Array Float) (epsilon : Float := 1e-5) (maskValue : Float := -1.0e9) : Array Float × AttnCache :=
  let cols := nEmbed
  let (xn, rms) := rmsnormFwd xPre rows cols epsilon
  let qFlat := matmulFwd xn rows cols wq cols
  let kFlat := matmulFwd xn rows cols wk cols
  let vFlat := matmulFwd xn rows cols wv cols
  let qs := splitHeadsFlat qFlat rows cols nHead
  let ks := splitHeadsFlat kFlat rows cols nHead
  let vs := splitHeadsFlat vFlat rows cols nHead
  let headDim := cols / nHead
  let invScale := 1.0 / Float.pow headDim.toFloat 0.5
  let (aws, outs) : Array (Array Float) × Array (Array Float) := Id.run do
    let mut aws : Array (Array Float) := #[]
    let mut outs : Array (Array Float) := #[]
    for h in [0:nHead] do
      let q := qs[h]!; let k := ks[h]!; let v := vs[h]!
      let kT := transposeFlat k rows headDim
      let scores := matmulFwd q rows headDim kT rows
      let scaled : Array Float := scores.map (· * invScale)
      let masked : Array Float := Id.run do
        let mut acc : Array Float := Array.replicate (rows * rows) 0.0
        for i in [0:rows] do
          for j in [0:rows] do
            let v := if j ≤ i then scaled[i * rows + j]! else maskValue
            acc := acc.set! (i * rows + j) v
        return acc
      let aw := softmaxRows masked rows rows
      aws := aws.push aw
      outs := outs.push (matmulFwd aw rows rows v headDim)
    return (aws, outs)
  let merged := mergeHeadsFlat outs rows nHead headDim
  let outFlat := matmulFwd merged rows cols wo cols
  let outRes := maddFlat xPre outFlat
  (outRes, { xPre := xPre, xPreRows := rows, xPreCols := cols, xn := xn, rms := rms, q := qs, k := ks, v := vs, attnW := aws, outFlat := merged })

def attnBwd (nEmbed nHead : Nat) (dout : Array Float) (rows : Nat) (wq wk wv wo : Array Float) (c : AttnCache) : Array Float × (Array Float × Array Float × Array Float × Array Float) :=
  let cols := nEmbed
  let dMerged := matmulBwdX dout rows cols wo cols
  let dWo := matmulBwdW dout rows cols c.outFlat cols
  let headDim := cols / nHead
  let dOutHeads := splitHeadsFlat dMerged rows cols nHead
  let invSqrt := 1.0 / Float.pow headDim.toFloat 0.5
  let (dqs, dks, dvs) : Array (Array Float) × Array (Array Float) × Array (Array Float) := Id.run do
    let mut dqs : Array (Array Float) := #[]
    let mut dks : Array (Array Float) := #[]
    let mut dvs : Array (Array Float) := #[]
    for h in [0:nHead] do
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
    return (dqs, dks, dvs)
  let dQflat := mergeHeadsFlat dqs rows nHead headDim
  let dKflat := mergeHeadsFlat dks rows nHead headDim
  let dVflat := mergeHeadsFlat dvs rows nHead headDim
  let dXnQ := matmulBwdX dQflat rows cols wq cols
  let dXnK := matmulBwdX dKflat rows cols wk cols
  let dXnV := matmulBwdX dVflat rows cols wv cols
  let dXn := maddFlat (maddFlat dXnQ dXnK) dXnV
  let dWq := matmulBwdW dQflat rows cols c.xn cols
  let dWk := matmulBwdW dKflat rows cols c.xn cols
  let dWv := matmulBwdW dVflat rows cols c.xn cols
  let dxPre := maddFlat dout (rmsnormBwd dXn c.xPre c.rms rows cols)
  (dxPre, (dWq, dWk, dWv, dWo))

-- tests
#guard let x : Array Float := #[1, 2, 3, 4, 5, 6, 7, 8]
       let z : Array Float := Array.replicate 16 0.0
       let (out, _) := attnFwd 4 2 x 2 z z z z
       arrApproxEq out x
#guard let x : Array Float := #[1, 2, 3, 4, 5, 6, 7, 8]
       let z : Array Float := Array.replicate 16 0.0
       let dout : Array Float := #[1, 1, 1, 1, 1, 1, 1, 1]
       let (_, c) := attnFwd 4 2 x 2 z z z z
       let (dxPre, (dWq, dWk, dWv, dWo)) := attnBwd 4 2 dout 2 z z z z c
       arrApproxEq dxPre dout && dWq.size == 16 && dWk.size == 16 && dWv.size == 16 && dWo.size == 16

/-!
===--------------------------------------------------------------------------===
MLP
===--------------------------------------------------------------------------===
-/

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

def mlpFwd (nEmbed : Nat) (xPre : Array Float) (rows : Nat) (fc1 fc2 : Array Float) (epsilon : Float := 1e-5) : Array Float × MlpCache :=
  let cols := nEmbed
  let hidden := 4 * cols
  let (xn, rms) := rmsnormFwd xPre rows cols epsilon
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

-- tests
#guard let x : Array Float := #[1, 2, 3, 4, 5, 6, 7, 8]
       let z : Array Float := Array.replicate 64 0.0
       let (out, _) := mlpFwd 4 x 2 z z
       arrApproxEq out x
#guard let x : Array Float := #[1, 2, 3, 4, 5, 6, 7, 8]
       let z : Array Float := Array.replicate 64 0.0
       let dout : Array Float := #[1, 1, 1, 1, 1, 1, 1, 1]
       let (_, c) := mlpFwd 4 x 2 z z
       let (dxPre, (df1, df2)) := mlpBwd dout z z c
       arrApproxEq dxPre dout && df1.size == 64 && df2.size == 64

end Autograd
