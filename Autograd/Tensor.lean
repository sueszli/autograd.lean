import Autograd.Ops
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped Matrix

namespace Autograd

/-!
===--------------------------------------------------------------------------===
Types

Mirrors the C struct from https://github.com/sueszli/autograd.c

  typedef struct Tensor {
    float32_t *data;     // flat contiguous array, row-major
    uint64_t *shape;     // array of dimension sizes
    uint64_t *strides;   // array of elements to skip to get to next element in each dimension
    uint64_t ndim;       // rank (ie. 1 for vector, 2 for matrix, etc.)
    uint64_t size;       // total number of elements

    bool requires_grad;  // whether to track operations for autograd
    struct Tensor *grad; // accumulated gradient (del loss / del tensor) during backprop
    Function *grad_fn;   // function that created this tensor (NULL for leaves)
    uint32_t ref_count;  // reference count for memory management
  } Tensor;
===--------------------------------------------------------------------------===
-/

mutual
structure Tensor where
  data : Array Float
  shape : Array Nat
  id : Nat
  requiresGrad : Bool
  gradFn : GradFn

inductive GradFn where
  | leaf
  | addOp     (a b : Tensor)
  | matmulOp  (x w : Tensor)
  | gatherOp  (table : Tensor) (ids : Array Nat)
  | rmsnormOp (a : Tensor) (rms : Array Float)
  | lossOp    (logits : Tensor) (probs : Array Float) (targets : Array Nat) (mask : Array Float) (sumMask : Float)
  | attnOp    (xPre wq wk wv wo : Tensor) (cache : AttnCache) (nEmbed nHead : Nat)
  | mlpOp     (xPre fc1 fc2 : Tensor) (cache : MlpCache) (nEmbed : Nat)
end

instance : Inhabited Tensor := ⟨{ data := #[], shape := #[], id := 0, requiresGrad := false, gradFn := .leaf }⟩
instance : Inhabited GradFn := ⟨.leaf⟩

namespace Tensor

def rows (t : Tensor) : Nat := if t.shape.size = 0 then 0 else t.shape[0]!
def cols (t : Tensor) : Nat := if t.shape.size < 2 then 1 else t.shape[1]!
def leaf (data : Array Float) (rows cols id : Nat) (requiresGrad : Bool) : Tensor := { data := data, shape := #[rows, cols], id := id, requiresGrad := requiresGrad, gradFn := .leaf }

-- tests
theorem leaf_rows (data : Array Float) (r : Nat) (c : Nat) (id : Nat) (rg : Bool) : (Tensor.leaf data r c id rg).rows = r := rfl
theorem leaf_cols (data : Array Float) (r : Nat) (c : Nat) (id : Nat) (rg : Bool) : (Tensor.leaf data r c id rg).cols = c := rfl
theorem leaf_data (data : Array Float) (r : Nat) (c : Nat) (id : Nat) (rg : Bool) : (Tensor.leaf data r c id rg).data = data := rfl
theorem leaf_id (data : Array Float) (r : Nat) (c : Nat) (id : Nat) (rg : Bool) : (Tensor.leaf data r c id rg).id = id := rfl
theorem leaf_requiresGrad (data : Array Float) (r : Nat) (c : Nat) (id : Nat) (rg : Bool) : (Tensor.leaf data r c id rg).requiresGrad = rg := rfl
theorem default_data_empty : (default : Tensor).data.size = 0 := rfl
theorem default_no_grad : (default : Tensor).requiresGrad = false := rfl
theorem default_gradfn_leaf : (default : GradFn) = .leaf := rfl
theorem cols_rank1 (t : Tensor) (h : t.shape.size < 2) : t.cols = 1 := by unfold cols; exact if_pos h

/-!
===--------------------------------------------------------------------------===
Forward ops

Ops implementations live in `Autograd/Ops.lean`.
===--------------------------------------------------------------------------===
-/

def gather (table : Tensor) (ids : Array Nat) : Tensor :=
  { data := gatherFlat table.data table.cols ids, shape := #[ids.size, table.cols], id := 0, requiresGrad := table.requiresGrad, gradFn := .gatherOp table ids }

instance : Add Tensor := ⟨fun a b => { data := maddFlat a.data b.data, shape := a.shape, id := 0, requiresGrad := a.requiresGrad || b.requiresGrad, gradFn := .addOp a b }⟩

private def matmul (x w : Tensor) : Tensor :=
  let n := x.rows; let k := x.cols; let m := w.cols
  { data := matmulFwd x.data n k w.data m, shape := #[n, m], id := 0, requiresGrad := x.requiresGrad || w.requiresGrad, gradFn := .matmulOp x w }

infixl:70 " @ " => Tensor.matmul

def rmsnorm (a : Tensor) (eps : Float) : Tensor :=
  let (y, rms) := rmsnormFwd a.data a.rows a.cols eps
  { data := y, shape := a.shape, id := 0, requiresGrad := a.requiresGrad, gradFn := .rmsnormOp a rms }

def attn (nEmbed nHead : Nat) (epsilon maskValue : Float) (xPre wq wk wv wo : Tensor) : Tensor :=
  let (out, cache) := attnFwd nEmbed nHead xPre.data xPre.rows wq.data wk.data wv.data wo.data epsilon maskValue
  { data := out, shape := xPre.shape, id := 0, requiresGrad := xPre.requiresGrad || wq.requiresGrad || wk.requiresGrad || wv.requiresGrad || wo.requiresGrad, gradFn := .attnOp xPre wq wk wv wo cache nEmbed nHead }

def mlp (nEmbed : Nat) (epsilon : Float) (xPre fc1 fc2 : Tensor) : Tensor :=
  let (out, cache) := mlpFwd nEmbed xPre.data xPre.rows fc1.data fc2.data epsilon
  { data := out, shape := xPre.shape, id := 0, requiresGrad := xPre.requiresGrad || fc1.requiresGrad || fc2.requiresGrad, gradFn := .mlpOp xPre fc1 fc2 cache nEmbed }

def maskedCE (logits : Tensor) (targets : Array Nat) (mask : Array Float) : Tensor :=
  let probs := softmaxRows logits.data logits.rows logits.cols
  let sumMask := mask.foldl (init := 0.0) (· + ·)
  let l := maskedCrossEntropy probs logits.rows logits.cols targets mask sumMask
  { data := #[l], shape := #[1, 1], id := 0, requiresGrad := logits.requiresGrad, gradFn := .lossOp logits probs targets mask sumMask }

private def oneHotRows {rows : Nat} (ids : Array Nat) : Matrix (Fin ids.size) (Fin rows) ℝ := Matrix.of fun r j => if ids[r.1]'r.2 = (j : Nat) then 1 else 0  -- gather pick matrix
private def toMat {K : Type} [Inhabited K] (a : Array K) (rows : Nat) (cols : Nat) : Matrix (Fin rows) (Fin cols) K := Matrix.of fun i j => a[i.1 * cols + j.1]!  -- reshape flat array
-- nonlinear ops: prove the backward rule equals the real derivative (Mathlib calculus), since the adjoint
-- identity is not the gradient here. foundation: `∂ lse / ∂z_i = softmax_i`.
private noncomputable def lse {c : Nat} (z : Fin c → ℝ) : ℝ := Real.log (∑ j, Real.exp (z j))
private noncomputable def softmaxReal {c : Nat} (z : Fin c → ℝ) (i : Fin c) : ℝ := Real.exp (z i) / ∑ j, Real.exp (z j)
private noncomputable def ceLoss {c : Nat} (z : Fin c → ℝ) (t : Fin c) : ℝ := lse z - z t

-- add's backward sends the cotangent unchanged to both summands
theorem add_adjoint {n : Nat} {m : Nat} (A : Matrix (Fin n) (Fin m) ℝ) (B : Matrix (Fin n) (Fin m) ℝ) (G : Matrix (Fin n) (Fin m) ℝ)
    : Matrix.trace (Gᵀ * (A + B)) = Matrix.trace (Gᵀ * A) + Matrix.trace (Gᵀ * B) := by rw [Matrix.mul_add, Matrix.trace_add]
-- `G ↦ G·Wᵀ` is the adjoint of `X ↦ X·W`
theorem matmulBwdX_adjoint {n : Nat} {k : Nat} {m : Nat} (X : Matrix (Fin n) (Fin k) ℝ) (W : Matrix (Fin k) (Fin m) ℝ) (G : Matrix (Fin n) (Fin m) ℝ)
    : Matrix.trace (Gᵀ * (X * W)) = Matrix.trace ((G * Wᵀ)ᵀ * X) := by rw [Matrix.transpose_mul, Matrix.transpose_transpose, ← Matrix.mul_assoc, Matrix.trace_mul_comm, ← Matrix.mul_assoc]
-- `G ↦ Xᵀ·G` is the adjoint of `W ↦ X·W`
theorem matmulBwdW_adjoint {n : Nat} {k : Nat} {m : Nat} (X : Matrix (Fin n) (Fin k) ℝ) (W : Matrix (Fin k) (Fin m) ℝ) (G : Matrix (Fin n) (Fin m) ℝ)
    : Matrix.trace (Gᵀ * (X * W)) = Matrix.trace ((Xᵀ * G)ᵀ * W) := by rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
-- transpose is its own adjoint
theorem transposeFlat_adjoint {n : Nat} {m : Nat} (X : Matrix (Fin m) (Fin n) ℝ) (G : Matrix (Fin n) (Fin m) ℝ)
    : Matrix.trace (Gᵀ * Xᵀ) = Matrix.trace (Gᵀᵀ * X) := by rw [Matrix.transpose_transpose, ← Matrix.transpose_mul, Matrix.trace_transpose, Matrix.trace_mul_comm]
-- gather's adjoint is scatter, left-multiplication by the 0/1 one-hot matrix
theorem gather_adjoint {rows : Nat} {cols : Nat} (ids : Array Nat) (table : Matrix (Fin rows) (Fin cols) ℝ) (G : Matrix (Fin ids.size) (Fin cols) ℝ)
    : Matrix.trace (Gᵀ * (oneHotRows (rows := rows) ids * table)) = Matrix.trace (((oneHotRows (rows := rows) ids)ᵀ * G)ᵀ * table) := matmulBwdW_adjoint (oneHotRows (rows := rows) ids) table G
-- the scattered entry sums the gradients of all rows that selected it
theorem scatter_apply {rows : Nat} {cols : Nat} (ids : Array Nat) (G : Matrix (Fin ids.size) (Fin cols) ℝ) (j : Fin rows) (c : Fin cols)
    : ((oneHotRows (rows := rows) ids)ᵀ * G) j c = ∑ r : Fin ids.size, (if ids[r.1]'r.2 = (j : Nat) then G r c else 0) := by simp [oneHotRows, Matrix.mul_apply, Matrix.transpose_apply, Matrix.of_apply]
-- gradient of a linear layer `⟪T, X·W+B⟫`: `dW = Xᵀ·T`, `dB = T`
theorem linearLayer_grad {n : Nat} {k : Nat} {m : Nat} (X : Matrix (Fin n) (Fin k) ℝ) (W : Matrix (Fin k) (Fin m) ℝ) (B : Matrix (Fin n) (Fin m) ℝ) (T : Matrix (Fin n) (Fin m) ℝ)
    : Matrix.trace (Tᵀ * (X * W + B)) = Matrix.trace ((Xᵀ * T)ᵀ * W) + Matrix.trace (Tᵀ * B) := by rw [add_adjoint, matmulBwdW_adjoint]

-- the float left-fold equals the mathematical `Finset.sum`
theorem foldl_range_sum {K : Type} [AddCommMonoid K] (n : Nat) (f : Nat → K)
    : (Array.range n).foldl (fun s i => s + f i) 0 = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Array.range_succ, Array.foldl_append, ih, Finset.sum_range_succ]; simp

-- a row-major index `a*q+b` stays in bounds (`< p*q`)
theorem idx_bound (a : Nat) (b : Nat) (p : Nat) (q : Nat) (ha : a < p) (hb : b < q)
    : a * q + b < p * q := by nlinarith

-- the flat `matmulFwd` kernel equals real `Matrix.mul`
theorem matmulFwd_bridge {K : Type} [CommRing K] [Inhabited K] (x : Array K) (W : Array K) (n : Nat) (k : Nat) (m : Nat)
    : toMat (matmulFwd x n k W m) n m = toMat x n k * toMat W k m := by
  ext i j
  rw [Matrix.mul_apply]
  simp only [toMat, Matrix.of_apply]
  have hb : i.1 * m + j.1 < n * m := idx_bound i.1 j.1 n m i.2 j.2
  have hmod : (i.1 * m + j.1) % m = j.1 := by rw [Nat.mul_add_mod', Nat.mod_eq_of_lt j.2]
  have hdiv : (i.1 * m + j.1) / m = i.1 := by rw [Nat.mul_comm i.1 m, Nat.mul_add_div (Nat.lt_of_le_of_lt (Nat.zero_le j.1) j.2), Nat.div_eq_of_lt j.2, Nat.add_zero]
  rw [matmulFwd, map_range_getElem! _ _ _ hb]
  simp only [hdiv, hmod]
  rw [foldl_range_sum, Fin.sum_univ_eq_sum_range (fun p => x[i.1 * k + p]! * W[p * m + j.1]!) k]

-- `matmulBwdX` equals `dout · Wᵀ` as matrices
theorem matmulBwdX_bridge {K : Type} [CommRing K] [Inhabited K] (dout : Array K) (W : Array K) (n : Nat) (m : Nat) (k : Nat)
    : toMat (matmulBwdX dout n m W k) n k = toMat dout n m * (toMat W k m)ᵀ := by
  ext i kk
  rw [Matrix.mul_apply]
  simp only [toMat, Matrix.of_apply, Matrix.transpose_apply]
  have hb : i.1 * k + kk.1 < n * k := idx_bound i.1 kk.1 n k i.2 kk.2
  have hmod : (i.1 * k + kk.1) % k = kk.1 := by rw [Nat.mul_add_mod', Nat.mod_eq_of_lt kk.2]
  have hdiv : (i.1 * k + kk.1) / k = i.1 := by rw [Nat.mul_comm i.1 k, Nat.mul_add_div (Nat.lt_of_le_of_lt (Nat.zero_le kk.1) kk.2), Nat.div_eq_of_lt kk.2, Nat.add_zero]
  rw [matmulBwdX, map_range_getElem! _ _ _ hb]
  simp only [hdiv, hmod]
  rw [foldl_range_sum, Fin.sum_univ_eq_sum_range (fun p => dout[i.1 * m + p]! * W[kk.1 * m + p]!) m]

-- `matmulBwdW` equals `Xᵀ · dout` as matrices
theorem matmulBwdW_bridge {K : Type} [CommRing K] [Inhabited K] (dout : Array K) (x : Array K) (n : Nat) (m : Nat) (k : Nat)
    : toMat (matmulBwdW dout n m x k) k m = (toMat x n k)ᵀ * toMat dout n m := by
  ext kk j
  rw [Matrix.mul_apply]
  simp only [toMat, Matrix.of_apply, Matrix.transpose_apply]
  have hb : kk.1 * m + j.1 < k * m := idx_bound kk.1 j.1 k m kk.2 j.2
  have hmod : (kk.1 * m + j.1) % m = j.1 := by rw [Nat.mul_add_mod', Nat.mod_eq_of_lt j.2]
  have hdiv : (kk.1 * m + j.1) / m = kk.1 := by rw [Nat.mul_comm kk.1 m, Nat.mul_add_div (Nat.lt_of_le_of_lt (Nat.zero_le j.1) j.2), Nat.div_eq_of_lt j.2, Nat.add_zero]
  rw [matmulBwdW, map_range_getElem! _ _ _ hb]
  simp only [hdiv, hmod]
  rw [foldl_range_sum, Fin.sum_univ_eq_sum_range (fun p => x[p * k + kk.1]! * dout[p * m + j.1]!) n]

-- `maddFlat` equals matrix addition
theorem maddFlat_bridge {K : Type} [CommRing K] [Inhabited K] (a : Array K) (b : Array K) (rows : Nat) (cols : Nat) (ha : a.size = rows * cols)
    : toMat (maddFlat a b) rows cols = toMat a rows cols + toMat b rows cols := by
  ext i j
  simp only [toMat, Matrix.of_apply, Matrix.add_apply]
  have hb : i.1 * cols + j.1 < a.size := by rw [ha]; exact idx_bound i.1 j.1 rows cols i.2 j.2
  rw [maddFlat, map_range_getElem! _ _ _ hb]

-- the `bwdW` kernel is the adjoint (gradient) of the forward kernel
theorem matmulBwdW_kernel_grad (x : Array ℝ) (W : Array ℝ) (gArr : Array ℝ) (n : Nat) (k : Nat) (m : Nat)
    : Matrix.trace ((toMat gArr n m)ᵀ * toMat (matmulFwd x n k W m) n m) = Matrix.trace ((toMat (matmulBwdW gArr n m x k) k m)ᵀ * toMat W k m) := by rw [matmulFwd_bridge, matmulBwdW_bridge, matmulBwdW_adjoint]
-- the `bwdX` kernel is the adjoint (gradient) of the forward kernel
theorem matmulBwdX_kernel_grad (x : Array ℝ) (W : Array ℝ) (gArr : Array ℝ) (n : Nat) (k : Nat) (m : Nat)
    : Matrix.trace ((toMat gArr n m)ᵀ * toMat (matmulFwd x n k W m) n m) = Matrix.trace ((toMat (matmulBwdX gArr n m W k) n k)ᵀ * toMat x n k) := by rw [matmulFwd_bridge, matmulBwdX_bridge, matmulBwdX_adjoint]
-- add's gradient splits unchanged to both inputs
theorem maddFlat_kernel_grad (a : Array ℝ) (b : Array ℝ) (gArr : Array ℝ) (rows : Nat) (cols : Nat) (ha : a.size = rows * cols)
    : Matrix.trace ((toMat gArr rows cols)ᵀ * toMat (maddFlat a b) rows cols) = Matrix.trace ((toMat gArr rows cols)ᵀ * toMat a rows cols) + Matrix.trace ((toMat gArr rows cols)ᵀ * toMat b rows cols) := by rw [maddFlat_bridge a b rows cols ha, add_adjoint]

-- `∂/∂zᵢ Σⱼ exp(zⱼ) = exp(zᵢ)`
theorem exp_sum_partial {c : Nat} (z : Fin c → ℝ) (i : Fin c)
    : HasDerivAt (fun t => ∑ j, Real.exp (Function.update z i t j)) (Real.exp (z i)) (z i) := by
  have hfun : (fun t : ℝ => ∑ j, Real.exp (Function.update z i t j)) = ∑ j : Fin c, (fun t : ℝ => Real.exp (Function.update z i t j)) := by funext t; rw [Finset.sum_apply]
  rw [hfun, show Real.exp (z i) = ∑ j : Fin c, (if j = i then Real.exp (z i) else 0) by rw [Finset.sum_ite_eq' Finset.univ i (fun _ => Real.exp (z i)), if_pos (Finset.mem_univ i)]]
  apply HasDerivAt.sum
  intro j _
  by_cases hj : j = i
  · subst hj
    rw [if_pos rfl]
    have e : (fun t : ℝ => Real.exp (Function.update z j t j)) = Real.exp := by funext t; rw [Function.update_self]
    rw [e]; exact Real.hasDerivAt_exp (z j)
  · rw [if_neg hj]
    have e : (fun t : ℝ => Real.exp (Function.update z i t j)) = fun _ => Real.exp (z j) := by funext t; rw [Function.update_of_ne hj]
    rw [e]; exact hasDerivAt_const (z i) (Real.exp (z j))

-- `∂(log-sum-exp)/∂zᵢ = softmaxᵢ`
theorem lse_partial_deriv {c : Nat} (z : Fin c → ℝ) (i : Fin c)
    : HasDerivAt (fun t => lse (Function.update z i t)) (softmaxReal z i) (z i) := by
  have hpos : (0 : ℝ) < ∑ j, Real.exp (z j) := Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩
  have hne : (fun t => ∑ j, Real.exp (Function.update z i t j)) (z i) ≠ 0 := by simp only [Function.update_eq_self]; exact ne_of_gt hpos
  have hlog := (exp_sum_partial z i).log hne
  simp only [Function.update_eq_self] at hlog
  exact hlog

-- `∂(cross-entropy)/∂zᵢ = softmaxᵢ − [i=t]` (probs − onehot)
theorem ce_partial_deriv {c : Nat} (z : Fin c → ℝ) (t : Fin c) (i : Fin c)
    : HasDerivAt (fun s => ceLoss (Function.update z i s) t) (softmaxReal z i - (if i = t then 1 else 0)) (z i) := by
  have hupd : HasDerivAt (fun s => (Function.update z i s) t) (if i = t then 1 else 0) (z i) := by
    by_cases h : i = t
    · subst h; simp only [Function.update_self]; exact hasDerivAt_id (z i)
    · rw [if_neg h]
      have e : (fun s : ℝ => (Function.update z i s) t) = fun _ => z t := by funext s; rw [Function.update_of_ne (Ne.symm h)]
      rw [e]; exact hasDerivAt_const (z i) (z t)
  exact (lse_partial_deriv z i).sub hupd

-- chaining the affine logit map into CE gives `d/dw = (softmaxᵢ − [i=t])·a`
theorem layer_ce_chain {c : Nat} (zbase : Fin c → ℝ) (t : Fin c) (i : Fin c) (a : ℝ) (b : ℝ) (w : ℝ)
    : HasDerivAt (fun w => ceLoss (Function.update zbase i (a * w + b)) t) ((softmaxReal (Function.update zbase i (a * w + b)) i - (if i = t then 1 else 0)) * a) w := by
  have hg : HasDerivAt (fun s => ceLoss (Function.update zbase i s) t) (softmaxReal (Function.update zbase i (a * w + b)) i - (if i = t then 1 else 0)) (a * w + b) := by
    simpa [Function.update_idem, Function.update_self] using ce_partial_deriv (Function.update zbase i (a * w + b)) t i
  have hh : HasDerivAt (fun w : ℝ => a * w + b) a w := by simpa using ((hasDerivAt_id w).const_mul a).add_const b
  rw [show (fun w => ceLoss (Function.update zbase i (a * w + b)) t) = (fun s => ceLoss (Function.update zbase i s) t) ∘ (fun w : ℝ => a * w + b) from rfl]
  exact hg.comp w hh

-- tests
theorem add_requiresGrad (a : Tensor) (b : Tensor) : (a + b).requiresGrad = (a.requiresGrad || b.requiresGrad) := rfl
theorem add_shape (a : Tensor) (b : Tensor) : (a + b).shape = a.shape := rfl
theorem matmul_requiresGrad (x : Tensor) (w : Tensor) : (x @ w).requiresGrad = (x.requiresGrad || w.requiresGrad) := rfl
theorem matmul_shape (x : Tensor) (w : Tensor) : (x @ w).shape = #[x.rows, w.cols] := rfl
theorem gather_requiresGrad (table : Tensor) (ids : Array Nat) : (table.gather ids).requiresGrad = table.requiresGrad := rfl
theorem rmsnorm_shape (a : Tensor) (eps : Float) : (a.rmsnorm eps).shape = a.shape := rfl
theorem maskedCE_requiresGrad (logits : Tensor) (targets : Array Nat) (mask : Array Float) : (logits.maskedCE targets mask).requiresGrad = logits.requiresGrad := rfl
theorem add_no_grad : (Tensor.leaf #[1, 2] 1 2 0 false + Tensor.leaf #[3, 4] 1 2 1 false).requiresGrad = false := rfl
theorem matmul_taints_grad : ((Tensor.leaf #[1, 2, 3, 4] 2 2 0 false) @ (Tensor.leaf #[1, 2, 3, 4] 2 2 1 true)).requiresGrad = true := rfl
#guard let c := Tensor.leaf #[1, 2] 1 2 0 true + Tensor.leaf #[3, 4] 1 2 1 false; arrApproxEq c.data #[4, 6] && c.requiresGrad
#guard let t := (Tensor.leaf #[10, 11, 20, 21, 30, 31] 3 2 0 true).gather #[2, 0]; arrApproxEq t.data #[30, 31, 10, 11] && t.shape == #[2, 2]
#guard arrApproxEq ((Tensor.leaf #[1, 2, 3, 4] 2 2 0 true) @ (Tensor.leaf #[1, 2, 3, 4] 2 2 1 true)).data #[7, 10, 15, 22]
#guard let l := (Tensor.leaf #[0, 0] 1 2 0 true).maskedCE #[0] #[1]; approxEq l.data[0]! (-Float.log 0.5) && l.shape == #[1, 1]
-- the proven kernels run on exact ℚ: computed value equals the proven formula under `==`, no float tolerance.
#guard matmulFwd (#[1, 2, 3, 4] : Array ℚ) 2 2 #[1, 2, 3, 4] 2 == #[7, 10, 15, 22]
#guard matmulFwd (#[1, 2, 3, 4, 5, 6] : Array ℚ) 2 3 #[1, 2, 3, 4, 5, 6] 2 == #[22, 28, 49, 64]
#guard maddFlat (#[1, 2, 3] : Array ℚ) #[3, 4, 5] == #[4, 6, 8]
#guard matmulBwdX (#[1, 2, 3, 4] : Array ℚ) 2 2 #[1, 0, 0, 1] 2 == #[1, 2, 3, 4]
#guard matmulBwdW (#[1, 2, 3, 4] : Array ℚ) 2 2 #[1, 0, 0, 1] 2 == #[1, 2, 3, 4]
#guard gatherFlat (#[10, 11, 20, 21, 30, 31] : Array ℚ) 2 #[2, 0] == #[30, 31, 10, 11]
#guard transposeFlat (#[1, 2, 3, 4, 5, 6] : Array ℚ) 2 3 == #[1, 4, 2, 5, 3, 6]

/-!
===--------------------------------------------------------------------------===
Backward

Immutable tensors have no `.grad` field.
`backwardAcc` returns a `gradientMap` of type `Array (Nat × Array Float)`
where each entry is `(t.id, gradient of t.data)`.

  let a := Tensor.leaf #[1,2] .. (id := 0)       -- a.data = [1,2]
  let b := Tensor.leaf #[3,4] .. (id := 1)       -- b.data = [3,4]
  let c := a + b + a                             -- a is used twice

`forward` builds this graph (each node links back to its inputs via `gradFn`):

        c
       / \
    (a+b) a
     / \
    a   b

backprop starts here with seed #[1,1] and the gradient flows down to the leaves.
`a` is a shared and reached via two paths, so it has two gradients sum.

  c.backwardAcc #[1,1] #[]
  --            ^^^^^^      gradient to start from, all 1s because backprop begins at c
  --                   ^^^  empty map to fill

the `gradientMap` evolves as the walk reaches each leaf, last line is the return value:

  #[]                               -- map starts empty
  #[(0, #[1,1])]                    -- reached a (id 0): not present yet, append
  #[(0, #[1,1]), (1, #[1,1])]       -- reached b (id 1): not present yet, append
  #[(0, #[2,2]), (1, #[1,1])]       -- reached a again (id 0): already present, so sum -> [2,2]

Each `#[1,1]` is the gradient of that tensor's `.data` (same length).
`gradientMapAdd` sums into an entry if its `id` is already present, else appends.

The optimizer pulls a weight's gradient out of the final map by its `id`:

  lookup gradientMap a.id       -- a.id = 0 -> #[2,2]
  lookup gradientMap b.id       -- b.id = 1 -> #[1,1]
===--------------------------------------------------------------------------===
-/

private def gradientMapAdd (gradientMap : Array (Nat × Array Float)) (id : Nat) (g : Array Float) : Array (Nat × Array Float) :=
  match gradientMap.findIdx? (fun (i, _) => i = id) with
  | some i => gradientMap.set! i (id, maddFlat gradientMap[i]!.2 g)
  | none => gradientMap.push (id, g)

-- walks the gradFn DAG, accumulating each leaf's gradient into the map (shared leaves sum)
private partial def backwardAcc (t : Tensor) (incoming : Array Float) (gradientMap : Array (Nat × Array Float)) : Array (Nat × Array Float) :=
  match t.gradFn with
  | .leaf =>
    if t.requiresGrad then gradientMapAdd gradientMap t.id incoming else gradientMap
  | .addOp a b =>
    let gradientMap := a.backwardAcc incoming gradientMap
    b.backwardAcc incoming gradientMap
  | .matmulOp x w =>
    let n := x.rows; let k := x.cols; let m := w.cols
    let dx := matmulBwdX incoming n m w.data k
    let dw := matmulBwdW incoming n m x.data k
    let gradientMap := x.backwardAcc dx gradientMap
    w.backwardAcc dw gradientMap
  | .gatherOp table ids =>
    let dTable := scatterAddFlat table.rows table.cols incoming ids
    table.backwardAcc dTable gradientMap
  | .rmsnormOp a rms =>
    a.backwardAcc (rmsnormBwd incoming a.data rms a.rows a.cols) gradientMap
  | .lossOp logits probs targets mask sumMask =>
    let scale := if incoming.size > 0 then incoming[0]! else 1.0
    let dLogits := maskedCrossEntropyBwd probs logits.rows logits.cols targets mask sumMask
    let scaled := dLogits.map (scale * ·)
    logits.backwardAcc scaled gradientMap
  | .attnOp xPre wq wk wv wo cache nEmbed nHead =>
    let (dxPre, (dWq, dWk, dWv, dWo)) := attnBwd nEmbed nHead incoming xPre.rows wq.data wk.data wv.data wo.data cache
    let gradientMap := xPre.backwardAcc dxPre gradientMap
    let gradientMap := wq.backwardAcc dWq gradientMap
    let gradientMap := wk.backwardAcc dWk gradientMap
    let gradientMap := wv.backwardAcc dWv gradientMap
    wo.backwardAcc dWo gradientMap
  | .mlpOp xPre fc1 fc2 cache nEmbed =>
    let (dxPre, (df1, df2)) := mlpBwd nEmbed xPre.rows incoming fc1.data fc2.data cache
    let gradientMap := xPre.backwardAcc dxPre gradientMap
    let gradientMap := fc1.backwardAcc df1 gradientMap
    fc2.backwardAcc df2 gradientMap

def backward (loss : Tensor) : Array (Nat × Array Float) :=
  loss.backwardAcc #[1.0] #[]

-- adding an unseen id appends a new entry
theorem gradientMapAdd_fresh (gm : Array (Nat × Array Float)) (id : Nat) (g : Array Float) (h : gm.findIdx? (fun (i, _) => i = id) = none)
    : gradientMapAdd gm id g = gm.push (id, g) := by unfold gradientMapAdd; rw [h]

-- adding the same id twice sums the two gradients
theorem gradientMapAdd_shared (id : Nat) (g : Array Float) (g' : Array Float)
    : gradientMapAdd (gradientMapAdd #[] id g) id g' = #[(id, maddFlat g g')] := by
  have h0 : gradientMapAdd #[] id g = #[(id, g)] := by unfold gradientMapAdd; simp [Array.findIdx?, Array.findIdx?.loop]
  rw [h0]; unfold gradientMapAdd; simp [Array.findIdx?, Array.findIdx?.loop]

-- after adding an id, an entry for it exists
theorem gradientMapAdd_mem (gm : Array (Nat × Array Float)) (id : Nat) (g : Array Float)
    : ∃ p ∈ (gradientMapAdd gm id g), p.1 = id := by
  unfold gradientMapAdd
  split
  case h_1 i h =>
    have hi : i < gm.size := (Array.findIdx?_eq_some_iff_getElem.mp h).1
    exact ⟨(id, maddFlat gm[i]!.2 g), by rw [Array.set!_eq_setIfInBounds]; exact Array.mem_setIfInBounds hi, rfl⟩
  case h_2 h => exact ⟨(id, g), by simp, rfl⟩

-- adding a gradient never drops an id already present
theorem gradientMapAdd_pres_ids (gm : Array (Nat × Array Float)) (id : Nat) (g : Array Float) (j : Nat) (hj : ∃ p ∈ gm, p.1 = j)
    : ∃ p ∈ (gradientMapAdd gm id g), p.1 = j := by
  obtain ⟨p, hp, hpj⟩ := hj
  rw [Array.mem_iff_getElem] at hp
  obtain ⟨k, hk, hkp⟩ := hp
  unfold gradientMapAdd
  split
  case h_2 h => exact ⟨p, Array.mem_push.mpr (Or.inl (Array.mem_iff_getElem.mpr ⟨k, hk, hkp⟩)), hpj⟩
  case h_1 i h =>
    have hi : i < gm.size := (Array.findIdx?_eq_some_iff_getElem.mp h).1
    rw [Array.set!_eq_setIfInBounds]
    by_cases hik : i = k
    · have hmatch : gm[i].1 = id := by simpa using (Array.findIdx?_eq_some_iff_getElem.mp h).2.1
      refine ⟨(id, maddFlat gm[i]!.2 g), Array.mem_setIfInBounds hi, ?_⟩
      subst hik
      rw [hkp] at hmatch
      rw [← hmatch]; exact hpj
    · have hsize : k < (gm.setIfInBounds i (id, maddFlat gm[i]!.2 g)).size := by rw [Array.size_setIfInBounds]; exact hk
      refine ⟨p, Array.mem_iff_getElem.mpr ⟨k, hsize, ?_⟩, hpj⟩
      rw [Array.getElem_setIfInBounds_ne hk hik, hkp]

-- every id after an add is the new id or one already present (no spurious ids)
theorem gradientMapAdd_ids_subset (gm : Array (Nat × Array Float)) (id : Nat) (g : Array Float) (j : Nat) (hj : ∃ p ∈ (gradientMapAdd gm id g), p.1 = j)
    : j = id ∨ ∃ p ∈ gm, p.1 = j := by
  obtain ⟨p, hp, hpj⟩ := hj
  unfold gradientMapAdd at hp
  split at hp
  case h_2 h =>
    rcases Array.mem_push.mp hp with hmem | heq
    · exact Or.inr ⟨p, hmem, hpj⟩
    · left; rw [← hpj, heq]
  case h_1 i h =>
    have hi : i < gm.size := (Array.findIdx?_eq_some_iff_getElem.mp h).1
    rw [Array.set!_eq_setIfInBounds] at hp
    rw [Array.mem_iff_getElem] at hp
    obtain ⟨k, hk, hkp⟩ := hp
    rw [Array.size_setIfInBounds] at hk
    by_cases hik : i = k
    · subst hik
      rw [Array.getElem_setIfInBounds_self] at hkp
      left; rw [← hpj, ← hkp]
    · rw [Array.getElem_setIfInBounds_ne hk hik] at hkp
      exact Or.inr ⟨p, Array.mem_iff_getElem.mpr ⟨k, hk, hkp⟩, hpj⟩

-- tests
#guard
  let gm := (Tensor.leaf #[1, 2] 1 2 0 true + Tensor.leaf #[3, 4] 1 2 1 true + Tensor.leaf #[1, 2] 1 2 0 true).backwardAcc #[1, 1] #[]
  let get := fun (id : Nat) => (gm.find? (fun p => p.1 == id)).map (·.2) |>.getD #[]
  arrApproxEq (get 0) #[2, 2] && arrApproxEq (get 1) #[1, 1]
#guard
  let gm := ((Tensor.leaf #[1, 2, 3, 4] 2 2 0 true) @ (Tensor.leaf #[1, 0, 0, 1] 2 2 1 true)).backwardAcc #[1, 1, 1, 1] #[]
  let get := fun (id : Nat) => (gm.find? (fun p => p.1 == id)).map (·.2) |>.getD #[]
  arrApproxEq (get 0) #[1, 1, 1, 1] && arrApproxEq (get 1) #[4, 4, 6, 6]
#guard
  let gm := ((Tensor.leaf #[0, 0] 1 2 0 true).maskedCE #[0] #[1]).backward
  let g := (gm.find? (fun p => p.1 == 0)).map (·.2) |>.getD #[]
  approxEq (g[0]! + g[1]!) 0.0

end Tensor

end Autograd
