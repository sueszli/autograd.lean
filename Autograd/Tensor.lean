import Autograd.Ops

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
  | attnOp    (xPre wq wk wv wo : Tensor) (cache : AttnCache) (cfg : AttnConfig)
  | mlpOp     (xPre fc1 fc2 : Tensor) (cache : MlpCache) (cfg : MlpConfig)
end

instance : Inhabited Tensor := ⟨{ data := #[], shape := #[], id := 0, requiresGrad := false, gradFn := .leaf }⟩
instance : Inhabited GradFn := ⟨.leaf⟩

namespace Tensor

def rows (t : Tensor) : Nat := if t.shape.size = 0 then 0 else t.shape[0]! -- `shape[0]` rows, `shape[1]` cols
def cols (t : Tensor) : Nat := if t.shape.size < 2 then 1 else t.shape[1]!
def leaf (data : Array Float) (rows cols id : Nat) (requiresGrad : Bool) : Tensor := { data := data, shape := #[rows, cols], id := id, requiresGrad := requiresGrad, gradFn := .leaf }

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

def add (a b : Tensor) : Tensor :=
  { data := maddFlat a.data b.data, shape := a.shape, id := 0, requiresGrad := a.requiresGrad || b.requiresGrad, gradFn := .addOp a b }

instance : Add Tensor := ⟨Tensor.add⟩

def matmul (x w : Tensor) : Tensor :=
  let n := x.rows; let k := x.cols; let m := w.cols
  { data := matmulFwd x.data n k w.data m, shape := #[n, m], id := 0, requiresGrad := x.requiresGrad || w.requiresGrad, gradFn := .matmulOp x w }

infixl:70 " @ " => Tensor.matmul

def rmsnorm (a : Tensor) (eps : Float) : Tensor :=
  let (y, rms) := rmsnormFwd a.data a.rows a.cols eps
  { data := y, shape := a.shape, id := 0, requiresGrad := a.requiresGrad, gradFn := .rmsnormOp a rms }

def attn (cfg : AttnConfig) (xPre wq wk wv wo : Tensor) : Tensor :=
  let (out, cache) := attnFwd cfg xPre.data xPre.rows wq.data wk.data wv.data wo.data
  { data := out, shape := xPre.shape, id := 0, requiresGrad := xPre.requiresGrad || wq.requiresGrad || wk.requiresGrad || wv.requiresGrad || wo.requiresGrad, gradFn := .attnOp xPre wq wk wv wo cache cfg }

def mlp (cfg : MlpConfig) (xPre fc1 fc2 : Tensor) : Tensor :=
  let (out, cache) := mlpFwd cfg xPre.data xPre.rows fc1.data fc2.data
  { data := out, shape := xPre.shape, id := 0, requiresGrad := xPre.requiresGrad || fc1.requiresGrad || fc2.requiresGrad, gradFn := .mlpOp xPre fc1 fc2 cache cfg }

def maskedCE (logits : Tensor) (targets : Array Nat) (mask : Array Float) : Tensor :=
  let probs := softmaxRows logits.data logits.rows logits.cols
  let sumMask := mask.foldl (init := 0.0) (· + ·)
  let l := maskedCrossEntropy probs logits.rows logits.cols targets mask sumMask
  { data := #[l], shape := #[1, 1], id := 0, requiresGrad := logits.requiresGrad, gradFn := .lossOp logits probs targets mask sumMask }

theorem add_requiresGrad (a : Tensor) (b : Tensor) : (a + b).requiresGrad = (a.requiresGrad || b.requiresGrad) := rfl
theorem add_shape (a : Tensor) (b : Tensor) : (a + b).shape = a.shape := rfl
theorem matmul_requiresGrad (x : Tensor) (w : Tensor) : (x @ w).requiresGrad = (x.requiresGrad || w.requiresGrad) := rfl
theorem matmul_shape (x : Tensor) (w : Tensor) : (x @ w).shape = #[x.rows, w.cols] := rfl
theorem gather_requiresGrad (table : Tensor) (ids : Array Nat) : (table.gather ids).requiresGrad = table.requiresGrad := rfl
theorem rmsnorm_shape (a : Tensor) (eps : Float) : (a.rmsnorm eps).shape = a.shape := rfl
theorem maskedCE_requiresGrad (logits : Tensor) (targets : Array Nat) (mask : Array Float) : (logits.maskedCE targets mask).requiresGrad = logits.requiresGrad := rfl

#guard let c := Tensor.leaf #[1, 2] 1 2 0 true + Tensor.leaf #[3, 4] 1 2 1 false; arrApproxEq c.data #[4, 6] && c.requiresGrad  -- `add` sums, ORs `requiresGrad`
#guard !(Tensor.leaf #[1, 2] 1 2 0 false + Tensor.leaf #[3, 4] 1 2 1 false).requiresGrad  -- no input tracks grad, neither does the sum
#guard ((Tensor.leaf #[1, 2, 3, 4] 2 2 0 false) @ (Tensor.leaf #[1, 2, 3, 4] 2 2 1 true)).requiresGrad  -- one tracked operand taints the matmul
#guard let t := (Tensor.leaf #[10, 11, 20, 21, 30, 31] 3 2 0 true).gather #[2, 0]; arrApproxEq t.data #[30, 31, 10, 11] && t.shape == #[2, 2]  -- `gather` picks rows 2, 0
#guard arrApproxEq ((Tensor.leaf #[1, 2, 3, 4] 2 2 0 true) @ (Tensor.leaf #[1, 2, 3, 4] 2 2 1 true)).data #[7, 10, 15, 22]  -- `@` matmul
#guard let l := (Tensor.leaf #[0, 0] 1 2 0 true).maskedCE #[0] #[1]; approxEq l.data[0]! (-Float.log 0.5) && l.shape == #[1, 1]  -- uniform logits cost `-log 0.5`, `[1,1]` scalar

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

partial def backwardAcc (t : Tensor) (incoming : Array Float) (gradientMap : Array (Nat × Array Float)) : Array (Nat × Array Float) :=
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
  | .attnOp xPre wq wk wv wo cache cfg =>
    let (dxPre, (dWq, dWk, dWv, dWo)) := attnBwd cfg incoming xPre.rows wq.data wk.data wv.data wo.data cache
    let gradientMap := xPre.backwardAcc dxPre gradientMap
    let gradientMap := wq.backwardAcc dWq gradientMap
    let gradientMap := wk.backwardAcc dWk gradientMap
    let gradientMap := wv.backwardAcc dWv gradientMap
    wo.backwardAcc dWo gradientMap
  | .mlpOp xPre fc1 fc2 cache _ =>
    let (dxPre, (df1, df2)) := mlpBwd incoming fc1.data fc2.data cache
    let gradientMap := xPre.backwardAcc dxPre gradientMap
    let gradientMap := fc1.backwardAcc df1 gradientMap
    fc2.backwardAcc df2 gradientMap

def backward (loss : Tensor) : Array (Nat × Array Float) :=
  loss.backwardAcc #[1.0] #[]

-- append base case: an unseen `id` becomes a new trailing entry holding `g` verbatim.
theorem gradientMapAdd_fresh (gm : Array (Nat × Array Float)) (id : Nat) (g : Array Float) (h : gm.findIdx? (fun (i, _) => i = id) = none) : gradientMapAdd gm id g = gm.push (id, g) := by
  unfold gradientMapAdd; rw [h]

-- shared-leaf accumulation: two adds for the same `id` collapse to ONE entry whose value is `maddFlat g g'`, the defining property of reverse-mode autograd.
theorem gradientMapAdd_shared (id : Nat) (g : Array Float) (g' : Array Float) : gradientMapAdd (gradientMapAdd #[] id g) id g' = #[(id, maddFlat g g')] := by
  have h0 : gradientMapAdd #[] id g = #[(id, g)] := by unfold gradientMapAdd; simp [Array.findIdx?, Array.findIdx?.loop]
  rw [h0]; unfold gradientMapAdd; simp [Array.findIdx?, Array.findIdx?.loop]

-- retrievability: after recording `id`, an entry for it always exists, so the optimizer's lookup can never miss it. covers both `set!` and `push` branches.
theorem gradientMapAdd_mem (gm : Array (Nat × Array Float)) (id : Nat) (g : Array Float) : ∃ p ∈ (gradientMapAdd gm id g), p.1 = id := by
  unfold gradientMapAdd
  split
  case h_1 i h =>
    have hi : i < gm.size := (Array.findIdx?_eq_some_iff_getElem.mp h).1
    refine ⟨(id, maddFlat gm[i]!.2 g), ?_, rfl⟩
    rw [Array.set!_eq_setIfInBounds]
    exact Array.mem_setIfInBounds hi
  case h_2 h => exact ⟨(id, g), by simp, rfl⟩

-- no-clobber: adding one leaf's gradient never drops any other leaf already in the map, so after a full backward pass every parameter that got a gradient still has it.
theorem gradientMapAdd_pres_ids (gm : Array (Nat × Array Float)) (id : Nat) (g : Array Float) (j : Nat) (hj : ∃ p ∈ gm, p.1 = j) : ∃ p ∈ (gradientMapAdd gm id g), p.1 = j := by
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

-- converse of `pres_ids`: every id present after an add was either `id` or already there, so accumulation invents no spurious ids. with `mem` + `pres_ids` this pins the id-set to exactly `old ∪ {id}`.
theorem gradientMapAdd_ids_subset (gm : Array (Nat × Array Float)) (id : Nat) (g : Array Float) (j : Nat) (hj : ∃ p ∈ (gradientMapAdd gm id g), p.1 = j) : j = id ∨ ∃ p ∈ gm, p.1 = j := by
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

-- the shared-leaf example from the section doc: `a` (id 0) is reached twice in `a + b + a`,
-- so its gradient accumulates to `[2,2]` while `b` (id 1) stays at `[1,1]`
#guard
  let gm := (Tensor.leaf #[1, 2] 1 2 0 true + Tensor.leaf #[3, 4] 1 2 1 true + Tensor.leaf #[1, 2] 1 2 0 true).backwardAcc #[1, 1] #[]
  let get := fun (id : Nat) => (gm.find? (fun p => p.1 == id)).map (·.2) |>.getD #[]
  arrApproxEq (get 0) #[2, 2] && arrApproxEq (get 1) #[1, 1]
-- matmul backward: `dx = dout @ wᵀ`, `dw = xᵀ @ dout` (here `w = I` so `dx = dout`)
#guard
  let gm := ((Tensor.leaf #[1, 2, 3, 4] 2 2 0 true) @ (Tensor.leaf #[1, 0, 0, 1] 2 2 1 true)).backwardAcc #[1, 1, 1, 1] #[]
  let get := fun (id : Nat) => (gm.find? (fun p => p.1 == id)).map (·.2) |>.getD #[]
  arrApproxEq (get 0) #[1, 1, 1, 1] && arrApproxEq (get 1) #[4, 4, 6, 6]
-- `backward` seeds `[1.0]`; cross-entropy gradient `probs - onehot` sums to zero across the row
#guard
  let gm := ((Tensor.leaf #[0, 0] 1 2 0 true).maskedCE #[0] #[1]).backward
  let g := (gm.find? (fun p => p.1 == 0)).map (·.2) |>.getD #[]
  approxEq (g[0]! + g[1]!) 0.0

end Tensor

end Autograd
