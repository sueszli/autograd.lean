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

/-!
===--------------------------------------------------------------------------===
Tensor methods
===--------------------------------------------------------------------------===
-/

namespace Tensor
def rows (t : Tensor) : Nat := if t.shape.size = 0 then 0 else t.shape[0]!
def cols (t : Tensor) : Nat := if t.shape.size < 2 then 1 else t.shape[1]!

def leaf (data : Array Float) (rows cols id : Nat) (requiresGrad : Bool) : Tensor :=
  { data := data, shape := #[rows, cols], id := id, requiresGrad := requiresGrad, gradFn := .leaf }

/-!
===--------------------------------------------------------------------------===
Forward ops
===--------------------------------------------------------------------------===
-/

def gather (table : Tensor) (ids : Array Nat) : Tensor :=
  let cols := table.cols
  let data : Array Float := Id.run do
    let mut acc : Array Float := Array.replicate (ids.size * cols) 0.0
    for i in [0:ids.size] do
      let id := ids[i]!
      for j in [0:cols] do
        acc := acc.set! (i * cols + j) table.data[id * cols + j]!
    return acc
  { data := data, shape := #[ids.size, cols], id := 0, requiresGrad := table.requiresGrad, gradFn := .gatherOp table ids }

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

-- scalar loss tensor, `backward` starts from a `[1.0]` buffer
def maskedCE (logits : Tensor) (targets : Array Nat) (mask : Array Float) : Tensor :=
  let probs := softmaxRows logits.data logits.rows logits.cols
  let sumMask := mask.foldl (init := 0.0) (· + ·)
  let l := maskedCrossEntropy probs logits.rows logits.cols targets mask sumMask
  { data := #[l], shape := #[1, 1], id := 0, requiresGrad := logits.requiresGrad, gradFn := .lossOp logits probs targets mask sumMask }

end Tensor

/-!
===--------------------------------------------------------------------------===
Backward

Immutable tensors have no `.grad` field.
`backwardAcc` returns a `gradientMap` of type `Array (Nat × Array Float)`
where each entry is `(t.id, gradient of t.data)`.

  let a := Tensor.leaf #[1,2] .. (id := 0)   -- a.data = [1,2]
  let b := Tensor.leaf #[3,4] .. (id := 1)   -- b.data = [3,4]
  let c := a + b + a                         -- a is used twice

`forward` builds this graph (each node links back to its inputs via `gradFn`):

        c                  backprop starts here with seed #[1,1].
       / \
    (a+b) a                gradient flows down to the leaves.
     / \
    a   b                  `a` is a shared leaf: reached via the left subtree
                           AND the right branch, so it has two gradients sum.

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

  lookup gradientMap a.id   -- a.id = 0 -> #[2,2]
  lookup gradientMap b.id   -- b.id = 1 -> #[1,1]
===--------------------------------------------------------------------------===
-/

private def gradientMapAdd (gradientMap : Array (Nat × Array Float)) (id : Nat) (g : Array Float) : Array (Nat × Array Float) :=
  match gradientMap.findIdx? (fun (i, _) => i = id) with
  | some i => gradientMap.set! i (id, maddFlat gradientMap[i]!.2 g)
  | none => gradientMap.push (id, g)

partial def Tensor.backwardAcc (t : Tensor) (incoming : Array Float) (gradientMap : Array (Nat × Array Float)) : Array (Nat × Array Float) :=
  match t.gradFn with
  | .leaf =>
    if t.requiresGrad then gradientMapAdd gradientMap t.id incoming else gradientMap
  -- feed each call's returned map into the next. `add` splits its gradient unchanged to both inputs.
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
    let (dxPre, (dWq, dWk, dWv, dWo)) :=
      attnBwd cfg incoming xPre.rows wq.data wk.data wv.data wo.data cache
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

def Tensor.backward (loss : Tensor) : Array (Nat × Array Float) :=
  loss.backwardAcc #[1.0] #[]

end Autograd
