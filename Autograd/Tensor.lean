import Autograd.Ops

namespace Autograd

/-! ## Tensor + GradFn + autograd-aware forward + backward

Mirrors autograd.c's `struct Tensor`:

    typedef struct Tensor {
        float32_t *data;   // flat contiguous array, row-major
        uint64_t *shape;   // dimension sizes
        uint64_t *strides; // (computed from shape; not stored here)
        uint64_t ndim;     // == shape.size
        uint64_t size;     // == ∏ shape
        bool requires_grad;
        struct Tensor *grad;
        Function *grad_fn;
        uint32_t ref_count;       // Lean GC handles this
    } Tensor;

Differences from the C version:
- `grad` isn't a mutating field; `Tensor.backward` returns a `Nat → Array Float`
  map keyed by leaf `id`. This is the immutable analog of walking the graph and
  writing into each input tensor's `grad` field.
- `grad_fn` is a closed `GradFn` enum (not a `Function*` callback); backward
  pattern-matches on it.
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
  | gather   (table : Tensor) (ids : Array Nat)
  | addOp    (a b : Tensor)
  | linearOp (x w : Tensor)
  | rmsnormOp (a : Tensor) (rms : Array Float)
  | attnOp   (xPre wq wk wv wo : Tensor) (cache : AttnCache) (cfg : Config)
  | mlpOp    (xPre fc1 fc2 : Tensor) (cache : MlpCache) (cfg : Config)
  | lossOp   (logits : Tensor) (probs : Array Float) (targets : Array Nat)
             (mask : Array Float) (sumMask : Float)
end

instance : Inhabited GradFn := ⟨.leaf⟩
instance : Inhabited Tensor := ⟨{ data := #[], shape := #[], id := 0,
                                  requiresGrad := false, gradFn := .leaf }⟩

namespace Tensor
def ndim (t : Tensor) : Nat := t.shape.size
def size (t : Tensor) : Nat := t.shape.foldl (init := 1) (· * ·)
def rows (t : Tensor) : Nat := if t.shape.size = 0 then 0 else t.shape[0]!
def cols (t : Tensor) : Nat := if t.shape.size < 2 then 1 else t.shape[1]!
def get (t : Tensor) (i j : Nat) : Float := t.data[i * t.cols + j]!

def zeros (rows cols : Nat) : Tensor :=
  { data := Array.replicate (rows * cols) 0.0, shape := #[rows, cols],
    id := 0, requiresGrad := false, gradFn := .leaf }

def ofFn (rows cols : Nat) (f : Nat → Nat → Float) : Tensor :=
  { data := (Array.range (rows * cols)).map fun k => f (k / cols) (k % cols),
    shape := #[rows, cols], id := 0, requiresGrad := false, gradFn := .leaf }

def leaf (data : Array Float) (rows cols id : Nat) (requiresGrad : Bool) : Tensor :=
  { data := data, shape := #[rows, cols], id := id,
    requiresGrad := requiresGrad, gradFn := .leaf }

/-! ## Forward ops — each builds a Tensor and records the GradFn -/

def gather (table : Tensor) (ids : Array Nat) : Tensor :=
  let cols := table.cols
  let data : Array Float := Id.run do
    let mut acc : Array Float := Array.replicate (ids.size * cols) 0.0
    for i in [0:ids.size] do
      let id := ids[i]!
      for j in [0:cols] do
        acc := acc.set! (i * cols + j) table.data[id * cols + j]!
    return acc
  { data := data, shape := #[ids.size, cols], id := 0,
    requiresGrad := table.requiresGrad, gradFn := .gather table ids }

def add (a b : Tensor) : Tensor :=
  { data := maddFlat a.data b.data, shape := a.shape, id := 0,
    requiresGrad := a.requiresGrad || b.requiresGrad, gradFn := .addOp a b }

-- x (n × k) @ w (k × m) → (n × m)
def linear (x w : Tensor) : Tensor :=
  let n := x.rows; let k := x.cols; let m := w.cols
  { data := linearFwd x.data n k w.data m, shape := #[n, m], id := 0,
    requiresGrad := x.requiresGrad || w.requiresGrad, gradFn := .linearOp x w }

def rmsnorm (a : Tensor) (eps : Float) : Tensor :=
  let (y, rms) := rmsnormFwd a.data a.rows a.cols eps
  { data := y, shape := a.shape, id := 0,
    requiresGrad := a.requiresGrad, gradFn := .rmsnormOp a rms }

def attn (cfg : Config) (xPre wq wk wv wo : Tensor) : Tensor :=
  let (out, cache) := attnFwd cfg xPre.data xPre.rows wq.data wk.data wv.data wo.data
  { data := out, shape := xPre.shape, id := 0,
    requiresGrad := xPre.requiresGrad || wq.requiresGrad || wk.requiresGrad
                    || wv.requiresGrad || wo.requiresGrad,
    gradFn := .attnOp xPre wq wk wv wo cache cfg }

def mlp (cfg : Config) (xPre fc1 fc2 : Tensor) : Tensor :=
  let (out, cache) := mlpFwd cfg xPre.data xPre.rows fc1.data fc2.data
  { data := out, shape := xPre.shape, id := 0,
    requiresGrad := xPre.requiresGrad || fc1.requiresGrad || fc2.requiresGrad,
    gradFn := .mlpOp xPre fc1 fc2 cache cfg }

-- scalar loss tensor; backward starts from a [1.0] buffer
def maskedCE (logits : Tensor) (targets : Array Nat) (mask : Array Float) : Tensor :=
  let probs := softmaxRows logits.data logits.rows logits.cols
  let sumMask := mask.foldl (init := 0.0) (· + ·)
  let l := maskedCrossEntropy probs logits.rows logits.cols targets mask sumMask
  { data := #[l], shape := #[1, 1], id := 0,
    requiresGrad := logits.requiresGrad,
    gradFn := .lossOp logits probs targets mask sumMask }

end Tensor

/-! ## Backward — walks the immutable graph, returns per-leaf accumulated grads -/

private def gmAdd (gm : Array (Nat × Array Float)) (id : Nat) (g : Array Float)
    : Array (Nat × Array Float) :=
  match gm.findIdx? (fun (i, _) => i = id) with
  | some i => gm.set! i (id, maddFlat gm[i]!.2 g)
  | none => gm.push (id, g)

partial def Tensor.backwardAcc (t : Tensor) (incoming : Array Float)
    (gm : Array (Nat × Array Float)) : Array (Nat × Array Float) :=
  match t.gradFn with
  | .leaf =>
    if t.requiresGrad then gmAdd gm t.id incoming else gm
  | .addOp a b =>
    let gm := a.backwardAcc incoming gm
    b.backwardAcc incoming gm
  | .linearOp x w =>
    let n := x.rows; let k := x.cols; let m := w.cols
    let dx := linearBwdX incoming n m w.data k
    let dw := linearBwdW incoming n m x.data k
    let gm := x.backwardAcc dx gm
    w.backwardAcc dw gm
  | .rmsnormOp a rms =>
    a.backwardAcc (rmsnormBwd incoming a.data rms a.rows a.cols) gm
  | .gather table ids =>
    let dTable := scatterAddFlat table.rows table.cols incoming ids
    table.backwardAcc dTable gm
  | .attnOp xPre wq wk wv wo cache cfg =>
    let (dxPre, (dWq, dWk, dWv, dWo)) :=
      attnBwd cfg incoming xPre.rows wq.data wk.data wv.data wo.data cache
    let gm := xPre.backwardAcc dxPre gm
    let gm := wq.backwardAcc dWq gm
    let gm := wk.backwardAcc dWk gm
    let gm := wv.backwardAcc dWv gm
    wo.backwardAcc dWo gm
  | .mlpOp xPre fc1 fc2 cache _ =>
    let (dxPre, (df1, df2)) := mlpBwd incoming fc1.data fc2.data cache
    let gm := xPre.backwardAcc dxPre gm
    let gm := fc1.backwardAcc df1 gm
    fc2.backwardAcc df2 gm
  | .lossOp logits probs targets mask sumMask =>
    let scale := if incoming.size > 0 then incoming[0]! else 1.0
    let dLogits := maskedCrossEntropyBwd probs logits.rows logits.cols targets mask sumMask
    let scaled := dLogits.map (scale * ·)
    logits.backwardAcc scaled gm

def Tensor.backward (loss : Tensor) : Array (Nat × Array Float) :=
  loss.backwardAcc #[1.0] #[]

end Autograd
