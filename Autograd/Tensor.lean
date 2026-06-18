import Autograd.Matrix
import Autograd.Ops

namespace Autograd

/-! ## Tensor + GradFn

Mirrors autograd.c's `struct Tensor { data, shape, requires_grad, grad, grad_fn }`.
- `data`        : numeric content (2D Matrix; we collapsed shape/strides/ndim to 2D
                  since microGPT doesn't use higher ranks).
- `id`          : identifies leaves (params/inputs) so `backward` can return their
                  accumulated grads in a `Nat → Matrix` map. Pure-functional analog
                  of mutating each Tensor's `.grad` field in-place.
- `requiresGrad`: same as autograd.c.
- `gradFn`      : the op that created this tensor. autograd.c uses a `Function*`
                  (function pointer + ctx); we use a closed enum with the inputs
                  and any cached state stored in each variant. Pattern-matching on
                  it during `backward` is the equivalent of calling `apply(self, …)`.
-/

mutual
structure Tensor where
  data : Matrix
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
  | lossOp   (logits : Tensor) (probs : Matrix) (targets : Array Nat)
             (mask : Array Float) (sumMask : Float)
end

instance : Inhabited GradFn := ⟨.leaf⟩
instance : Inhabited Tensor := ⟨{ data := #[], id := 0, requiresGrad := false, gradFn := .leaf }⟩

-- shorthand for leaf tensors (params, inputs)
def Tensor.leaf (data : Matrix) (id : Nat := 0) (requiresGrad : Bool := false) : Tensor :=
  { data := data, id := id, requiresGrad := requiresGrad, gradFn := .leaf }

/-! ## Tensor-level forward ops

Each takes/returns Tensors and records the GradFn so backward can reconstruct
the gradient flow without re-running the forward.
-/

def Tensor.gather (table : Tensor) (ids : Array Nat) : Tensor :=
  { data := ids.map fun i => table.data[i]!, id := 0,
    requiresGrad := table.requiresGrad, gradFn := .gather table ids }

def Tensor.add (a b : Tensor) : Tensor :=
  { data := madd a.data b.data, id := 0,
    requiresGrad := a.requiresGrad || b.requiresGrad, gradFn := .addOp a b }

def Tensor.linear (x w : Tensor) : Tensor :=
  { data := linearFwd x.data w.data, id := 0,
    requiresGrad := x.requiresGrad || w.requiresGrad, gradFn := .linearOp x w }

def Tensor.rmsnorm (a : Tensor) (eps : Float) : Tensor :=
  let (y, rms) := rmsnormFwd a.data eps
  { data := y, id := 0, requiresGrad := a.requiresGrad, gradFn := .rmsnormOp a rms }

def Tensor.attn (cfg : Config) (xPre wq wk wv wo : Tensor) : Tensor :=
  let (out, cache) := attnFwd cfg xPre.data wq.data wk.data wv.data wo.data
  { data := out, id := 0,
    requiresGrad := xPre.requiresGrad || wq.requiresGrad || wk.requiresGrad
                    || wv.requiresGrad || wo.requiresGrad,
    gradFn := .attnOp xPre wq wk wv wo cache cfg }

def Tensor.mlp (cfg : Config) (xPre fc1 fc2 : Tensor) : Tensor :=
  let (out, cache) := mlpFwd cfg xPre.data fc1.data fc2.data
  { data := out, id := 0,
    requiresGrad := xPre.requiresGrad || fc1.requiresGrad || fc2.requiresGrad,
    gradFn := .mlpOp xPre fc1 fc2 cache cfg }

-- scalar loss. data is a 1×1 matrix so backward can start with [[1.0]].
def Tensor.maskedCE (logits : Tensor) (targets : Array Nat) (mask : Array Float) : Tensor :=
  let probs := logits.data.map softmax
  let sumMask := mask.foldl (init := 0.0) (· + ·)
  let l := maskedCrossEntropy probs targets mask sumMask
  { data := #[#[l]], id := 0,
    requiresGrad := logits.requiresGrad,
    gradFn := .lossOp logits probs targets mask sumMask }

/-! ## Backward

`Tensor.backward(loss)` returns the accumulated gradient for every leaf with
`requiresGrad=true`, keyed by its `id`. Uses a tiny Array (Nat × Matrix) as the
map since # of params is small (≤ 9 for 1-layer microGPT). For shared leaves
(a param used multiple times in forward) the recursive visits add into the same
slot — same accumulation semantics as autograd.c's `accumulate_grad`.
-/

private def gmAdd (gm : Array (Nat × Matrix)) (id : Nat) (g : Matrix) : Array (Nat × Matrix) :=
  match gm.findIdx? (fun (i, _) => i = id) with
  | some i => gm.set! i (id, madd gm[i]!.2 g)
  | none => gm.push (id, g)

-- recursion is over the (acyclic, finite) graph, but Lean's termination checker
-- can't see this without an explicit depth bound. `partial def` is the pragma.
partial def Tensor.backwardAcc (t : Tensor) (incoming : Matrix)
    (gm : Array (Nat × Matrix)) : Array (Nat × Matrix) :=
  match t.gradFn with
  | .leaf =>
    if t.requiresGrad then gmAdd gm t.id incoming else gm
  | .addOp a b =>
    let gm := a.backwardAcc incoming gm
    b.backwardAcc incoming gm
  | .linearOp x w =>
    let dx := linearBwdX incoming w.data
    let dw := linearBwdW incoming x.data
    let gm := x.backwardAcc dx gm
    w.backwardAcc dw gm
  | .rmsnormOp a rms =>
    a.backwardAcc (rmsnormBwd incoming a.data rms) gm
  | .gather table ids =>
    let dTable := scatterAdd table.data.size (Matrix.cols table.data) incoming ids
    table.backwardAcc dTable gm
  | .attnOp xPre wq wk wv wo cache cfg =>
    let (dxPre, (dWq, dWk, dWv, dWo)) :=
      attnBwd cfg incoming wq.data wk.data wv.data wo.data cache
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
    -- scale fused softmax+CE backward by the incoming dLoss
    let scale := if incoming.size > 0 && incoming[0]!.size > 0 then incoming[0]![0]! else 1.0
    let dLogits := maskedCrossEntropyBwd probs targets mask sumMask
    let scaled : Matrix := dLogits.map (·.map (scale * ·))
    logits.backwardAcc scaled gm

-- entry point: start with d_loss/d_loss = 1
def Tensor.backward (loss : Tensor) : Array (Nat × Matrix) :=
  loss.backwardAcc #[#[1.0]] #[]

-- look up a leaf's grad by id; zeros-shape if missing (shouldn't happen for
-- params we actually trained on this step).
def gradFor (gm : Array (Nat × Matrix)) (id : Nat) (shape : Nat × Nat) : Matrix :=
  match gm.find? (fun (i, _) => i = id) with
  | some (_, g) => g
  | none => Matrix.zeros shape.1 shape.2

end Autograd
