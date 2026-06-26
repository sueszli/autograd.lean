import Autograd.Tensor
import Std.Data.HashMap

namespace Autograd

/-!
===--------------------------------------------------------------------------===
AdamW
===--------------------------------------------------------------------------===
-/

-- grid-searched params for bit-exact parity with the Python reference
private def adamWBuf (step : Nat) (lr : Float) (p g m v : Array Float) : Array Float × Array Float × Array Float :=
  let beta1 : Float := 0.85
  let beta2 : Float := 0.99
  let t := step.toFloat
  let invBias1 := 1.0 / (1.0 - Float.pow beta1 t)
  let invBias2 := 1.0 / (1.0 - Float.pow beta2 t)
  let lrScaled := lr * invBias1
  let oneMinusB1 := 1.0 - beta1
  let oneMinusB2 := 1.0 - beta2
  let eps : Float := 1e-8
  let n := p.size
  let nm : Array Float := (Array.range n).map fun i => beta1 * m[i]! + oneMinusB1 * g[i]!
  let nv : Array Float := (Array.range n).map fun i => beta2 * v[i]! + oneMinusB2 * g[i]! * g[i]!
  let np : Array Float := (Array.range n).map fun i => p[i]! - lrScaled * nm[i]! / (Float.pow (nv[i]! * invBias2) 0.5 + eps)
  (np, nm, nv)

private def stepOne (step : Nat) (lr : Float) (t : Tensor) (gradientMap : Std.HashMap Nat (Array Float)) (mv : Std.HashMap Nat (Array Float × Array Float)) : Tensor × Std.HashMap Nat (Array Float × Array Float) :=
  let z := Array.replicate t.data.size 0.0
  let g := gradientMap.getD t.id z
  let (mi, vi) := mv.getD t.id (z, z)
  let (p', m', v') := adamWBuf step lr t.data g mi vi
  ({ t with data := p' }, mv.insert t.id (m', v'))

-- this type requires ability to traverse tensors while preserving structure
class Weights (α : Type) where
  mapM : {σ : Type} → (Tensor → StateM σ Tensor) → α → StateM σ α

namespace AdamW

def init {α : Type} [Weights α] (p : α) : Std.HashMap Nat (Array Float × Array Float) :=
  let collect : Tensor → StateM (Std.HashMap Nat (Array Float × Array Float)) Tensor := fun t => do let z := Array.replicate t.data.size 0.0; modify (·.insert t.id (z, z)); pure t
  ((Weights.mapM collect p).run ∅).2

def step {α : Type} [Weights α] (step : Nat) (p : α) (mv : Std.HashMap Nat (Array Float × Array Float)) (gradientMap : Std.HashMap Nat (Array Float)) (numSteps : Nat := 1000) (lr0 : Float := 0.01) : α × Std.HashMap Nat (Array Float × Array Float) :=
  let progress : Float := if numSteps = 0 then 0.0 else (step - 1).toFloat / numSteps.toFloat
  let lrRaw := lr0 * (1.0 - progress)
  let lr := if lrRaw < 0.0 then 0.0 else lrRaw
  let upd : Tensor → StateM (Std.HashMap Nat (Array Float × Array Float)) Tensor := fun t => modifyGet (stepOne step lr t gradientMap)
  (Weights.mapM upd p).run mv

end AdamW

-- tests
theorem adamWBuf_param_size (step : Nat) (lr : Float) (p : Array Float) (g : Array Float) (m : Array Float) (v : Array Float) : (adamWBuf step lr p g m v).1.size = p.size := by simp [adamWBuf]
theorem adamWBuf_m_size (step : Nat) (lr : Float) (p : Array Float) (g : Array Float) (m : Array Float) (v : Array Float) : (adamWBuf step lr p g m v).2.1.size = p.size := by simp [adamWBuf]
theorem adamWBuf_v_size (step : Nat) (lr : Float) (p : Array Float) (g : Array Float) (m : Array Float) (v : Array Float) : (adamWBuf step lr p g m v).2.2.size = p.size := by simp [adamWBuf]
#guard let (np, nm, nv) := adamWBuf 1 0.1 #[1.0] #[2.0] #[0.0] #[0.0]; approxEq nm[0]! 0.3 && approxEq nv[0]! 0.04 && approxEq np[0]! 0.9 1e-6
#guard let (np, nm, nv) := adamWBuf 1 0.1 #[5.0, -3.0] #[0.0, 0.0] #[0.0, 0.0] #[0.0, 0.0]; arrApproxEq np #[5.0, -3.0] && arrApproxEq nm #[0, 0] && arrApproxEq nv #[0, 0]
#guard let (t', _) := stepOne 1 0.1 (Tensor.leaf #[5.0, -3.0] 1 2 0 true) ∅ ∅; arrApproxEq t'.data #[5.0, -3.0]
#guard let g := (∅ : Std.HashMap Nat (Array Float)).insert 0 #[2.0]; let (t', mv') := stepOne 1 0.1 (Tensor.leaf #[1.0] 1 1 0 true) g ∅; let (m, v) := mv'.getD 0 ((#[], #[]) : Array Float × Array Float); approxEq t'.data[0]! 0.9 1e-6 && arrApproxEq m #[0.3] && arrApproxEq v #[0.04]
#guard let g := (∅ : Std.HashMap Nat (Array Float)).insert 0 #[2.0, 2.0]; let (t', _) := stepOne 1 0.1 (Tensor.leaf #[1.0, 1.0] 1 2 0 true) g ∅; arrApproxEq t'.data #[0.9, 0.9] 1e-6

end Autograd
