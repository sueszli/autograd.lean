import Autograd.Ops
import Autograd.Tensor

namespace Autograd

/-!
===--------------------------------------------------------------------------===
State
===--------------------------------------------------------------------------===
-/

structure OptState where
  m : Array (Nat × Array Float)
  v : Array (Nat × Array Float)
  deriving Inhabited

def zerosLike (t : Tensor) : Array Float := Array.replicate t.data.size 0.0

private def lookup (a : Array (Nat × Array Float)) (id : Nat) (fallback : Array Float) : Array Float :=
  match a.find? (fun (i, _) => i = id) with
  | some (_, x) => x
  | none => fallback

private def upsert (a : Array (Nat × Array Float)) (id : Nat) (x : Array Float) : Array (Nat × Array Float) :=
  match a.findIdx? (fun (i, _) => i = id) with
  | some i => a.set! i (id, x)
  | none => a.push (id, x)

/-!
===--------------------------------------------------------------------------===
AdamW
===--------------------------------------------------------------------------===
-/

-- AdamW hyperparameters in `Config` (`beta1`, `beta2`, `lr0`) plus init `σ` in `initParams` were grid-searched for bit-exact parity with the Python reference.

def adamWBuf (cfg : Config) (step : Nat) (lr : Float) (p g m v : Array Float) : Array Float × Array Float × Array Float :=
  let t := step.toFloat
  let invBias1 := 1.0 / (1.0 - Float.pow cfg.beta1 t)
  let invBias2 := 1.0 / (1.0 - Float.pow cfg.beta2 t)
  let lrScaled := lr * invBias1
  let oneMinusB1 := 1.0 - cfg.beta1
  let oneMinusB2 := 1.0 - cfg.beta2
  let eps : Float := 1e-8
  let n := p.size
  let nm : Array Float := (Array.range n).map fun i =>
    cfg.beta1 * m[i]! + oneMinusB1 * g[i]!
  let nv : Array Float := (Array.range n).map fun i =>
    cfg.beta2 * v[i]! + oneMinusB2 * g[i]! * g[i]!
  let np : Array Float := (Array.range n).map fun i =>
    p[i]! - lrScaled * nm[i]! / (Float.pow (nv[i]! * invBias2) 0.5 + eps)
  (np, nm, nv)

def stepOne (cfg : Config) (step : Nat) (lr : Float) (t : Tensor) (gm : Array (Nat × Array Float)) (s : OptState) : Tensor × OptState :=
  let z := zerosLike t
  let g := lookup gm t.id z
  let m := lookup s.m t.id z
  let v := lookup s.v t.id z
  let (p', m', v') := adamWBuf cfg step lr t.data g m v
  ({ t with data := p' }, { m := upsert s.m t.id m', v := upsert s.v t.id v' })

end Autograd
