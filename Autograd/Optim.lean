import Autograd.Matrix
import Autograd.Ops
import Autograd.Tensor
import Autograd.Model

namespace Autograd

-- per-parameter Adam state. Indexed by leaf id; we keep one Matrix per id.
structure OptState where
  m : Array (Nat × Matrix)
  v : Array (Nat × Matrix)
  deriving Inhabited

private def zerosLike (t : Tensor) : Matrix :=
  Matrix.zeros t.data.size (Matrix.cols t.data)

-- enumerate every leaf in `p` once; used to seed Adam state and apply updates.
private def paramLeaves (p : Params) : Array Tensor := Id.run do
  let mut a : Array Tensor := #[p.wte, p.wpe, p.lmHead]
  for b in p.blocks do
    a := a.push b.attnWq |>.push b.attnWk |>.push b.attnWv |>.push b.attnWo
            |>.push b.mlpFc1 |>.push b.mlpFc2
  return a

def OptState.zeros (p : Params) : OptState :=
  let entries := (paramLeaves p).map fun t => (t.id, zerosLike t)
  { m := entries, v := entries }

private def lookup (a : Array (Nat × Matrix)) (id : Nat) (fallback : Matrix) : Matrix :=
  match a.find? (fun (i, _) => i = id) with
  | some (_, x) => x
  | none => fallback

private def upsert (a : Array (Nat × Matrix)) (id : Nat) (x : Matrix) : Array (Nat × Matrix) :=
  match a.findIdx? (fun (i, _) => i = id) with
  | some i => a.set! i (id, x)
  | none => a.push (id, x)

-- one AdamW step on a single param: returns (p', m', v').
private def adamWMat (cfg : Config) (step : Nat) (lr : Float) (p g m v : Matrix)
    : Matrix × Matrix × Matrix :=
  let t := step.toFloat
  let b1t := 1.0 - Float.pow cfg.beta1 t
  let b2t := 1.0 - Float.pow cfg.beta2 t
  let eps : Float := 1e-8
  let n := p.size
  let cols := if n = 0 then 0 else p[0]!.size
  let nm : Matrix := (Array.range n).map fun i => (Array.range cols).map fun j =>
    cfg.beta1 * m[i]![j]! + (1.0 - cfg.beta1) * g[i]![j]!
  let nv : Matrix := (Array.range n).map fun i => (Array.range cols).map fun j =>
    cfg.beta2 * v[i]![j]! + (1.0 - cfg.beta2) * g[i]![j]! * g[i]![j]!
  let np : Matrix := (Array.range n).map fun i => (Array.range cols).map fun j =>
    (1.0 - lr * cfg.weightDecay) * p[i]![j]! -
      lr * (nm[i]![j]! / b1t) / (Float.sqrt (nv[i]![j]! / b2t) + eps)
  (np, nm, nv)

-- update one leaf tensor by id, returning (new tensor, new state).
private def stepOne (cfg : Config) (step : Nat) (lr : Float)
    (t : Tensor) (gm : Array (Nat × Matrix)) (s : OptState) : Tensor × OptState :=
  let z := zerosLike t
  let g := lookup gm t.id z
  let m := lookup s.m t.id z
  let v := lookup s.v t.id z
  let (p', m', v') := adamWMat cfg step lr t.data g m v
  ({ t with data := p' }, { m := upsert s.m t.id m', v := upsert s.v t.id v' })

def adamWStep (cfg : Config) (step : Nat) (p : Params) (s : OptState)
    (gm : Array (Nat × Matrix)) : Params × OptState :=
  let progress : Float := if cfg.numSteps = 0 then 0.0
                          else (step - 1).toFloat / cfg.numSteps.toFloat
  let lr0 := cfg.lr0 * (1.0 - progress)
  let lr := if lr0 < 0.0 then 0.0 else lr0
  let (wte', s) := stepOne cfg step lr p.wte gm s
  let (wpe', s) := stepOne cfg step lr p.wpe gm s
  let (lm',  s) := stepOne cfg step lr p.lmHead gm s
  let (blocks, sFinal) : Array TransformerBlock × OptState := Id.run do
    let mut acc : Array TransformerBlock := #[]
    let mut s' := s
    for b in p.blocks do
      let (wq, s1) := stepOne cfg step lr b.attnWq gm s'; s' := s1
      let (wk, s2) := stepOne cfg step lr b.attnWk gm s'; s' := s2
      let (wv, s3) := stepOne cfg step lr b.attnWv gm s'; s' := s3
      let (wo, s4) := stepOne cfg step lr b.attnWo gm s'; s' := s4
      let (f1, s5) := stepOne cfg step lr b.mlpFc1 gm s'; s' := s5
      let (f2, s6) := stepOne cfg step lr b.mlpFc2 gm s'; s' := s6
      acc := acc.push { attnWq := wq, attnWk := wk, attnWv := wv, attnWo := wo,
                        mlpFc1 := f1, mlpFc2 := f2 }
    return (acc, s')
  ({ wte := wte', wpe := wpe', lmHead := lm', blocks := blocks }, sFinal)

end Autograd
