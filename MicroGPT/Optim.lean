import Autograd.Ops
import Autograd.Tensor
import Autograd.Optim
import MicroGPT.Model

namespace MicroGPT
open Autograd

/-!
===--------------------------------------------------------------------------===
Params walker
===--------------------------------------------------------------------------===
-/

private def paramLeaves (p : Params) : Array Tensor := Id.run do
  let mut a : Array Tensor := #[p.wte, p.wpe, p.lmHead]
  for b in p.blocks do
    a := a.push b.attnWq |>.push b.attnWk |>.push b.attnWv |>.push b.attnWo |>.push b.mlpFc1 |>.push b.mlpFc2
  return a

def OptState.zeros (p : Params) : OptState :=
  let entries := (paramLeaves p).map fun t => (t.id, zerosLike t)
  { m := entries, v := entries }

def adamWStep (cfg : Config) (step : Nat) (p : Params) (s : OptState) (gradientMap : Array (Nat × Array Float)) : Params × OptState :=
  let adamCfg := cfg.toAdamWConfig
  let progress : Float := if cfg.numSteps = 0 then 0.0 else (step - 1).toFloat / cfg.numSteps.toFloat
  let lr0 := cfg.lr0 * (1.0 - progress)
  let lr := if lr0 < 0.0 then 0.0 else lr0
  let (wte', s) := stepOne adamCfg step lr p.wte gradientMap s
  let (wpe', s) := stepOne adamCfg step lr p.wpe gradientMap s
  let (lm',  s) := stepOne adamCfg step lr p.lmHead gradientMap s
  let (blocks, sFinal) : Array TransformerBlock × OptState := Id.run do
    let mut acc : Array TransformerBlock := #[]
    let mut s' := s
    for b in p.blocks do
      let (wq, s1) := stepOne adamCfg step lr b.attnWq gradientMap s'; s' := s1
      let (wk, s2) := stepOne adamCfg step lr b.attnWk gradientMap s'; s' := s2
      let (wv, s3) := stepOne adamCfg step lr b.attnWv gradientMap s'; s' := s3
      let (wo, s4) := stepOne adamCfg step lr b.attnWo gradientMap s'; s' := s4
      let (f1, s5) := stepOne adamCfg step lr b.mlpFc1 gradientMap s'; s' := s5
      let (f2, s6) := stepOne adamCfg step lr b.mlpFc2 gradientMap s'; s' := s6
      acc := acc.push { attnWq := wq, attnWk := wk, attnWv := wv, attnWo := wo, mlpFc1 := f1, mlpFc2 := f2 }
    return (acc, s')
  ({ wte := wte', wpe := wpe', lmHead := lm', blocks := blocks }, sFinal)

end MicroGPT
