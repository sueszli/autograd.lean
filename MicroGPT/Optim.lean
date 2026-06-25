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

def zeroMoments (p : Params) : Array (Nat × Array Float) × Array (Nat × Array Float) :=
  let leaves : Array Tensor := Id.run do
    let mut a : Array Tensor := #[p.wte, p.wpe, p.lmHead]
    for b in p.blocks do
      a := a.push b.attnWq |>.push b.attnWk |>.push b.attnWv |>.push b.attnWo |>.push b.mlpFc1 |>.push b.mlpFc2
    return a
  let entries := leaves.map fun t => (t.id, zerosLike t)
  (entries, entries)

def adamWStep (step : Nat) (p : Params) (m v : Array (Nat × Array Float)) (gradientMap : Array (Nat × Array Float)) (numSteps : Nat := 1000) (lr0 : Float := 0.01) : Params × Array (Nat × Array Float) × Array (Nat × Array Float) :=
  let progress : Float := if numSteps = 0 then 0.0 else (step - 1).toFloat / numSteps.toFloat
  let lrRaw := lr0 * (1.0 - progress)
  let lr := if lrRaw < 0.0 then 0.0 else lrRaw
  let (wte', m, v) := stepOne step lr p.wte gradientMap m v
  let (wpe', m, v) := stepOne step lr p.wpe gradientMap m v
  let (lm',  m, v) := stepOne step lr p.lmHead gradientMap m v
  let (blocks, mFinal, vFinal) : Array TransformerBlock × Array (Nat × Array Float) × Array (Nat × Array Float) := Id.run do
    let mut acc : Array TransformerBlock := #[]
    let mut m' := m
    let mut v' := v
    for b in p.blocks do
      let (wq, mq, vq) := stepOne step lr b.attnWq gradientMap m' v'; m' := mq; v' := vq
      let (wk, mk, vk) := stepOne step lr b.attnWk gradientMap m' v'; m' := mk; v' := vk
      let (wv, mw, vw) := stepOne step lr b.attnWv gradientMap m' v'; m' := mw; v' := vw
      let (wo, mo, vo) := stepOne step lr b.attnWo gradientMap m' v'; m' := mo; v' := vo
      let (f1, m5, v5) := stepOne step lr b.mlpFc1 gradientMap m' v'; m' := m5; v' := v5
      let (f2, m6, v6) := stepOne step lr b.mlpFc2 gradientMap m' v'; m' := m6; v' := v6
      acc := acc.push { attnWq := wq, attnWk := wk, attnWv := wv, attnWo := wo, mlpFc1 := f1, mlpFc2 := f2 }
    return (acc, m', v')
  ({ wte := wte', wpe := wpe', lmHead := lm', blocks := blocks }, mFinal, vFinal)

private def testParams : Params :=
  let mk := fun (rows : Nat) (cols : Nat) (id : Nat) => Tensor.leaf (Array.replicate (rows * cols) 0.0) rows cols id true
  let blk : TransformerBlock := { attnWq := mk 2 2 (ParamIds.attnWq 0), attnWk := mk 2 2 (ParamIds.attnWk 0), attnWv := mk 2 2 (ParamIds.attnWv 0), attnWo := mk 2 2 (ParamIds.attnWo 0), mlpFc1 := mk 2 8 (ParamIds.mlpFc1 0), mlpFc2 := mk 8 2 (ParamIds.mlpFc2 0) }
  { wte := mk 3 2 ParamIds.wte, wpe := mk 2 2 ParamIds.wpe, lmHead := mk 2 3 ParamIds.lmHead, blocks := #[blk] }

-- tests
#guard (zeroMoments testParams).1.size == 9 && (zeroMoments testParams).2.size == 9
#guard
  let (m, v) := zeroMoments testParams
  let (p', _, _) := adamWStep 1 testParams m v #[] 10
  arrApproxEq p'.wte.data testParams.wte.data && arrApproxEq p'.blocks[0]!.mlpFc2.data testParams.blocks[0]!.mlpFc2.data

end MicroGPT
