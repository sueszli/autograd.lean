import Autograd.Ops
import Autograd.Tensor
import Autograd.Optim
import MicroGPT.Model

namespace MicroGPT
open Autograd

/-!
===--------------------------------------------------------------------------===
Weights instance
===--------------------------------------------------------------------------===
-/

instance : Weights Params where
  mapM f p := do
    let wte ← f p.wte
    let wpe ← f p.wpe
    let lmHead ← f p.lmHead
    let blocks ← p.blocks.mapM fun b => do
      let attnWq ← f b.attnWq
      let attnWk ← f b.attnWk
      let attnWv ← f b.attnWv
      let attnWo ← f b.attnWo
      let mlpFc1 ← f b.mlpFc1
      let mlpFc2 ← f b.mlpFc2
      pure { attnWq, attnWk, attnWv, attnWo, mlpFc1, mlpFc2 }
    pure { wte, wpe, lmHead, blocks }

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
