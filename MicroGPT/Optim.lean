import Autograd.Ops
import Autograd.Tensor
import Autograd.Optim
import MicroGPT.Model

namespace MicroGPT
open Autograd

/-!
===--------------------------------------------------------------------------===
Param tensors
===--------------------------------------------------------------------------===
-/

def paramTensors (p : Params) : Array Tensor := Id.run do
  let mut a : Array Tensor := #[p.wte, p.wpe, p.lmHead]
  for b in p.blocks do
    a := a.push b.attnWq |>.push b.attnWk |>.push b.attnWv |>.push b.attnWo |>.push b.mlpFc1 |>.push b.mlpFc2
  return a

def ofTensors (ws : Array Tensor) : Params :=
  let nBlocks := (ws.size - 3) / 6
  { wte := ws[0]!, wpe := ws[1]!, lmHead := ws[2]!,
    blocks := (Array.range nBlocks).map fun h =>
      let o := 3 + h * 6
      { attnWq := ws[o]!, attnWk := ws[o + 1]!, attnWv := ws[o + 2]!, attnWo := ws[o + 3]!, mlpFc1 := ws[o + 4]!, mlpFc2 := ws[o + 5]! } }

private def testParams : Params :=
  let mk := fun (rows : Nat) (cols : Nat) (id : Nat) => Tensor.leaf (Array.replicate (rows * cols) 0.0) rows cols id true
  let blk : TransformerBlock := { attnWq := mk 2 2 (ParamIds.attnWq 0), attnWk := mk 2 2 (ParamIds.attnWk 0), attnWv := mk 2 2 (ParamIds.attnWv 0), attnWo := mk 2 2 (ParamIds.attnWo 0), mlpFc1 := mk 2 8 (ParamIds.mlpFc1 0), mlpFc2 := mk 8 2 (ParamIds.mlpFc2 0) }
  { wte := mk 3 2 ParamIds.wte, wpe := mk 2 2 ParamIds.wpe, lmHead := mk 2 3 ParamIds.lmHead, blocks := #[blk] }

-- tests
#guard (paramTensors testParams).size == 9
#guard (ofTensors (paramTensors testParams)).blocks.size == 1
#guard
  let ws := paramTensors testParams
  let (m, v) := zeroMoments ws
  let (ws', _, _) := adamWStep 1 ws m v #[] 10
  let p' := ofTensors ws'
  arrApproxEq p'.wte.data testParams.wte.data && arrApproxEq p'.blocks[0]!.mlpFc2.data testParams.blocks[0]!.mlpFc2.data

end MicroGPT
