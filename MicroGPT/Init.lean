import Autograd.Ops
import Autograd.Tensor
import Autograd.Utils
import MicroGPT.Model

namespace MicroGPT
open Autograd

/-!
===--------------------------------------------------------------------------===
Random init
===--------------------------------------------------------------------------===
-/

private def initParams (nLayer nEmbed blockSize vocabSize : Nat) (rng : UInt64) : Params × UInt64 := Id.run do
  let σ : Float := 0.08
  let mkLeaf (data : Array Float) (rows cols id : Nat) : Tensor :=
    Tensor.leaf data rows cols id true
  let (wte, r1) := rngGaussFlat vocabSize nEmbed σ rng
  let (wpe, r2) := rngGaussFlat blockSize nEmbed σ r1
  let (lmHead, r3) := rngGaussFlat nEmbed vocabSize σ r2
  let mut r := r3
  let mut blocks : Array TransformerBlock := #[]
  for h in [0:nLayer] do
    let (wq, r') := rngGaussFlat nEmbed nEmbed σ r; r := r'
    let (wk, r') := rngGaussFlat nEmbed nEmbed σ r; r := r'
    let (wv, r') := rngGaussFlat nEmbed nEmbed σ r; r := r'
    let (wo, r') := rngGaussFlat nEmbed nEmbed σ r; r := r'
    let (fc1, r') := rngGaussFlat nEmbed (4 * nEmbed) σ r; r := r'
    let (fc2, r') := rngGaussFlat (4 * nEmbed) nEmbed σ r; r := r'
    blocks := blocks.push { attnWq := mkLeaf wq nEmbed nEmbed (ParamIds.attnWq h), attnWk := mkLeaf wk nEmbed nEmbed (ParamIds.attnWk h), attnWv := mkLeaf wv nEmbed nEmbed (ParamIds.attnWv h), attnWo := mkLeaf wo nEmbed nEmbed (ParamIds.attnWo h), mlpFc1 := mkLeaf fc1 nEmbed (4 * nEmbed) (ParamIds.mlpFc1 h), mlpFc2 := mkLeaf fc2 (4 * nEmbed) nEmbed (ParamIds.mlpFc2 h) }
  return ({ wte := mkLeaf wte vocabSize nEmbed ParamIds.wte, wpe := mkLeaf wpe blockSize nEmbed ParamIds.wpe, lmHead := mkLeaf lmHead nEmbed vocabSize ParamIds.lmHead, blocks := blocks }, r)

-- tests
#guard let (p, _) := initParams 1 2 2 3 123; p.wte.data.size == 6 && p.wte.id == ParamIds.wte && p.wte.requiresGrad && p.blocks.size == 1 && p.blocks[0]!.mlpFc1.data.size == 16
#guard arrApproxEq (initParams 1 2 2 3 123).1.wte.data (initParams 1 2 2 3 123).1.wte.data
#guard !arrApproxEq (initParams 1 2 2 3 1).1.wte.data (initParams 1 2 2 3 2).1.wte.data

end MicroGPT
