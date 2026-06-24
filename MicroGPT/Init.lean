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

private def initParams (cfg : Config) (rng : UInt64) : Params × UInt64 := Id.run do
  let σ : Float := 0.08
  let mkLeaf (data : Array Float) (rows cols id : Nat) : Tensor :=
    Tensor.leaf data rows cols id true
  let (wte, r1) := rngGaussFlat cfg.vocabSize cfg.nEmbed σ rng
  let (wpe, r2) := rngGaussFlat cfg.blockSize cfg.nEmbed σ r1
  let (lmHead, r3) := rngGaussFlat cfg.nEmbed cfg.vocabSize σ r2
  let mut r := r3
  let mut blocks : Array TransformerBlock := #[]
  for h in [0:cfg.nLayer] do
    let (wq, r') := rngGaussFlat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (wk, r') := rngGaussFlat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (wv, r') := rngGaussFlat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (wo, r') := rngGaussFlat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (fc1, r') := rngGaussFlat cfg.nEmbed (4 * cfg.nEmbed) σ r; r := r'
    let (fc2, r') := rngGaussFlat (4 * cfg.nEmbed) cfg.nEmbed σ r; r := r'
    blocks := blocks.push { attnWq := mkLeaf wq cfg.nEmbed cfg.nEmbed (ParamIds.attnWq h), attnWk := mkLeaf wk cfg.nEmbed cfg.nEmbed (ParamIds.attnWk h), attnWv := mkLeaf wv cfg.nEmbed cfg.nEmbed (ParamIds.attnWv h), attnWo := mkLeaf wo cfg.nEmbed cfg.nEmbed (ParamIds.attnWo h), mlpFc1 := mkLeaf fc1 cfg.nEmbed (4 * cfg.nEmbed) (ParamIds.mlpFc1 h), mlpFc2 := mkLeaf fc2 (4 * cfg.nEmbed) cfg.nEmbed (ParamIds.mlpFc2 h) }
  return ({ wte := mkLeaf wte cfg.vocabSize cfg.nEmbed ParamIds.wte, wpe := mkLeaf wpe cfg.blockSize cfg.nEmbed ParamIds.wpe, lmHead := mkLeaf lmHead cfg.nEmbed cfg.vocabSize ParamIds.lmHead, blocks := blocks }, r)

-- tests
#guard let (p, _) := initParams { nLayer := 1, nEmbed := 2, blockSize := 2, nHead := 1, vocabSize := 3, numSteps := 1 } 123; p.wte.data.size == 6 && p.wte.id == ParamIds.wte && p.wte.requiresGrad && p.blocks.size == 1 && p.blocks[0]!.mlpFc1.data.size == 16
#guard arrApproxEq (initParams { nLayer := 1, nEmbed := 2, blockSize := 2, nHead := 1, vocabSize := 3, numSteps := 1 } 123).1.wte.data (initParams { nLayer := 1, nEmbed := 2, blockSize := 2, nHead := 1, vocabSize := 3, numSteps := 1 } 123).1.wte.data
#guard !arrApproxEq (initParams { nLayer := 1, nEmbed := 2, blockSize := 2, nHead := 1, vocabSize := 3, numSteps := 1 } 1).1.wte.data (initParams { nLayer := 1, nEmbed := 2, blockSize := 2, nHead := 1, vocabSize := 3, numSteps := 1 } 2).1.wte.data

end MicroGPT
