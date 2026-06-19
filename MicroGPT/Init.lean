import Autograd.Ops
import Autograd.Tensor
import Autograd.Rng
import MicroGPT.Model

namespace MicroGPT
open Autograd

/-!
===--------------------------------------------------------------------------===
Random init
===--------------------------------------------------------------------------===
-/

def initParams (cfg : Config) (rng : RngState) : Params × RngState := Id.run do
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

end MicroGPT
