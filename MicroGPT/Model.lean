import Autograd.Ops
import Autograd.Tensor
import Autograd.Optim

namespace MicroGPT
open Autograd

/-!
===--------------------------------------------------------------------------===
Config
===--------------------------------------------------------------------------===
-/

structure Config where
  nLayer : Nat
  nEmbed : Nat
  blockSize : Nat
  nHead : Nat
  vocabSize : Nat
  numSteps : Nat
  epsilon : Float := 1e-5
  lr0 : Float := 0.01
  beta1 : Float := 0.85
  beta2 : Float := 0.99
  maskValue : Float := -1.0e9
  deriving Inhabited

def Config.toAttnConfig (c : Config) : AttnConfig :=
  { nEmbed := c.nEmbed, nHead := c.nHead, epsilon := c.epsilon, maskValue := c.maskValue }

def Config.toMlpConfig (c : Config) : MlpConfig :=
  { nEmbed := c.nEmbed, epsilon := c.epsilon }

def Config.toAdamWConfig (c : Config) : AdamWConfig :=
  { beta1 := c.beta1, beta2 := c.beta2 }

-- the per-op config projections copy the relevant fields out of the master `Config`
#guard let c : Config := { nLayer := 1, nEmbed := 16, blockSize := 8, nHead := 4, vocabSize := 10, numSteps := 5 }; c.toAttnConfig.nEmbed == 16 && c.toAttnConfig.nHead == 4 && c.toMlpConfig.nEmbed == 16
#guard let c : Config := { nLayer := 1, nEmbed := 16, blockSize := 8, nHead := 4, vocabSize := 10, numSteps := 5 }; approxEq c.toAdamWConfig.beta1 0.85 && approxEq c.toAdamWConfig.beta2 0.99

/-!
===--------------------------------------------------------------------------===
Parameter structures
===--------------------------------------------------------------------------===
-/

structure TransformerBlock where
  attnWq : Tensor
  attnWk : Tensor
  attnWv : Tensor
  attnWo : Tensor
  mlpFc1 : Tensor
  mlpFc2 : Tensor
  deriving Inhabited

structure Params where
  wte : Tensor
  wpe : Tensor
  lmHead : Tensor
  blocks : Array TransformerBlock
  deriving Inhabited

#guard (default : Params).blocks.size == 0
#guard let p : Params := { wte := Tensor.leaf #[1] 1 1 0 true, wpe := Tensor.leaf #[2] 1 1 1 true, lmHead := Tensor.leaf #[3] 1 1 2 true, blocks := #[] }; arrApproxEq p.wte.data #[1] && p.blocks.size == 0

/-!
===--------------------------------------------------------------------------===
Parameter ids
===--------------------------------------------------------------------------===
-/

namespace ParamIds
def wte : Nat := 0
def wpe : Nat := 1
def lmHead : Nat := 2
def blockBase (h : Nat) : Nat := 3 + h * 6
def attnWq (h : Nat) : Nat := blockBase h + 0
def attnWk (h : Nat) : Nat := blockBase h + 1
def attnWv (h : Nat) : Nat := blockBase h + 2
def attnWo (h : Nat) : Nat := blockBase h + 3
def mlpFc1 (h : Nat) : Nat := blockBase h + 4
def mlpFc2 (h : Nat) : Nat := blockBase h + 5
end ParamIds

#guard ParamIds.wte == 0 && ParamIds.wpe == 1 && ParamIds.lmHead == 2
#guard ParamIds.attnWq 0 == 3 && ParamIds.mlpFc2 0 == 8                        -- block 0 occupies ids 3..8
#guard ParamIds.blockBase 1 == 9 && ParamIds.attnWq 1 == 9                     -- block 1 starts right after, no overlap

/-!
===--------------------------------------------------------------------------===
Forward pass
===--------------------------------------------------------------------------===
-/

def forward (p : Params) (cfg : Config) (input target : Array Nat) (mask : Array Float) : Tensor :=
  let attnCfg := cfg.toAttnConfig
  let mlpCfg := cfg.toMlpConfig
  let tokEmb := p.wte.gather input
  let posEmb := p.wpe.gather (Array.range input.size)
  let xInit := (tokEmb + posEmb).rmsnorm cfg.epsilon
  let x : Tensor := p.blocks.foldl (init := xInit) fun acc b => Tensor.mlp mlpCfg (Tensor.attn attnCfg acc b.attnWq b.attnWk b.attnWv b.attnWo) b.mlpFc1 b.mlpFc2
  (x @ p.lmHead).maskedCE target mask

-- end-to-end smoke test: zero weights drive every logit to 0, so softmax is uniform `0.5` and the
-- masked cross-entropy is `-log 0.5` per token. Also confirms backward reaches `lm_head`.
#guard
  let mk := fun (rows : Nat) (cols : Nat) (id : Nat) => Tensor.leaf (Array.replicate (rows * cols) 0.0) rows cols id true
  let blk : TransformerBlock := { attnWq := mk 2 2 (ParamIds.attnWq 0), attnWk := mk 2 2 (ParamIds.attnWk 0), attnWv := mk 2 2 (ParamIds.attnWv 0), attnWo := mk 2 2 (ParamIds.attnWo 0), mlpFc1 := mk 2 8 (ParamIds.mlpFc1 0), mlpFc2 := mk 8 2 (ParamIds.mlpFc2 0) }
  let p : Params := { wte := mk 2 2 ParamIds.wte, wpe := mk 2 2 ParamIds.wpe, lmHead := mk 2 2 ParamIds.lmHead, blocks := #[blk] }
  let cfg : Config := { nLayer := 1, nEmbed := 2, blockSize := 2, nHead := 1, vocabSize := 2, numSteps := 1 }
  let loss := forward p cfg #[0, 1] #[0, 1] #[1, 1]
  let gm := loss.backward
  loss.shape == #[1, 1] && approxEq loss.data[0]! (-Float.log 0.5) && (gm.find? (fun e => e.1 == ParamIds.lmHead)).isSome

end MicroGPT
