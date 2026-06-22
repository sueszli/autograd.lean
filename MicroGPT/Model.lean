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
  let x : Tensor := p.blocks.foldl (init := xInit) fun acc b =>
    Tensor.mlp mlpCfg (Tensor.attn attnCfg acc b.attnWq b.attnWk b.attnWv b.attnWo) b.mlpFc1 b.mlpFc2
  (x @ p.lmHead).maskedCE target mask

end MicroGPT
