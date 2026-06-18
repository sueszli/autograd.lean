import Autograd.Ops
import Autograd.Tensor

namespace Autograd

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

-- mirrors original.py: rmsnorm-after-emb, per-layer (attn + residual) then (mlp + residual),
-- linear lm_head. Returns the pre-loss logits Tensor so parity-check can read it.
def forwardLogits (p : Params) (cfg : Config) (input : Array Nat) : Tensor :=
  let tokEmb := p.wte.gather input
  let posEmb := p.wpe.gather (Array.range input.size)
  let emb := tokEmb.add posEmb
  let xInit := emb.rmsnorm cfg.epsilon
  let x : Tensor := p.blocks.foldl (init := xInit) fun acc b =>
    let xa := Tensor.attn cfg acc b.attnWq b.attnWk b.attnWv b.attnWo
    Tensor.mlp cfg xa b.mlpFc1 b.mlpFc2
  x.linear p.lmHead

def forward (p : Params) (cfg : Config) (input target : Array Nat) (mask : Array Float) : Tensor :=
  (forwardLogits p cfg input).maskedCE target mask

end Autograd
