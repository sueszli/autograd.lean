import Autograd.Ops
import Autograd.Tensor

namespace MicroGPT
open Autograd

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
  let tokEmb := p.wte.gather input
  let posEmb := p.wpe.gather (Array.range input.size)
  let xInit := (tokEmb.add posEmb).rmsnorm cfg.epsilon
  let x : Tensor := p.blocks.foldl (init := xInit) fun acc b =>
    Tensor.mlp cfg (Tensor.attn cfg acc b.attnWq b.attnWk b.attnWv b.attnWo) b.mlpFc1 b.mlpFc2
  (x.linear p.lmHead).maskedCE target mask

end MicroGPT
