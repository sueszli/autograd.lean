import Autograd.Tensor
import Autograd.Optim

namespace MicroGPT
open Autograd

/-!
===--------------------------------------------------------------------------===
MicroGPT
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

inductive Role where
  | attnWq
  | attnWk
  | attnWv
  | attnWo
  | mlpFc1
  | mlpFc2

inductive Slot where
  | wte
  | wpe
  | lmHead
  | block (h : Nat) (r : Role)

private def Role.offset : Role → Nat
  | .attnWq => 0
  | .attnWk => 1
  | .attnWv => 2
  | .attnWo => 3
  | .mlpFc1 => 4
  | .mlpFc2 => 5

def Slot.id : Slot → Nat
  | .wte => 0
  | .wpe => 1
  | .lmHead => 2
  | .block h r => 3 + h * 6 + r.offset

def forward (p : Params) (input target : Array Nat) (mask : Array Float) (nEmbed nHead : Nat) (epsilon : Float := 1e-5) (maskValue : Float := -1.0e9) : Tensor :=
  let tokEmb := p.wte.gather input
  let posEmb := p.wpe.gather (Array.range input.size)
  let xInit := (tokEmb + posEmb).rmsnorm epsilon
  let x : Tensor := p.blocks.foldl (init := xInit) fun acc b => Tensor.mlp nEmbed epsilon (Tensor.attn nEmbed nHead epsilon maskValue acc b.attnWq b.attnWk b.attnWv b.attnWo) b.mlpFc1 b.mlpFc2
  (x @ p.lmHead).maskedCE target mask

-- every role offset is < 6
theorem Role.offset_lt (r : Role)
    : r.offset < 6 := by cases r <;> decide
-- distinct roles have distinct offsets
theorem Role.offset_inj (r : Role) (r' : Role) (h : r.offset = r'.offset)
    : r = r' := by cases r <;> cases r' <;> simp_all [Role.offset]

-- ids must be globally distinct: a collision would make `backwardAcc` sum unrelated gradients.
-- distinct slots get distinct ids (no collision, for any model size)
theorem Slot.id_injective (s : Slot) (t : Slot) (h : s.id = t.id)
    : s = t := by
  cases s <;> cases t <;> simp_all [Slot.id]
  case wte.block hh r | wpe.block hh r | lmHead.block hh r =>
    have := r.offset_lt; omega
  case block.wpe hh r | block.lmHead hh r =>
    have := r.offset_lt; omega
  case block.block ha r hb r' =>
    have h1 := r.offset_lt
    have h2 := r'.offset_lt
    refine ⟨by omega, ?_⟩
    apply Role.offset_inj
    omega

-- tests
theorem default_params_no_blocks : (default : Params).blocks.size = 0 := rfl
theorem global_ids : (Slot.wte).id = 0 ∧ (Slot.wpe).id = 1 ∧ (Slot.lmHead).id = 2 := ⟨rfl, rfl, rfl⟩
theorem block0_ids : (Slot.block 0 .attnWq).id = 3 ∧ (Slot.block 0 .mlpFc2).id = 8 := ⟨rfl, rfl⟩
theorem block1_ids : (Slot.block 1 .attnWq).id = 9 ∧ (Slot.block 1 .mlpFc2).id = 14 := ⟨rfl, rfl⟩
#guard let p : Params := { wte := Tensor.leaf #[1] 1 1 0 true, wpe := Tensor.leaf #[2] 1 1 1 true, lmHead := Tensor.leaf #[3] 1 1 2 true, blocks := #[] }; arrApproxEq p.wte.data #[1] && p.blocks.size == 0
#guard
  let mk := fun (rows : Nat) (cols : Nat) (id : Nat) => Tensor.leaf (Array.replicate (rows * cols) 0.0) rows cols id true
  let blk : TransformerBlock := { attnWq := mk 2 2 (Slot.block 0 .attnWq).id, attnWk := mk 2 2 (Slot.block 0 .attnWk).id, attnWv := mk 2 2 (Slot.block 0 .attnWv).id, attnWo := mk 2 2 (Slot.block 0 .attnWo).id, mlpFc1 := mk 2 8 (Slot.block 0 .mlpFc1).id, mlpFc2 := mk 8 2 (Slot.block 0 .mlpFc2).id }
  let p : Params := { wte := mk 2 2 (Slot.wte).id, wpe := mk 2 2 (Slot.wpe).id, lmHead := mk 2 2 (Slot.lmHead).id, blocks := #[blk] }
  let loss := forward p #[0, 1] #[0, 1] #[1, 1] 2 1
  let gm := loss.backward
  loss.shape == #[1, 1] && approxEq loss.data[0]! (-Float.log 0.5) && gm.contains (Slot.lmHead).id

end MicroGPT
