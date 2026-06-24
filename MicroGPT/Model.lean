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

theorem toAttnConfig_nEmbed (c : Config) : c.toAttnConfig.nEmbed = c.nEmbed := rfl
theorem toAttnConfig_nHead (c : Config) : c.toAttnConfig.nHead = c.nHead := rfl
theorem toMlpConfig_nEmbed (c : Config) : c.toMlpConfig.nEmbed = c.nEmbed := rfl
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

theorem default_params_no_blocks : (default : Params).blocks.size = 0 := rfl
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

-- ids must be globally distinct: a collision would make `backwardAcc` sum unrelated gradients.
theorem block_ids_increasing (h : Nat) : ParamIds.attnWq h < ParamIds.attnWk h ∧ ParamIds.attnWk h < ParamIds.attnWv h ∧ ParamIds.attnWv h < ParamIds.attnWo h ∧ ParamIds.attnWo h < ParamIds.mlpFc1 h ∧ ParamIds.mlpFc1 h < ParamIds.mlpFc2 h := by unfold ParamIds.attnWq ParamIds.attnWk ParamIds.attnWv ParamIds.attnWo ParamIds.mlpFc1 ParamIds.mlpFc2 ParamIds.blockBase; omega
theorem blocks_disjoint (h : Nat) : ParamIds.mlpFc2 h < ParamIds.attnWq (h + 1) := by unfold ParamIds.mlpFc2 ParamIds.attnWq ParamIds.blockBase; omega
theorem globals_precede_blocks : ParamIds.lmHead < ParamIds.attnWq 0 := by unfold ParamIds.lmHead ParamIds.attnWq ParamIds.blockBase; omega

theorem blockBase_eq (h : Nat) : ParamIds.blockBase h = 3 + h * 6 := rfl
theorem global_ids : ParamIds.wte = 0 ∧ ParamIds.wpe = 1 ∧ ParamIds.lmHead = 2 := ⟨rfl, rfl, rfl⟩
theorem block0_ids : ParamIds.attnWq 0 = 3 ∧ ParamIds.mlpFc2 0 = 8 := ⟨rfl, rfl⟩
theorem block1_ids : ParamIds.blockBase 1 = 9 ∧ ParamIds.attnWq 1 = 9 := ⟨rfl, rfl⟩

/-!
===--------------------------------------------------------------------------===
Parameter id injectivity

The theorems above pin a few concrete ids; these prove the layout is
collision-free for EVERY model size. `Slot` enumerates the parameter slots (3 globals plus 6 roles
per block), `Slot.id` mirrors the `ParamIds` assignment, and the theorems show
that map is injective and that `allIds` is exactly `List.range (3 + 6n)`: a
gap-free bijection onto the gradient-buffer indices, so `backwardAcc` can never
fold two distinct weights into one slot.
===--------------------------------------------------------------------------===
-/

inductive Role where
  | attnWq | attnWk | attnWv | attnWo | mlpFc1 | mlpFc2
  deriving DecidableEq

inductive Slot where
  | wte
  | wpe
  | lmHead
  | block (h : Nat) (r : Role)
  deriving DecidableEq

def Role.offset : Role → Nat
  | .attnWq => 0
  | .attnWk => 1
  | .attnWv => 2
  | .attnWo => 3
  | .mlpFc1 => 4
  | .mlpFc2 => 5

def Slot.id : Slot → Nat
  | .wte => ParamIds.wte
  | .wpe => ParamIds.wpe
  | .lmHead => ParamIds.lmHead
  | .block h r => ParamIds.blockBase h + r.offset

theorem Role.offset_lt (r : Role) : r.offset < 6 := by cases r <;> decide

theorem Role.offset_inj (r : Role) (r' : Role) (h : r.offset = r'.offset) : r = r' := by cases r <;> cases r' <;> simp_all [Role.offset]

theorem Slot.id_injective (s : Slot) (t : Slot) (h : s.id = t.id) : s = t := by
  cases s <;> cases t <;>
    simp_all [Slot.id, ParamIds.wte, ParamIds.wpe, ParamIds.lmHead, ParamIds.blockBase]
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

theorem Slot.id_lt (s : Slot) (n : Nat) (hb : ∀ h r, s = Slot.block h r → h < n) : s.id < 3 + 6 * n := by
  cases s with
  | wte => simp only [Slot.id, ParamIds.wte]; omega
  | wpe => simp only [Slot.id, ParamIds.wpe]; omega
  | lmHead => simp only [Slot.id, ParamIds.lmHead]; omega
  | block h r =>
    have := hb h r rfl
    have := r.offset_lt
    simp only [Slot.id, ParamIds.blockBase]
    omega

def ParamIds.blockIds (h : Nat) : List Nat :=
  [ParamIds.attnWq h, ParamIds.attnWk h, ParamIds.attnWv h, ParamIds.attnWo h, ParamIds.mlpFc1 h, ParamIds.mlpFc2 h]

def ParamIds.allIds : Nat → List Nat
  | 0 => [ParamIds.wte, ParamIds.wpe, ParamIds.lmHead]
  | n + 1 => ParamIds.allIds n ++ ParamIds.blockIds n

theorem rangeAddSix (m : Nat) : List.range (m + 6) = List.range m ++ [m, m + 1, m + 2, m + 3, m + 4, m + 5] := by rw [List.range_add]; rfl

theorem ParamIds.allIds_eq_range (n : Nat) : ParamIds.allIds n = List.range (3 + 6 * n) := by
  induction n with
  | zero => rfl
  | succ k ih =>
    unfold ParamIds.allIds
    rw [ih]
    have e1 : 3 + 6 * (k + 1) = (3 + 6 * k) + 6 := by omega
    rw [e1, rangeAddSix]
    simp only [ParamIds.blockIds, ParamIds.attnWq, ParamIds.attnWk, ParamIds.attnWv, ParamIds.attnWo, ParamIds.mlpFc1, ParamIds.mlpFc2, ParamIds.blockBase]
    have h0 : 3 + k * 6 + 0 = 3 + 6 * k := by omega
    have h1 : 3 + k * 6 + 1 = 3 + 6 * k + 1 := by omega
    have h2 : 3 + k * 6 + 2 = 3 + 6 * k + 2 := by omega
    have h3 : 3 + k * 6 + 3 = 3 + 6 * k + 3 := by omega
    have h4 : 3 + k * 6 + 4 = 3 + 6 * k + 4 := by omega
    have h5 : 3 + k * 6 + 5 = 3 + 6 * k + 5 := by omega
    rw [h0, h1, h2, h3, h4, h5]

theorem ParamIds.allIds_nodup (n : Nat) : (ParamIds.allIds n).Nodup := by rw [ParamIds.allIds_eq_range]; exact List.nodup_range

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

#guard
  let mk := fun (rows : Nat) (cols : Nat) (id : Nat) => Tensor.leaf (Array.replicate (rows * cols) 0.0) rows cols id true
  let blk : TransformerBlock := { attnWq := mk 2 2 (ParamIds.attnWq 0), attnWk := mk 2 2 (ParamIds.attnWk 0), attnWv := mk 2 2 (ParamIds.attnWv 0), attnWo := mk 2 2 (ParamIds.attnWo 0), mlpFc1 := mk 2 8 (ParamIds.mlpFc1 0), mlpFc2 := mk 8 2 (ParamIds.mlpFc2 0) }
  let p : Params := { wte := mk 2 2 ParamIds.wte, wpe := mk 2 2 ParamIds.wpe, lmHead := mk 2 2 ParamIds.lmHead, blocks := #[blk] }
  let cfg : Config := { nLayer := 1, nEmbed := 2, blockSize := 2, nHead := 1, vocabSize := 2, numSteps := 1 }
  let loss := forward p cfg #[0, 1] #[0, 1] #[1, 1]
  let gm := loss.backward
  loss.shape == #[1, 1] && approxEq loss.data[0]! (-Float.log 0.5) && (gm.find? (fun e => e.1 == ParamIds.lmHead)).isSome

end MicroGPT
