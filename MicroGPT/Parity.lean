import MicroGPT.Model
import Lean.Data.Json

open Autograd
open MicroGPT
open Lean (Json fromJson?)

/-!
===--------------------------------------------------------------------------===
JSON weight loading
===--------------------------------------------------------------------------===
-/

private def jsonField (j : Json) (k : String) : IO Json := IO.ofExcept <| (j.getObjVal? k).mapError fun e => s!"missing key '{k}': {e}"
private def decodeField {α : Type} [Lean.FromJson α] (j : Json) (k : String) : IO α := do IO.ofExcept (fromJson? (← jsonField j k))
private def loadMatrix (j : Json) : IO (Array Float × Nat × Nat) := IO.ofExcept <| (fun rows => (rows.flatten, rows.size, if 0 < rows.size then rows[0]!.size else 0)) <$> (fromJson? j : Except String (Array (Array Float)))
private def loadMatrixT (j : Json) : IO (Array Float × Nat × Nat) := (fun (flat, r, c) => (transposeFlat flat r c, c, r)) <$> loadMatrix j
private def leafTensor (triple : Array Float × Nat × Nat) (id : Nat) : Tensor := Tensor.leaf triple.1 triple.2.1 triple.2.2 id true

private def paramsFromJson (j : Json) : IO Params := do return { wte := leafTensor (← loadMatrix (← jsonField j "wte")) (Slot.wte).id, wpe := leafTensor (← loadMatrix (← jsonField j "wpe")) (Slot.wpe).id, lmHead := leafTensor (← loadMatrixT (← jsonField j "lm_head")) (Slot.lmHead).id, blocks := #[{ attnWq := leafTensor (← loadMatrixT (← jsonField j "layer0.attn_wq")) (Slot.block 0 .attnWq).id, attnWk := leafTensor (← loadMatrixT (← jsonField j "layer0.attn_wk")) (Slot.block 0 .attnWk).id, attnWv := leafTensor (← loadMatrixT (← jsonField j "layer0.attn_wv")) (Slot.block 0 .attnWv).id, attnWo := leafTensor (← loadMatrixT (← jsonField j "layer0.attn_wo")) (Slot.block 0 .attnWo).id, mlpFc1 := leafTensor (← loadMatrixT (← jsonField j "layer0.mlp_fc1")) (Slot.block 0 .mlpFc1).id, mlpFc2 := leafTensor (← loadMatrixT (← jsonField j "layer0.mlp_fc2")) (Slot.block 0 .mlpFc2).id }] }

/-!
===--------------------------------------------------------------------------===
Inference
===--------------------------------------------------------------------------===
-/

-- forward without the cross-entropy head
private def logits (p : Params) (input : Array Nat) (nEmbed : Nat) (nHead : Nat) (epsilon : Float := 1e-5) (maskValue : Float := -1.0e9) : Tensor :=
  let tokEmb := p.wte.gather input
  let posEmb := p.wpe.gather (Array.range input.size)
  let xInit := (tokEmb + posEmb).rmsnorm epsilon
  let x : Tensor := p.blocks.foldl (init := xInit) fun acc b => Tensor.mlp nEmbed epsilon (Tensor.attn nEmbed nHead epsilon maskValue acc b.attnWq b.attnWk b.attnWv b.attnWo) b.mlpFc1 b.mlpFc2
  x @ p.lmHead

-- sample a token
private def sampleRow (t : Tensor) (row : Nat) : StateM UInt64 Nat := do
  let c := t.cols
  let base := row * c
  let mut mx := t.data[base]!
  for j in [1:c] do mx := max mx t.data[base + j]!
  let mut exps : Array Float := Array.mkEmpty c
  let mut sum := 0.0
  for j in [0:c] do let e := (t.data[base + j]! - mx).exp; exps := exps.push e; sum := sum + e
  let u ← rngFloat
  let mut acc := 0.0
  let mut pick := c - 1
  for j in [0:c] do
    acc := acc + exps[j]! / sum
    if u < acc then pick := j; break
  return pick

-- numbers to characters
private def decode (boundary : Nat) (tokens : Array Nat) : String := tokens.foldl (init := "") fun s i => if i == boundary then s else s.push (Char.ofNat ('a'.toNat + i))

/-!
===--------------------------------------------------------------------------===
Parity check
===--------------------------------------------------------------------------===
-/

private def diffParams (p : Params) (refP : Params) (nLayer : Nat) : Array (String × Tensor × Tensor) :=
  #[("wte", p.wte, refP.wte), ("wpe", p.wpe, refP.wpe), ("lm_head", p.lmHead, refP.lmHead)] ++
  (Array.range nLayer).flatMap fun h =>
    let b := p.blocks[h]!
    let r := refP.blocks[h]!
    #[(s!"layer{h}.attn_wq", b.attnWq, r.attnWq), (s!"layer{h}.attn_wk", b.attnWk, r.attnWk), (s!"layer{h}.attn_wv", b.attnWv, r.attnWv), (s!"layer{h}.attn_wo", b.attnWo, r.attnWo), (s!"layer{h}.mlp_fc1", b.mlpFc1, r.mlpFc1), (s!"layer{h}.mlp_fc2", b.mlpFc2, r.mlpFc2)]

def main : IO Unit := do
  let raw ← IO.FS.readFile "data/parity.json"
  let j ← match Json.parse raw with
    | .ok j => pure j
    | .error e => throw (IO.userError s!"parse: {e}")
  let nLayer := 1
  let nEmbed := 16
  let nHead := 4
  let numSteps := 1000
  let inputs : Array (Array Nat) ← decodeField j "inputs"
  let targets : Array (Array Nat) ← decodeField j "targets"
  let masks : Array (Array Float) ← decodeField j "masks"
  let initP ← paramsFromJson (← jsonField j "init_weights")
  let refP ← paramsFromJson (← jsonField j "final_weights")

  -- train
  let mut p := initP
  let mut mv := AdamW.init initP
  let startMs ← IO.monoMsNow
  for step in [0:numSteps] do
    let lossT := forward p inputs[step]! targets[step]! masks[step]! nEmbed nHead
    let (p', mv') := AdamW.step (step + 1) p mv lossT.backward
    p := p'; mv := mv'
    tqdmTick startMs (step + 1) numSteps
  tqdmDone startMs

  -- parity
  let atol : Float := (10.0 : Float) ^ (-(11 : Float))
  let pairs := diffParams p refP nLayer
  let failures : Array (String × Float) := pairs.filterMap fun (label, a, b) =>
    let mx := arrMaxDiff a.data b.data
    if mx > atol then some (label, mx) else none
  if failures.isEmpty then
    IO.println s!"passed parity check"
  else
    for (label, mx) in failures do
      IO.eprintln s!"  {label}: max |Δ| = {mx}"
    throw (IO.userError s!"FAIL: {failures.size}/{pairs.size} tensors exceed atol")
  IO.println ""

  -- inference
  let bos : Nat ← decodeField j "bos"
  let blockSize := p.wpe.rows
  let mut seed := UInt64.ofNat (← IO.monoNanosNow)
  let mut tokens : Array Nat := #[bos]
  for _ in [0:blockSize - 1] do
    let (nxt, seed') : Nat × UInt64 := (sampleRow (logits p tokens nEmbed nHead) (tokens.size - 1)).run seed
    seed := seed'
    tokens := tokens.push nxt
    if nxt == bos then break
  IO.println s!"inference (random sampled name): {decode bos tokens}"
  IO.println ""
