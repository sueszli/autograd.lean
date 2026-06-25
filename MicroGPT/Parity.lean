import MicroGPT.Utils
import MicroGPT.Model
import Lean.Data.Json

open Autograd
open MicroGPT
open Lean (Json fromJson?)

/-!
===--------------------------------------------------------------------------===
Weight loader
===--------------------------------------------------------------------------===
-/

private def paramsFromJson (j : Json) : IO Params := do
  return { wte := leafFrom (← asRows (← getObj j "wte")) ParamIds.wte, wpe := leafFrom (← asRows (← getObj j "wpe")) ParamIds.wpe, lmHead := leafFrom (← asRowsT (← getObj j "lm_head")) ParamIds.lmHead, blocks := #[{ attnWq := leafFrom (← asRowsT (← getObj j "layer0.attn_wq")) (ParamIds.attnWq 0), attnWk := leafFrom (← asRowsT (← getObj j "layer0.attn_wk")) (ParamIds.attnWk 0), attnWv := leafFrom (← asRowsT (← getObj j "layer0.attn_wv")) (ParamIds.attnWv 0), attnWo := leafFrom (← asRowsT (← getObj j "layer0.attn_wo")) (ParamIds.attnWo 0), mlpFc1 := leafFrom (← asRowsT (← getObj j "layer0.mlp_fc1")) (ParamIds.mlpFc1 0), mlpFc2 := leafFrom (← asRowsT (← getObj j "layer0.mlp_fc2")) (ParamIds.mlpFc2 0) }] }

private def miniWeights : String := "{\"wte\":[[1]],\"wpe\":[[2]],\"lm_head\":[[3]],\"layer0.attn_wq\":[[4]],\"layer0.attn_wk\":[[5]],\"layer0.attn_wv\":[[6]],\"layer0.attn_wo\":[[7]],\"layer0.mlp_fc1\":[[8]],\"layer0.mlp_fc2\":[[9]]}"

-- tests
#eval do
  let p ← paramsFromJson (← IO.ofExcept (Json.parse miniWeights))
  unless p.wte.id == ParamIds.wte && p.lmHead.id == ParamIds.lmHead && p.blocks[0]!.attnWq.id == ParamIds.attnWq 0 && p.blocks[0]!.mlpFc2.id == ParamIds.mlpFc2 0 do throw (IO.userError "paramsFromJson ids")
  unless arrApproxEq p.wte.data #[1] && arrApproxEq p.lmHead.data #[3] && arrApproxEq p.blocks[0]!.mlpFc2.data #[9] do throw (IO.userError "paramsFromJson data")

/-!
===--------------------------------------------------------------------------===
Parity entrypoint
===--------------------------------------------------------------------------===
-/

def main : IO Unit := do
  let raw ← IO.FS.readFile "data/parity.json"
  let j ← match Json.parse raw with
    | .ok j => pure j
    | .error e => throw (IO.userError s!"parse: {e}")
  let nLayer := 1
  let nEmbed := 16
  let nHead := 4
  let numSteps := 1000
  let inputs : Array (Array Nat) ← IO.ofExcept (fromJson? (← getObj j "inputs"))
  let targets : Array (Array Nat) ← IO.ofExcept (fromJson? (← getObj j "targets"))
  let masks : Array (Array Float) ← IO.ofExcept (fromJson? (← getObj j "masks"))
  let initP ← paramsFromJson (← getObj j "init_weights")
  let refP ← paramsFromJson (← getObj j "final_weights")

  let mut p := initP
  let (m0, v0) := zeroMoments initP
  let mut m := m0
  let mut v := v0
  let startMs ← IO.monoMsNow
  let stdout ← IO.getStdout
  for step in [0:numSteps] do
    let lossT := forward p inputs[step]! targets[step]! masks[step]! nEmbed nHead
    let (p', m', v') := adamWStep (step + 1) p m v lossT.backward
    p := p'; m := m'; v := v'
    let elapsed := (← IO.monoMsNow) - startMs
    IO.print s!"\r{tqdm (step + 1) numSteps elapsed}  "
    stdout.flush
  IO.println ""

  let atol : Float := 1e-11
  let atolStr := "1e-11"
  let globals : Array (String × Tensor × Tensor) := #[("wte", p.wte, refP.wte), ("wpe", p.wpe, refP.wpe), ("lm_head", p.lmHead, refP.lmHead)]
  let pairs : Array (String × Tensor × Tensor) := globals ++ (Array.range nLayer).flatMap fun h =>
    let b := p.blocks[h]!
    let r := refP.blocks[h]!
    #[(s!"layer{h}.attn_wq", b.attnWq, r.attnWq), (s!"layer{h}.attn_wk", b.attnWk, r.attnWk), (s!"layer{h}.attn_wv", b.attnWv, r.attnWv), (s!"layer{h}.attn_wo", b.attnWo, r.attnWo), (s!"layer{h}.mlp_fc1", b.mlpFc1, r.mlpFc1), (s!"layer{h}.mlp_fc2", b.mlpFc2, r.mlpFc2)]
  let failures : Array (String × Float) := pairs.filterMap fun (label, a, b) =>
    let mx := arrMaxDiff a.data b.data
    if mx > atol then some (label, mx) else none
  let totalMs := (← IO.monoMsNow) - startMs
  let ms := totalMs % 1000
  let msStr := String.leftpad 3 '0' (toString ms)
  if failures.isEmpty then
    IO.println s!"passed parity check (atol={atolStr})"
    IO.println s!"total time: {totalMs / 1000}.{msStr}s"
  else
    for (label, mx) in failures do
      IO.eprintln s!"  {label}: max |Δ| = {mx}"
    throw (IO.userError s!"FAIL: {failures.size}/{pairs.size} tensors exceed atol={atolStr}")
