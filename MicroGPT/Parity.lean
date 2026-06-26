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

private def getObj (j : Json) (k : String) : IO Json := IO.ofExcept <| (j.getObjVal? k).mapError fun e => s!"missing key '{k}': {e}"

private def asRows (j : Json) : IO (Array Float × Nat × Nat) := IO.ofExcept <| (fun rows => (rows.flatten, rows.size, if 0 < rows.size then rows[0]!.size else 0)) <$> (fromJson? j : Except String (Array (Array Float)))

-- Python stores linear weights `[out × in]`, `matmulFwd` wants `[in × out]`. Transpose.
private def asRowsT (j : Json) : IO (Array Float × Nat × Nat) := (fun (flat, r, c) => (transposeFlat flat r c, c, r)) <$> asRows j

private def leafFrom (triple : Array Float × Nat × Nat) (id : Nat) : Tensor := Tensor.leaf triple.1 triple.2.1 triple.2.2 id true

private def paramsFromJson (j : Json) : IO Params := do
  return { wte := leafFrom (← asRows (← getObj j "wte")) ParamIds.wte, wpe := leafFrom (← asRows (← getObj j "wpe")) ParamIds.wpe, lmHead := leafFrom (← asRowsT (← getObj j "lm_head")) ParamIds.lmHead, blocks := #[{ attnWq := leafFrom (← asRowsT (← getObj j "layer0.attn_wq")) (ParamIds.attnWq 0), attnWk := leafFrom (← asRowsT (← getObj j "layer0.attn_wk")) (ParamIds.attnWk 0), attnWv := leafFrom (← asRowsT (← getObj j "layer0.attn_wv")) (ParamIds.attnWv 0), attnWo := leafFrom (← asRowsT (← getObj j "layer0.attn_wo")) (ParamIds.attnWo 0), mlpFc1 := leafFrom (← asRowsT (← getObj j "layer0.mlp_fc1")) (ParamIds.mlpFc1 0), mlpFc2 := leafFrom (← asRowsT (← getObj j "layer0.mlp_fc2")) (ParamIds.mlpFc2 0) }] }

private def miniWeights : String := "{\"wte\":[[1]],\"wpe\":[[2]],\"lm_head\":[[3]],\"layer0.attn_wq\":[[4]],\"layer0.attn_wk\":[[5]],\"layer0.attn_wv\":[[6]],\"layer0.attn_wo\":[[7]],\"layer0.mlp_fc1\":[[8]],\"layer0.mlp_fc2\":[[9]]}"

-- tests
-- `#eval`-as-test: a throwing IO action fails `lake build`, same as a false `#guard`
#eval do
  let j ← IO.ofExcept (Json.parse "[[1, 2, 3], [4, 5, 6]]")
  let (flat, r, c) ← asRows j
  unless r == 2 && c == 3 && arrApproxEq flat #[1, 2, 3, 4, 5, 6] do throw (IO.userError "asRows")
  let (flatT, rT, cT) ← asRowsT j
  unless rT == 3 && cT == 2 && arrApproxEq flatT #[1, 4, 2, 5, 3, 6] do throw (IO.userError "asRowsT")
  let t := leafFrom (flat, r, c) 7
  unless t.id == 7 && t.rows == 2 && t.cols == 3 do throw (IO.userError "leafFrom")
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
  let mut mv := AdamW.init initP
  let startMs ← IO.monoMsNow
  let stdout ← IO.getStdout
  for step in [0:numSteps] do
    let lossT := forward p inputs[step]! targets[step]! masks[step]! nEmbed nHead
    let (p', mv') := AdamW.step (step + 1) p mv lossT.backward
    p := p'; mv := mv'
    let elapsed := (← IO.monoMsNow) - startMs
    IO.print s!"\r{tqdm (step + 1) numSteps elapsed}  "
    stdout.flush
  IO.println ""

  let atol : Float := 1e-11
  let atolStr := "1e-11"
  let pairs : Array (String × Tensor × Tensor) :=
    #[("wte", p.wte, refP.wte), ("wpe", p.wpe, refP.wpe), ("lm_head", p.lmHead, refP.lmHead)] ++
    (Array.range nLayer).flatMap fun h =>
      let b := p.blocks[h]!
      let r := refP.blocks[h]!
      #[
        (s!"layer{h}.attn_wq", b.attnWq, r.attnWq),
        (s!"layer{h}.attn_wk", b.attnWk, r.attnWk),
        (s!"layer{h}.attn_wv", b.attnWv, r.attnWv),
        (s!"layer{h}.attn_wo", b.attnWo, r.attnWo),
        (s!"layer{h}.mlp_fc1", b.mlpFc1, r.mlpFc1),
        (s!"layer{h}.mlp_fc2", b.mlpFc2, r.mlpFc2)
      ]
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
