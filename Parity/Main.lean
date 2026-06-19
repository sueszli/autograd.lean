import Autograd.Ops
import Autograd.Tensor
import MicroGPT.Init
import MicroGPT.Model
import MicroGPT.Optim
import Lean.Data.Json

open Autograd
open MicroGPT
open Lean (Json)

/-!
===--------------------------------------------------------------------------===
JSON primitives
===--------------------------------------------------------------------------===
-/

private def asArr (j : Json) : IO (Array Json) :=
  match j.getArr? with
  | .ok a => pure a
  | .error e => throw (IO.userError s!"expected array: {e}")

private def asNat (j : Json) : IO Nat :=
  match j.getNat? with
  | .ok n => pure n
  | .error e => throw (IO.userError s!"expected nat: {e}")

private def asFloat (j : Json) : IO Float :=
  match j.getNum? with
  | .ok n => pure n.toFloat
  | .error e => throw (IO.userError s!"expected number: {e}")

private def getObj (j : Json) (k : String) : IO Json :=
  match j.getObjVal? k with
  | .ok v => pure v
  | .error e => throw (IO.userError s!"missing key '{k}': {e}")

private def parseNatRows (j : Json) : IO (Array (Array Nat)) := do
  (← asArr j).mapM fun row => do (← asArr row).mapM asNat

private def parseFloatRows (j : Json) : IO (Array (Array Float)) := do
  (← asArr j).mapM fun row => do (← asArr row).mapM asFloat

/-!
===--------------------------------------------------------------------------===
Tensor JSON loaders
===--------------------------------------------------------------------------===
-/

private def asRows (j : Json) : IO (Array Float × Nat × Nat) := do
  let rows ← asArr j
  let r := rows.size
  let c : Nat := ← if r = 0 then pure 0 else do let row0 ← asArr rows[0]!; pure row0.size
  let flat ← rows.foldlM (init := (#[] : Array Float)) fun acc row => do
    let cols ← asArr row
    return acc ++ (← cols.mapM asFloat)
  return (flat, r, c)

-- Python stores linear weights `[out × in]`, `linearFwd` wants `[in × out]`. Transpose.
private def asRowsT (j : Json) : IO (Array Float × Nat × Nat) := do
  let (flat, r, c) ← asRows j
  return (transposeFlat flat r c, c, r)

private def leafFrom (triple : Array Float × Nat × Nat) (id : Nat) : Tensor :=
  Tensor.leaf triple.1 triple.2.1 triple.2.2 id true

/-!
===--------------------------------------------------------------------------===
Weight loader
===--------------------------------------------------------------------------===
-/

private def paramsFromJson (j : Json) : IO Params := do
  return { wte := leafFrom (← asRows (← getObj j "wte")) ParamIds.wte, wpe := leafFrom (← asRows (← getObj j "wpe")) ParamIds.wpe, lmHead := leafFrom (← asRowsT (← getObj j "lm_head")) ParamIds.lmHead, blocks := #[{ attnWq := leafFrom (← asRowsT (← getObj j "layer0.attn_wq")) (ParamIds.attnWq 0), attnWk := leafFrom (← asRowsT (← getObj j "layer0.attn_wk")) (ParamIds.attnWk 0), attnWv := leafFrom (← asRowsT (← getObj j "layer0.attn_wv")) (ParamIds.attnWv 0), attnWo := leafFrom (← asRowsT (← getObj j "layer0.attn_wo")) (ParamIds.attnWo 0), mlpFc1 := leafFrom (← asRowsT (← getObj j "layer0.mlp_fc1")) (ParamIds.mlpFc1 0), mlpFc2 := leafFrom (← asRowsT (← getObj j "layer0.mlp_fc2")) (ParamIds.mlpFc2 0) }] }

/-!
===--------------------------------------------------------------------------===
Tensor diff
===--------------------------------------------------------------------------===
-/

private def maxDiff (a b : Tensor) : Float :=
  (Array.range a.data.size).foldl (init := 0.0) fun mx i =>
    let d := (a.data[i]! - b.data[i]!).abs
    if d > mx then d else mx

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
  let vocabSize ← asNat (← getObj j "vocab_size")
  let cfg : Config := { nLayer := 1, nEmbed := 16, blockSize := 16, nHead := 4, vocabSize := vocabSize, numSteps := 1000 }
  let inputs ← parseNatRows (← getObj j "inputs")
  let targets ← parseNatRows (← getObj j "targets")
  let masks ← parseFloatRows (← getObj j "masks")
  let initP ← paramsFromJson (← getObj j "init_weights")
  let refP ← paramsFromJson (← getObj j "final_weights")

  let mut p := initP
  let mut s := OptState.zeros initP
  for step in [0:cfg.numSteps] do
    let lossT := forward p cfg inputs[step]! targets[step]! masks[step]!
    let (p', s') := adamWStep cfg (step + 1) p s lossT.backward
    p := p'; s := s'
    if step = 0 || (step + 1) % 100 = 0 then
      IO.println s!"step {step + 1}/{cfg.numSteps}  loss={lossT.data[0]!}"

  let atol : Float := 1e-11
  let atolStr := "1e-11"
  let pairs : Array (String × Tensor × Tensor) := Id.run do
    let mut a : Array (String × Tensor × Tensor) := #[("wte", p.wte, refP.wte), ("wpe", p.wpe, refP.wpe), ("lm_head", p.lmHead, refP.lmHead)]
    for h in [0:cfg.nLayer] do
      let b := p.blocks[h]!
      let r := refP.blocks[h]!
      a := a.push (s!"layer{h}.attn_wq", b.attnWq, r.attnWq)
      a := a.push (s!"layer{h}.attn_wk", b.attnWk, r.attnWk)
      a := a.push (s!"layer{h}.attn_wv", b.attnWv, r.attnWv)
      a := a.push (s!"layer{h}.attn_wo", b.attnWo, r.attnWo)
      a := a.push (s!"layer{h}.mlp_fc1", b.mlpFc1, r.mlpFc1)
      a := a.push (s!"layer{h}.mlp_fc2", b.mlpFc2, r.mlpFc2)
    return a
  let mut failures : Array (String × Float) := #[]
  let mut cells : Nat := 0
  for (label, a, b) in pairs do
    cells := cells + a.data.size
    let mx := maxDiff a b
    if mx > atol then failures := failures.push (label, mx)
  if failures.isEmpty then
    IO.println s!"PASS: all {cells} weight cells within atol={atolStr}"
  else
    for (label, mx) in failures do
      IO.eprintln s!"  {label}: max |Δ| = {mx}"
    throw (IO.userError s!"FAIL: {failures.size}/{pairs.size} tensors exceed atol={atolStr}")
