import MicroGPT.Init
import MicroGPT.Optim
import Lean.Data.Json

open Autograd
open MicroGPT
open Lean (Json)

private def asArr (j : Json) : IO (Array Json) :=
  match j.getArr? with | .ok a => pure a | .error e => throw (IO.userError e)

private def asNat (j : Json) : IO Nat :=
  match j.getNat? with | .ok n => pure n | .error e => throw (IO.userError e)

private def asFloat (j : Json) : IO Float :=
  match j.getNum? with | .ok n => pure n.toFloat | .error e => throw (IO.userError e)

private def getObj (j : Json) (k : String) : IO Json :=
  match j.getObjVal? k with | .ok v => pure v | .error e => throw (IO.userError e)

private def parseNatRows (j : Json) : IO (Array (Array Nat)) := do
  (← asArr j).mapM fun row => do (← asArr row).mapM asNat

private def parseFloatRows (j : Json) : IO (Array (Array Float)) := do
  (← asArr j).mapM fun row => do (← asArr row).mapM asFloat

-- count cells exceeding `atol` and the max |Δ| over the whole tensor
private def diff (a b : Tensor) (atol : Float) : Nat × Float :=
  (Array.range a.data.size).foldl (init := (0, 0.0)) fun (vio, mx) i =>
    let d := (a.data[i]! - b.data[i]!).abs
    (if d > atol then vio + 1 else vio, if d > mx then d else mx)

def main : IO Unit := do
  let raw ← IO.FS.readFile "data/parity.json"
  let j ← match Json.parse raw with
    | .ok j => pure j
    | .error e => throw (IO.userError s!"parse: {e}")
  let vocabSize ← asNat (← getObj j "vocab_size")
  let cfg : Config := {
    nLayer := 1, nEmbed := 16, blockSize := 16, nHead := 4,
    vocabSize := vocabSize, numSteps := 1000
  }
  let inputs ← parseNatRows (← getObj j "inputs")
  let targets ← parseNatRows (← getObj j "targets")
  let masks ← parseFloatRows (← getObj j "masks")
  let initP ← paramsFromJson (← getObj j "init_weights")
  let refP ← paramsFromJson (← getObj j "final_weights")
  IO.println s!"loaded vocab={vocabSize}, training {cfg.numSteps} steps..."

  let mut p := initP
  let mut s := OptState.zeros initP
  for step in [0:cfg.numSteps] do
    let lossT := forward p cfg inputs[step]! targets[step]! masks[step]!
    let (p', s') := adamWStep cfg (step + 1) p s lossT.backward
    p := p'; s := s'
    if step = 0 || (step + 1) % 100 = 0 then
      IO.println s!"step {step + 1}/{cfg.numSteps}  loss={lossT.data[0]!}"

  let atol : Float := 1e-5
  IO.println s!"comparing against reference final weights at atol={atol}:"
  let pairs : Array (String × Tensor × Tensor) := Id.run do
    let mut a : Array (String × Tensor × Tensor) :=
      #[("wte", p.wte, refP.wte),
        ("wpe", p.wpe, refP.wpe),
        ("lm_head", p.lmHead, refP.lmHead)]
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
  let mut totalVio : Nat := 0
  let mut totalCells : Nat := 0
  let mut globalMax : Float := 0.0
  for (label, a, b) in pairs do
    let (vio, mx) := diff a b atol
    IO.println s!"  {label}: {vio}/{a.data.size} violations, max |Δ| = {mx}"
    totalVio := totalVio + vio
    totalCells := totalCells + a.data.size
    if mx > globalMax then globalMax := mx
  IO.println s!"TOTAL: {totalVio}/{totalCells} violations, max |Δ| = {globalMax}"
  if totalVio = 0 then
    IO.println s!"PASS: weights match within atol={atol}"
  else
    IO.println s!"FAIL: {totalVio} cells exceed atol={atol}"
