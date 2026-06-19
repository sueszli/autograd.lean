import MicroGPT.Init
import Lean.Data.Json

open Autograd
open MicroGPT
open Lean (Json)

def main : IO Unit := do
  let raw ← IO.FS.readFile "data/parity.json"
  let j ← match Json.parse raw with
    | .ok j => pure j
    | .error e => throw (IO.userError s!"parse: {e}")
  let getObj k := match j.getObjVal? k with
    | .ok v => pure v | .error e => throw (IO.userError e)
  let getNat k := do
    match (← getObj k).getNat? with
    | .ok n => pure n | .error e => throw (IO.userError e)
  let getNum k := do
    match (← getObj k).getNum? with
    | .ok n => pure n.toFloat | .error e => throw (IO.userError e)
  let getArr k := do
    match (← getObj k).getArr? with
    | .ok a => pure a | .error e => throw (IO.userError e)
  let vocabSize ← getNat "vocab_size"
  let n ← getNat "n"
  let bos ← getNat "bos"
  let lossPy ← getNum "loss"
  let cfg : Config := {
    nLayer := 1, nEmbed := 16, blockSize := 16, nHead := 4,
    vocabSize := vocabSize, numSteps := 0
  }
  let p ← paramsFromJson (← getObj "weights")
  let tokens : Array Nat ← (← getArr "tokens").mapM fun t => match t.getNat? with
    | .ok n => pure n | .error e => throw (IO.userError e)
  let logitsPy : Array (Array Float) ← (← getArr "logits").mapM fun row =>
    match row.getArr? with
    | .ok a => a.mapM fun c => match c.getNum? with
                | .ok x => pure x.toFloat | .error e => throw (IO.userError e)
    | .error e => throw (IO.userError e)
  let input : Array Nat := (Array.range cfg.blockSize).map fun i =>
    if i < n then tokens[i]! else bos
  let target : Array Nat := (Array.range cfg.blockSize).map fun i =>
    if i < n then tokens[i + 1]! else bos
  let mask : Array Float := (Array.range cfg.blockSize).map fun i =>
    if i < n then 1.0 else 0.0
  let logitsT := forwardLogits p cfg input
  let cols := logitsT.cols
  let lossLean := (forward p cfg input target mask).data[0]!
  -- bit-level cell-by-cell comparison
  let mut maxDiff : Float := 0.0
  let mut nz : Nat := 0
  let mut total : Nat := 0
  for i in [0:n] do
    let py := logitsPy[i]!
    for jj in [0:py.size] do
      let d := (py[jj]! - logitsT.data[i * cols + jj]!).abs
      total := total + 1
      if d > 0.0 then nz := nz + 1
      if d > maxDiff then maxDiff := d
  let bucket (d : Float) : String :=
    if d == 0.0 then "exact bit-equivalence (0.0)"
    else if d < 1e-15 then "< 1e-15"
    else if d < 1e-12 then "< 1e-12"
    else if d < 1e-9  then "< 1e-9"
    else if d < 1e-6  then "< 1e-6"
    else if d < 1e-3  then "< 1e-3"
    else s!"≥ 1e-3"
  IO.println s!"loss_py    = {lossPy}"
  IO.println s!"loss_lean  = {lossLean}"
  IO.println s!"|Δloss|    = {bucket (lossLean - lossPy).abs}  (raw: {(lossLean - lossPy).abs})"
  IO.println s!"logit cells: {total} total, {nz} with nonzero diff"
  IO.println s!"max |Δlogits| = {bucket maxDiff}  (raw: {maxDiff})"
