import Autograd
import Lean.Data.Json

open Autograd
open Lean (Json)

def main : IO Unit := do
  let raw ← IO.FS.readFile "data/parity.json"
  let j ← match Json.parse raw with
    | .ok j => pure j
    | .error e => throw (IO.userError s!"parse: {e}")
  let vocabSize : Nat ← match (← match j.getObjVal? "vocab_size" with | .ok v => pure v | .error e => throw (IO.userError e)).getNat? with
    | .ok n => pure n | .error e => throw (IO.userError e)
  let cfg : Config := {
    nLayer := 1, nEmbed := 16, blockSize := 16, nHead := 4,
    vocabSize := vocabSize, numSteps := 0
  }
  let weights ← match j.getObjVal? "weights" with | .ok v => pure v | .error e => throw (IO.userError e)
  let p ← paramsFromJson weights
  let getArr (k : String) : IO (Array Json) := match j.getObjVal? k with
    | .ok v => match v.getArr? with | .ok a => pure a | .error e => throw (IO.userError e)
    | .error e => throw (IO.userError e)
  let getNat (k : String) : IO Nat := match j.getObjVal? k with
    | .ok v => match v.getNat? with | .ok n => pure n | .error e => throw (IO.userError e)
    | .error e => throw (IO.userError e)
  let getNum (k : String) : IO Float := match j.getObjVal? k with
    | .ok v => match v.getNum? with | .ok n => pure n.toFloat | .error e => throw (IO.userError e)
    | .error e => throw (IO.userError e)
  let n ← getNat "n"
  let bos ← getNat "bos"
  let lossPy ← getNum "loss"
  let tokens : Array Nat ← (← getArr "tokens").mapM fun t => match t.getNat? with
    | .ok n => pure n | .error e => throw (IO.userError e)
  let logitsPy : Array (Array Float) ← (← getArr "logits").mapM fun row =>
    match row.getArr? with
    | .ok a => a.mapM fun c => match c.getNum? with | .ok x => pure x.toFloat | .error e => throw (IO.userError e)
    | .error e => throw (IO.userError e)
  let input : Array Nat := (Array.range cfg.blockSize).map fun i => if i < n then tokens[i]! else bos
  let logitsT := forwardLogits p cfg input
  let cols := logitsT.cols
  let mut maxDiff : Float := 0.0
  for i in [0:n] do
    let py := logitsPy[i]!
    for jj in [0:py.size] do
      let d := (py[jj]! - logitsT.data[i * cols + jj]!).abs
      if d > maxDiff then maxDiff := d
  let target : Array Nat := (Array.range cfg.blockSize).map fun i => if i < n then tokens[i + 1]! else bos
  let mask : Array Float := (Array.range cfg.blockSize).map fun i => if i < n then 1.0 else 0.0
  let lossT := forward p cfg input target mask
  let lossLean := lossT.data[0]!
  IO.println s!"loss_py={lossPy}  loss_lean={lossLean}  |Δloss|={(lossLean - lossPy).abs}  max|Δlogits|={maxDiff}"
