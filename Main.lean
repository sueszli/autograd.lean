import Autograd
import Lean.Data.Json

open Autograd
open Lean (Json)

private def train : IO Unit := do
  let cfg0 : Config := {
    nLayer := 1, nEmbed := 16, blockSize := 16, nHead := 4,
    vocabSize := 0, numSteps := 1000
  }
  let text ← IO.FS.readFile "data/input.txt"
  let docsRaw : Array String := (text.splitOn "\n").toArray.filter (·.length > 0)
  let (docs, rngShuf) := shuffleDocs docsRaw { s := 42 }
  let vocab := buildVocab docs
  let cfg : Config := { cfg0 with vocabSize := vocab.size }
  let (p0, _) := initParams cfg rngShuf
  IO.println s!"num docs: {docs.size}  vocab: {vocab.size}  training {cfg.numSteps} steps..."
  let mut p := p0
  let mut s := OptState.zeros p0
  let mut ema : Float := 0.0
  for step in [0:cfg.numSteps] do
    let doc := docs[step % docs.size]!
    let (input, target, mask) := docPair (encodeDoc vocab doc) cfg.blockSize (vocab.size - 1)
    let lossT := forward p cfg input target mask
    let loss := lossT.data[0]![0]!
    let (p', s') := adamWStep cfg (step + 1) p s lossT.backward
    p := p'; s := s'
    ema := if step = 0 then loss else 0.9 * ema + 0.1 * loss
    if step = 0 || (step + 1) % 50 = 0 then
      IO.println s!"step {step + 1}/{cfg.numSteps}  ema_loss={ema}"

private def parity : IO Unit := do
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
  let mut maxDiff : Float := 0.0
  for i in [0:n] do
    let py := logitsPy[i]!
    let leanRow := logitsT.data[i]!
    for jj in [0:py.size] do
      let d := (py[jj]! - leanRow[jj]!).abs
      if d > maxDiff then maxDiff := d
  let target : Array Nat := (Array.range cfg.blockSize).map fun i => if i < n then tokens[i + 1]! else bos
  let mask : Array Float := (Array.range cfg.blockSize).map fun i => if i < n then 1.0 else 0.0
  let lossT := forward p cfg input target mask
  let lossLean := lossT.data[0]![0]!
  IO.println s!"loss_py={lossPy}  loss_lean={lossLean}  |Δloss|={(lossLean - lossPy).abs}  max|Δlogits|={maxDiff}"

def main (args : List String) : IO Unit :=
  match args with
  | ["parity"] => parity
  | _          => train
