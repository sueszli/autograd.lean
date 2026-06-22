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

-- `#eval`-as-test: a throwing IO action fails `lake build`, same as a false `#guard`
#eval do
  let j ← IO.ofExcept (Json.parse "[[0, 2], [1, 3]]")
  unless (← parseNatRows j) == #[#[0, 2], #[1, 3]] do throw (IO.userError "parseNatRows")
#eval do
  let j ← IO.ofExcept (Json.parse "[[0.5, 1.5]]")
  unless arrApproxEq (← parseFloatRows j)[0]! #[0.5, 1.5] do throw (IO.userError "parseFloatRows")

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

-- Python stores linear weights `[out × in]`, `matmulFwd` wants `[in × out]`. Transpose.
private def asRowsT (j : Json) : IO (Array Float × Nat × Nat) := do
  let (flat, r, c) ← asRows j
  return (transposeFlat flat r c, c, r)

private def leafFrom (triple : Array Float × Nat × Nat) (id : Nat) : Tensor :=
  Tensor.leaf triple.1 triple.2.1 triple.2.2 id true

#eval do
  let j ← IO.ofExcept (Json.parse "[[1, 2, 3], [4, 5, 6]]")
  let (flat, r, c) ← asRows j
  unless r == 2 && c == 3 && arrApproxEq flat #[1, 2, 3, 4, 5, 6] do throw (IO.userError "asRows")
  -- asRowsT flips Python's [out × in] to matmul's [in × out], swapping the dims
  let (flatT, rT, cT) ← asRowsT j
  unless rT == 3 && cT == 2 && arrApproxEq flatT #[1, 4, 2, 5, 3, 6] do throw (IO.userError "asRowsT")
  let t := leafFrom (flat, r, c) 7
  unless t.id == 7 && t.rows == 2 && t.cols == 3 do throw (IO.userError "leafFrom")

/-!
===--------------------------------------------------------------------------===
Weight loader
===--------------------------------------------------------------------------===
-/

private def paramsFromJson (j : Json) : IO Params := do
  return { wte := leafFrom (← asRows (← getObj j "wte")) ParamIds.wte, wpe := leafFrom (← asRows (← getObj j "wpe")) ParamIds.wpe, lmHead := leafFrom (← asRowsT (← getObj j "lm_head")) ParamIds.lmHead, blocks := #[{ attnWq := leafFrom (← asRowsT (← getObj j "layer0.attn_wq")) (ParamIds.attnWq 0), attnWk := leafFrom (← asRowsT (← getObj j "layer0.attn_wk")) (ParamIds.attnWk 0), attnWv := leafFrom (← asRowsT (← getObj j "layer0.attn_wv")) (ParamIds.attnWv 0), attnWo := leafFrom (← asRowsT (← getObj j "layer0.attn_wo")) (ParamIds.attnWo 0), mlpFc1 := leafFrom (← asRowsT (← getObj j "layer0.mlp_fc1")) (ParamIds.mlpFc1 0), mlpFc2 := leafFrom (← asRowsT (← getObj j "layer0.mlp_fc2")) (ParamIds.mlpFc2 0) }] }

-- minimal weight blob (1x1 matrices) to check the loader assigns the right `ParamIds` to each key
private def miniWeights : String := "{\"wte\":[[1]],\"wpe\":[[2]],\"lm_head\":[[3]],\"layer0.attn_wq\":[[4]],\"layer0.attn_wk\":[[5]],\"layer0.attn_wv\":[[6]],\"layer0.attn_wo\":[[7]],\"layer0.mlp_fc1\":[[8]],\"layer0.mlp_fc2\":[[9]]}"

#eval do
  let p ← paramsFromJson (← IO.ofExcept (Json.parse miniWeights))
  unless p.wte.id == ParamIds.wte && p.lmHead.id == ParamIds.lmHead && p.blocks[0]!.attnWq.id == ParamIds.attnWq 0 && p.blocks[0]!.mlpFc2.id == ParamIds.mlpFc2 0 do throw (IO.userError "paramsFromJson ids")
  unless arrApproxEq p.wte.data #[1] && arrApproxEq p.lmHead.data #[3] && arrApproxEq p.blocks[0]!.mlpFc2.data #[9] do throw (IO.userError "paramsFromJson data")

/-!
===--------------------------------------------------------------------------===
Tensor diff
===--------------------------------------------------------------------------===
-/

private def maxDiff (a b : Tensor) : Float :=
  (Array.range a.data.size).foldl (init := 0.0) fun mx i =>
    let d := (a.data[i]! - b.data[i]!).abs
    if d > mx then d else mx

-- max elementwise gap: |2-4|=2, |5-1|=4 -> 4
#guard approxEq (maxDiff (Tensor.leaf #[1, 2, 5] 1 3 0 true) (Tensor.leaf #[1, 4, 1] 1 3 1 true)) 4.0

/-!
===--------------------------------------------------------------------------===
Parity entrypoint
===--------------------------------------------------------------------------===
-/

-- tqdm-style in-place bar: `\r` rewrites the same line, flush forces it to show.
private def progressBar (cur : Nat) (total : Nat) (elapsedMs : Nat) : String :=
  let width := 30
  let filled := (cur * width) / total
  let bar := String.ofList (List.replicate filled '█') ++ String.ofList (List.replicate (width - filled) ' ')
  let pct := (cur * 100) / total
  let rate := if elapsedMs == 0 then 0 else (cur * 1000) / elapsedMs
  let etaMs := if cur == 0 then 0 else (elapsedMs * (total - cur)) / cur
  let fmt := fun (ms : Nat) => let s := ms / 1000; let r := s % 60; s!"{s / 60}:" ++ (if r < 10 then s!"0{r}" else s!"{r}")
  s!"{pct}%|{bar}| {cur}/{total} [{fmt elapsedMs}<{fmt etaMs}, {rate}it/s]"

-- the percentage prefix tracks `cur/total` (floor division)
#guard "0%|".isPrefixOf (progressBar 0 10 0)
#guard "50%|".isPrefixOf (progressBar 5 10 0)
#guard "100%|".isPrefixOf (progressBar 10 10 1000)

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
  let startMs ← IO.monoMsNow
  let stdout ← IO.getStdout
  for step in [0:cfg.numSteps] do
    let lossT := forward p cfg inputs[step]! targets[step]! masks[step]!
    let (p', s') := adamWStep cfg (step + 1) p s lossT.backward
    p := p'; s := s'
    let elapsed := (← IO.monoMsNow) - startMs
    IO.print s!"\r{progressBar (step + 1) cfg.numSteps elapsed}  "
    stdout.flush
  IO.println ""

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
  for (label, a, b) in pairs do
    let mx := maxDiff a b
    if mx > atol then failures := failures.push (label, mx)
  let totalMs := (← IO.monoMsNow) - startMs
  let ms := totalMs % 1000
  let msStr := if ms < 10 then s!"00{ms}" else if ms < 100 then s!"0{ms}" else s!"{ms}"
  if failures.isEmpty then
    IO.println s!"passed parity check (atol={atolStr})"
    IO.println s!"total time: {totalMs / 1000}.{msStr}s"
  else
    for (label, mx) in failures do
      IO.eprintln s!"  {label}: max |Δ| = {mx}"
    throw (IO.userError s!"FAIL: {failures.size}/{pairs.size} tensors exceed atol={atolStr}")
