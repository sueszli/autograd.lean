import Autograd.Tensor
import Lean.Data.Json

namespace Autograd
open Lean (Json)

/-!
===--------------------------------------------------------------------------===
JSON primitives
===--------------------------------------------------------------------------===
-/

def asArr (j : Json) : IO (Array Json) :=
  match j.getArr? with
  | .ok a => pure a
  | .error e => throw (IO.userError s!"expected array: {e}")

def asNat (j : Json) : IO Nat :=
  match j.getNat? with
  | .ok n => pure n
  | .error e => throw (IO.userError s!"expected nat: {e}")

def asFloat (j : Json) : IO Float :=
  match j.getNum? with
  | .ok n => pure n.toFloat
  | .error e => throw (IO.userError s!"expected number: {e}")

def getObj (j : Json) (k : String) : IO Json :=
  match j.getObjVal? k with
  | .ok v => pure v
  | .error e => throw (IO.userError s!"missing key '{k}': {e}")

def parseNatRows (j : Json) : IO (Array (Array Nat)) := do
  (← asArr j).mapM fun row => do (← asArr row).mapM asNat

def parseFloatRows (j : Json) : IO (Array (Array Float)) := do
  (← asArr j).mapM fun row => do (← asArr row).mapM asFloat

-- tests
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

def asRows (j : Json) : IO (Array Float × Nat × Nat) := do
  let rows ← asArr j
  let r := rows.size
  let c : Nat := ← if r = 0 then pure 0 else do let row0 ← asArr rows[0]!; pure row0.size
  let flat ← rows.foldlM (init := (#[] : Array Float)) fun acc row => do
    let cols ← asArr row
    return acc ++ (← cols.mapM asFloat)
  return (flat, r, c)

-- Python stores linear weights `[out × in]`, `matmulFwd` wants `[in × out]`. Transpose.
def asRowsT (j : Json) : IO (Array Float × Nat × Nat) := do
  let (flat, r, c) ← asRows j
  return (transposeFlat flat r c, c, r)

def leafFrom (triple : Array Float × Nat × Nat) (id : Nat) : Tensor :=
  Tensor.leaf triple.1 triple.2.1 triple.2.2 id true

-- tests
#eval do
  let j ← IO.ofExcept (Json.parse "[[1, 2, 3], [4, 5, 6]]")
  let (flat, r, c) ← asRows j
  unless r == 2 && c == 3 && arrApproxEq flat #[1, 2, 3, 4, 5, 6] do throw (IO.userError "asRows")
  let (flatT, rT, cT) ← asRowsT j
  unless rT == 3 && cT == 2 && arrApproxEq flatT #[1, 4, 2, 5, 3, 6] do throw (IO.userError "asRowsT")
  let t := leafFrom (flat, r, c) 7
  unless t.id == 7 && t.rows == 2 && t.cols == 3 do throw (IO.userError "leafFrom")

/-!
===--------------------------------------------------------------------------===
Tensor diff
===--------------------------------------------------------------------------===
-/

def maxDiff (a b : Tensor) : Float :=
  (Array.range a.data.size).foldl (init := 0.0) fun mx i =>
    let d := (a.data[i]! - b.data[i]!).abs
    if d > mx then d else mx

-- tests
#guard approxEq (maxDiff (Tensor.leaf #[1, 2, 5] 1 3 0 true) (Tensor.leaf #[1, 4, 1] 1 3 1 true)) 4.0

end Autograd
