import Autograd.Tensor
import Lean.Data.Json

namespace MicroGPT
open Autograd
open Lean (Json fromJson?)

/-!
===--------------------------------------------------------------------------===
JSON
===--------------------------------------------------------------------------===
-/

def getObj (j : Json) (k : String) : IO Json :=
  match j.getObjVal? k with
  | .ok v => pure v
  | .error e => throw (IO.userError s!"missing key '{k}': {e}")

def parseNatRows (j : Json) : IO (Array (Array Nat)) := IO.ofExcept (fromJson? j)

def parseFloatRows (j : Json) : IO (Array (Array Float)) := IO.ofExcept (fromJson? j)

def asRows (j : Json) : IO (Array Float × Nat × Nat) := do
  let rows : Array (Array Float) ← IO.ofExcept (fromJson? j)
  let c := if 0 < rows.size then rows[0]!.size else 0
  return (rows.flatten, rows.size, c)

-- Python stores linear weights `[out × in]`, `matmulFwd` wants `[in × out]`. Transpose.
def asRowsT (j : Json) : IO (Array Float × Nat × Nat) := do
  let (flat, r, c) ← asRows j
  return (transposeFlat flat r c, c, r)

def leafFrom (triple : Array Float × Nat × Nat) (id : Nat) : Tensor :=
  Tensor.leaf triple.1 triple.2.1 triple.2.2 id true

-- tests
-- `#eval`-as-test: a throwing IO action fails `lake build`, same as a false `#guard`
#eval do
  let j ← IO.ofExcept (Json.parse "[[0, 2], [1, 3]]")
  unless (← parseNatRows j) == #[#[0, 2], #[1, 3]] do throw (IO.userError "parseNatRows")
#eval do
  let j ← IO.ofExcept (Json.parse "[[0.5, 1.5]]")
  unless arrApproxEq (← parseFloatRows j)[0]! #[0.5, 1.5] do throw (IO.userError "parseFloatRows")
#eval do
  let j ← IO.ofExcept (Json.parse "[[1, 2, 3], [4, 5, 6]]")
  let (flat, r, c) ← asRows j
  unless r == 2 && c == 3 && arrApproxEq flat #[1, 2, 3, 4, 5, 6] do throw (IO.userError "asRows")
  let (flatT, rT, cT) ← asRowsT j
  unless rT == 3 && cT == 2 && arrApproxEq flatT #[1, 4, 2, 5, 3, 6] do throw (IO.userError "asRowsT")
  let t := leafFrom (flat, r, c) 7
  unless t.id == 7 && t.rows == 2 && t.cols == 3 do throw (IO.userError "leafFrom")

end MicroGPT
