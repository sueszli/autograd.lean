namespace Autograd

/-!
===--------------------------------------------------------------------------===
Numerical comparison
===--------------------------------------------------------------------------===
-/

def approxEq (a : Float) (b : Float) (tol : Float := 1e-9) : Bool := (a - b).abs ≤ tol

def arrApproxEq (a : Array Float) (b : Array Float) (tol : Float := 1e-9) : Bool := a.isEqv b (approxEq · · tol)

def arrMaxDiff (a : Array Float) (b : Array Float) : Float := (Array.zipWith (fun x y => (x - y).abs) a b).foldl max 0.0

-- tests
#guard approxEq 1.0 1.0
#guard approxEq 1.0 (1.0 + 1e-12)
#guard !approxEq 1.0 1.1
#guard arrApproxEq #[1.0, 2.0, 3.0] #[1.0, 2.0, 3.0]
#guard arrApproxEq #[] #[]
#guard !arrApproxEq #[1.0, 2.0] #[1.0, 2.0, 3.0]
#guard approxEq (arrMaxDiff #[1, 2, 5] #[1, 4, 1]) 4.0
#guard approxEq (arrMaxDiff #[] #[]) 0.0
#guard approxEq (arrMaxDiff #[3, 3] #[3, 3]) 0.0

/-!
===--------------------------------------------------------------------------===
Pseudo-random numbers
===--------------------------------------------------------------------------===
-/

def rngNext (s : UInt64) : UInt64 := s * 6364136223846793005 + 1442695040888963407

def rngFloat : StateM UInt64 Float := do let s := rngNext (← get); set s; pure ((s >>> 11).toNat.toFloat / (1 <<< 53 : Nat).toFloat)

-- tests
#guard rngNext 1 != 1
#guard let a : Float := rngFloat.run' 7; let b : Float := rngFloat.run' 7; a == b
#guard let x : Float := rngFloat.run' 7; 0.0 ≤ x && x < 1.0

/-!
===--------------------------------------------------------------------------===
Progress bar
===--------------------------------------------------------------------------===
-/

private def tqdmBar (cur : Nat) (total : Nat) (elapsedMs : Nat) : String :=
  let width := 30
  let filled := (cur * width) / total
  let bar := ("".pushn '█' filled).pushn ' ' (width - filled)
  let pct := (cur * 100) / total
  let rate := if elapsedMs == 0 then 0 else (cur * 1000) / elapsedMs
  let etaMs := if cur == 0 then 0 else (elapsedMs * (total - cur)) / cur
  let fmt := fun (ms : Nat) => let s := ms / 1000; let r := s % 60; s!"{s / 60}:" ++ (if r < 10 then s!"0{r}" else s!"{r}")
  s!"{pct}%|{bar}| {cur}/{total} [{fmt elapsedMs}<{fmt etaMs}, {rate}it/s]"

-- redraw
def tqdmTick (startMs : Nat) (cur : Nat) (total : Nat) : IO Unit := do
  let elapsedMs := (← IO.monoMsNow) - startMs
  IO.print s!"\r{tqdmBar cur total elapsedMs}  "
  (← IO.getStdout).flush

-- report wall-clock time since startMs
def tqdmDone (startMs : Nat) : IO Unit := do
  let ms := (← IO.monoMsNow) - startMs
  let frac := ms % 1000
  let pad := if frac < 10 then "00" else if frac < 100 then "0" else ""
  IO.println ""
  IO.println s!"total time: {ms / 1000}.{pad}{frac}s"

-- tests
#guard "0%|".isPrefixOf (tqdmBar 0 10 0)
#guard "50%|".isPrefixOf (tqdmBar 5 10 0)
#guard "100%|".isPrefixOf (tqdmBar 10 10 1000)

end Autograd
