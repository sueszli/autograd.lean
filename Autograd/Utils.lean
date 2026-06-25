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
Progress bar
===--------------------------------------------------------------------------===
-/

def tqdm (cur : Nat) (total : Nat) (elapsedMs : Nat) : String :=
  let width := 30
  let filled := (cur * width) / total
  let bar := ("".pushn '█' filled).pushn ' ' (width - filled)
  let pct := (cur * 100) / total
  let rate := if elapsedMs == 0 then 0 else (cur * 1000) / elapsedMs
  let etaMs := if cur == 0 then 0 else (elapsedMs * (total - cur)) / cur
  let fmt := fun (ms : Nat) => let s := ms / 1000; let r := s % 60; s!"{s / 60}:" ++ (if r < 10 then s!"0{r}" else s!"{r}")
  s!"{pct}%|{bar}| {cur}/{total} [{fmt elapsedMs}<{fmt etaMs}, {rate}it/s]"

-- tests
#guard "0%|".isPrefixOf (tqdm 0 10 0)
#guard "50%|".isPrefixOf (tqdm 5 10 0)
#guard "100%|".isPrefixOf (tqdm 10 10 1000)

end Autograd
