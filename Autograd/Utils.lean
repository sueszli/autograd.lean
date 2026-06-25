namespace Autograd

/-!
===--------------------------------------------------------------------------===
Numerical comparison
===--------------------------------------------------------------------------===
-/

def approxEq (a : Float) (b : Float) (tol : Float := 1e-9) : Bool := (a - b).abs ≤ tol

def arrApproxEq (a : Array Float) (b : Array Float) (tol : Float := 1e-9) : Bool := a.isEqv b (approxEq · · tol)

-- tests
#guard approxEq 1.0 1.0
#guard approxEq 1.0 (1.0 + 1e-12)
#guard !approxEq 1.0 1.1
#guard arrApproxEq #[1.0, 2.0, 3.0] #[1.0, 2.0, 3.0]
#guard arrApproxEq #[] #[]
#guard !arrApproxEq #[1.0, 2.0] #[1.0, 2.0, 3.0]

/-!
===--------------------------------------------------------------------------===
RNG
===--------------------------------------------------------------------------===
-/

private def rngNext (s : UInt64) : Float × UInt64 :=
  let s' : UInt64 := 6364136223846793005 * s + 1442695040888963407
  let u : Float := (s' >>> 11).toFloat / 9007199254740992.0
  (u, s')

private def rngGauss (mean stddev : Float) (s : UInt64) : Float × UInt64 :=
  let (u1, s1) := rngNext s
  let (u2, s2) := rngNext s1
  let u1' := if u1 < 1e-300 then 1e-300 else u1
  let z := Float.sqrt (-2.0 * Float.log u1') * Float.cos (2.0 * 3.141592653589793 * u2)
  (mean + stddev * z, s2)

def rngGaussFlat (r c : Nat) (σ : Float) (st : UInt64) : Array Float × UInt64 := Id.run do
  let mut acc : Array Float := Array.mkEmpty (r * c)
  let mut s := st
  for _ in [0:r * c] do
    let (x, s') := rngGauss 0.0 σ s
    acc := acc.push x; s := s'
  return (acc, s)

-- tests
#guard approxEq (rngNext 42).1 (rngNext 42).1
#guard let u := (rngNext 42).1; 0.0 ≤ u && u < 1.0
#guard !approxEq (rngNext 42).1 (rngNext (rngNext 42).2).1
#guard approxEq (rngGauss 0.0 0.08 7).1 (0.08 * (rngGauss 0.0 1.0 7).1)
#guard (rngGaussFlat 3 4 0.08 1).1.size == 12

/-!
===--------------------------------------------------------------------------===
Progress bar
===--------------------------------------------------------------------------===
-/

-- tqdm-style in-place bar: `\r` rewrites the same line, flush forces it to show.
def progressBar (cur : Nat) (total : Nat) (elapsedMs : Nat) : String :=
  let width := 30
  let filled := (cur * width) / total
  let bar := String.ofList (List.replicate filled '█') ++ String.ofList (List.replicate (width - filled) ' ')
  let pct := (cur * 100) / total
  let rate := if elapsedMs == 0 then 0 else (cur * 1000) / elapsedMs
  let etaMs := if cur == 0 then 0 else (elapsedMs * (total - cur)) / cur
  let fmt := fun (ms : Nat) => let s := ms / 1000; let r := s % 60; s!"{s / 60}:" ++ (if r < 10 then s!"0{r}" else s!"{r}")
  s!"{pct}%|{bar}| {cur}/{total} [{fmt elapsedMs}<{fmt etaMs}, {rate}it/s]"

-- tests
#guard "0%|".isPrefixOf (progressBar 0 10 0)
#guard "50%|".isPrefixOf (progressBar 5 10 0)
#guard "100%|".isPrefixOf (progressBar 10 10 1000)

end Autograd
