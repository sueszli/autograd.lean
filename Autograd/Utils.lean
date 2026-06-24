namespace Autograd

/-!
===--------------------------------------------------------------------------===
Numerical comparison

Helpers for the `#guard` test cases scattered through the library. `#guard e`
errors at build time when `e : Bool` is `false`, so these double as a test
suite that runs on every `lake build`.
===--------------------------------------------------------------------------===
-/

def approxEq (a : Float) (b : Float) (tol : Float := 1e-9) : Bool := (a - b).abs ≤ tol

def arrApproxEq (a : Array Float) (b : Array Float) (tol : Float := 1e-9) : Bool :=
  a.size == b.size && (Array.range a.size).all (fun i => approxEq a[i]! b[i]! tol)

def maxAbsDiff (a : Array Float) (b : Array Float) : Float :=
  (Array.range a.size).foldl (init := 0.0) (fun mx i => max mx (a[i]! - b[i]!).abs)

#guard approxEq 1.0 1.0
#guard approxEq 1.0 (1.0 + 1e-12)
#guard !approxEq 1.0 1.1
#guard arrApproxEq #[1.0, 2.0, 3.0] #[1.0, 2.0, 3.0]
#guard arrApproxEq #[] #[]
#guard !arrApproxEq #[1.0, 2.0] #[1.0, 2.0, 3.0]
#guard approxEq (maxAbsDiff #[1.0, 5.0] #[1.0, 2.0]) 3.0

/-!
===--------------------------------------------------------------------------===
RNG
===--------------------------------------------------------------------------===
-/

structure RngState where s : UInt64 deriving Inhabited

def rngNext (st : RngState) : Float × RngState :=
  let s' : UInt64 := 6364136223846793005 * st.s + 1442695040888963407
  let u : Float := (s' >>> 11).toFloat / 9007199254740992.0
  (u, { s := s' })

def rngGauss (mean stddev : Float) (st : RngState) : Float × RngState :=
  let (u1, st1) := rngNext st
  let (u2, st2) := rngNext st1
  let u1' := if u1 < 1e-300 then 1e-300 else u1
  let z := Float.sqrt (-2.0 * Float.log u1') * Float.cos (2.0 * 3.141592653589793 * u2)
  (mean + stddev * z, st2)

def rngGaussFlat (r c : Nat) (σ : Float) (st : RngState) : Array Float × RngState := Id.run do
  let mut acc : Array Float := Array.mkEmpty (r * c)
  let mut s := st
  for _ in [0:r * c] do
    let (x, s') := rngGauss 0.0 σ s
    acc := acc.push x; s := s'
  return (acc, s)

#guard approxEq (rngNext { s := 42 }).1 (rngNext { s := 42 }).1
#guard let u := (rngNext { s := 42 }).1; 0.0 ≤ u && u < 1.0
#guard !approxEq (rngNext { s := 42 }).1 (rngNext (rngNext { s := 42 }).2).1
#guard approxEq (rngGauss 0.0 0.08 { s := 7 }).1 (0.08 * (rngGauss 0.0 1.0 { s := 7 }).1)
#guard (rngGaussFlat 3 4 0.08 { s := 1 }).1.size == 12

end Autograd
