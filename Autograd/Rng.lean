import Autograd.Matrix

namespace Autograd

-- pure LCG state. MMIX (Knuth) constants.
structure RngState where s : UInt64 deriving Inhabited

def rngNext (st : RngState) : Float × RngState :=
  let s' : UInt64 := 6364136223846793005 * st.s + 1442695040888963407
  let u : Float := (s' >>> 11).toFloat / 9007199254740992.0  -- top 53 bits → [0,1)
  (u, { s := s' })

-- Box-Muller: two uniforms → one N(mean, stddev²) sample
def rngGauss (mean stddev : Float) (st : RngState) : Float × RngState :=
  let (u1, st1) := rngNext st
  let (u2, st2) := rngNext st1
  let u1' := if u1 < 1e-300 then 1e-300 else u1
  let z := Float.sqrt (-2.0 * Float.log u1') * Float.cos (2.0 * 3.141592653589793 * u2)
  (mean + stddev * z, st2)

def rngGaussMat (r c : Nat) (σ : Float) (st : RngState) : Matrix × RngState :=
  let (rows, st') : Array (Array Float) × RngState := Id.run do
    let mut acc : Array (Array Float) := Array.mkEmpty r
    let mut s := st
    for _ in [0:r] do
      let mut row : Array Float := Array.mkEmpty c
      for _ in [0:c] do
        let (x, s') := rngGauss 0.0 σ s
        row := row.push x; s := s'
      acc := acc.push row
    return (acc, s)
  (rows, st')

end Autograd
