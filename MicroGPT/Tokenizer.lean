import MicroGPT.Rng

namespace MicroGPT

-- char-level vocab. token 0..|chars|−1 are real chars; BOS = chars.size.
structure Vocab where
  size : Nat
  chars : Array Char
  toId : Array Nat   -- 128-entry ASCII lookup
  deriving Inhabited

-- collect unique chars across docs, sorted ascending (matches Python's sorted(set(...))).
def buildVocab (docs : Array String) : Vocab :=
  let seen0 : Array Bool := Array.replicate 128 false
  let seen : Array Bool := docs.foldl (init := seen0) fun acc s =>
    s.toList.foldl (init := acc) fun acc c =>
      let i := c.toNat
      if i < 128 then acc.set! i true else acc
  let chars : Array Char := (Array.range 128).foldl (init := #[]) fun acc i =>
    if seen[i]! then acc.push (Char.ofNat i) else acc
  let toId : Array Nat := Id.run do
    let mut acc : Array Nat := Array.replicate 128 0
    for i in [0:chars.size] do acc := acc.set! chars[i]!.toNat i
    return acc
  { size := chars.size + 1, chars := chars, toId := toId }

-- encode one doc as [BOS, ch_0, …, ch_{n−1}, BOS] (matches original.py).
def encodeDoc (v : Vocab) (doc : String) : Array Nat :=
  let bos := v.chars.size
  let body : Array Nat := doc.toList.foldl (init := #[]) fun acc c =>
    let i := c.toNat
    if i < 128 then acc.push v.toId[i]! else acc
  (#[bos] ++ body).push bos

-- input[0..n−1] = tokens[0..n−1]; target[0..n−1] = tokens[1..n]; mask 1 on real positions.
-- pad to blockSize with BOS so the forward kernel sees valid indices everywhere.
def docPair (tokens : Array Nat) (blockSize bos : Nat) : Array Nat × Array Nat × Array Float :=
  let n : Nat := Nat.min blockSize (tokens.size - 1)
  let input : Array Nat := (Array.range blockSize).map fun i =>
    if i < n then tokens[i]! else bos
  let target : Array Nat := (Array.range blockSize).map fun i =>
    if i < n then tokens[i + 1]! else bos
  let mask : Array Float := (Array.range blockSize).map fun i =>
    if i < n then 1.0 else 0.0
  (input, target, mask)

-- Fisher-Yates shuffle in-place via a mut Array.
def shuffleDocs (xs : Array String) (st : RngState) : Array String × RngState := Id.run do
  let mut a := xs
  let mut s := st
  let mut i := a.size
  while i > 1 do
    i := i - 1
    let (u, s') := rngNext s; s := s'
    let j : Nat := (u * (i + 1).toFloat).toUInt32.toNat % (i + 1)
    let ai := a[i]!; let aj := a[j]!
    a := (a.set! i aj).set! j ai
  return (a, s)

end MicroGPT
