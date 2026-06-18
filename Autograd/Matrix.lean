namespace Autograd

-- raw 2D numeric data. Tensor (in Tensor.lean) wraps this with autograd metadata.
abbrev Matrix := Array (Array Float)

-- model hyperparams; lives here because every layer + optimizer + Tensor.GradFn
-- references it.
structure Config where
  nLayer : Nat
  nEmbed : Nat
  blockSize : Nat
  nHead : Nat
  vocabSize : Nat
  numSteps : Nat
  epsilon : Float := 1e-5
  lr0 : Float := 0.01
  beta1 : Float := 0.85
  beta2 : Float := 0.99
  weightDecay : Float := 0.0
  maskValue : Float := -1.0e9
  deriving Inhabited

namespace Matrix
def rows (m : Matrix) : Nat := m.size
def cols (m : Matrix) : Nat := if m.size = 0 then 0 else m[0]!.size
def zeros (r c : Nat) : Matrix := Array.replicate r (Array.replicate c 0.0)
def ofFn (r c : Nat) (f : Nat → Nat → Float) : Matrix :=
  (Array.range r).map fun i => (Array.range c).map fun j => f i j
def transpose (m : Matrix) : Matrix :=
  let r := m.size
  let c := if r = 0 then 0 else m[0]!.size
  ofFn c r fun i j => m[j]![i]!
end Matrix

-- elementwise tolerance check (Bool-coerced to Prop in examples).
def allcloseM (a b : Matrix) (atol : Float := 1e-3) : Bool :=
  a.size = b.size &&
  (Array.range a.size).all fun i =>
    a[i]!.size = b[i]!.size &&
    (Array.range a[i]!.size).all fun j => (a[i]![j]! - b[i]![j]!).abs < atol

def allcloseV (a b : Array Float) (atol : Float := 1e-3) : Bool :=
  a.size = b.size &&
  (Array.range a.size).all fun i => (a[i]! - b[i]!).abs < atol

-- shared FD helpers: matrix- and vector-valued gradient of a scalar loss
def fdGradMat (loss : Matrix → Float) (x : Matrix) (h : Float := 1e-4) : Matrix :=
  let r := x.size
  let c := if r = 0 then 0 else x[0]!.size
  Matrix.ofFn r c fun i k =>
    let row := x[i]!
    let xP := x.set! i (row.set! k (row[k]! + h))
    let xM := x.set! i (row.set! k (row[k]! - h))
    (loss xP - loss xM) / (2.0 * h)

def fdGradVec (loss : Array Float → Float) (x : Array Float) (h : Float := 1e-4) : Array Float :=
  (Array.range x.size).map fun i =>
    let xP := x.set! i (x[i]! + h)
    let xM := x.set! i (x[i]! - h)
    (loss xP - loss xM) / (2.0 * h)

-- elementwise sum and ⟨dout, y⟩ — used to build scalar losses in tests
def matSum (m : Matrix) : Float :=
  m.foldl (init := 0.0) fun a r => a + r.foldl (init := 0.0) (· + ·)

def wLoss (dout y : Matrix) : Float :=
  (Array.range y.size).foldl (init := 0.0) fun acc i =>
    (Array.range y[i]!.size).foldl (init := acc) fun acc j =>
      acc + dout[i]![j]! * y[i]![j]!

example : (Matrix.zeros 2 3).size = 2 := by native_decide
example : Matrix.cols #[#[1.0, 2.0, 3.0]] = 3 := by native_decide
example : allcloseM (Matrix.transpose (Matrix.transpose #[#[1.0, 2.0, 3.0]])) #[#[1.0, 2.0, 3.0]] := by native_decide

end Autograd
