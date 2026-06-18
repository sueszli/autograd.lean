namespace Autograd.Scalar

/--
algebraic data type for scalar expressions.
e.g. `arr[0] * arr[1] + 1` equiv to `.add (.mul (.var 0) (.var 1)) (.const 1.0)`
-/
inductive Expr where
  | var : Nat → Expr
  | const : Float → Expr
  | add : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | relu : Expr → Expr
  | tanh : Expr → Expr
  deriving Repr, Inhabited

/--
syntactic sugar for infix operators.
e.g. `x + y` equiv to `.add x y`, `-x` equiv to `.mul (.const (-1.0)) x`
-/
instance : Add Expr := ⟨.add⟩
instance : Mul Expr := ⟨.mul⟩
instance : Neg Expr := ⟨.mul (.const (-1.0))⟩

namespace Expr

def eval (e : Expr) (arr : Array Float) : Float :=
  match e with
  | .var i => arr[i]!
  | .const c => c
  | .add a b => a.eval arr + b.eval arr
  | .mul a b => a.eval arr * b.eval arr
  | .relu a => if a.eval arr > 0.0 then a.eval arr else 0.0
  | .tanh a => Float.tanh (a.eval arr)

/--
one step of the backward pass through a subtree.
- `arr`: the input values (mul needs them since (a*b)' uses both operands)
- `i`: which input slot we want the gradient for
- `up`: gradient flowing in from the parent (root call starts with up = 1.0)
-/
def backward (e : Expr) (arr : Array Float) (i : Nat) (up : Float := 1.0) : Float :=
  match e with
  | .var j => if j = i then up else 0.0
  | .const _ => 0.0
  | .add a b => a.backward arr i up + b.backward arr i up
  | .mul a b => a.backward arr i (up * b.eval arr) + b.backward arr i (up * a.eval arr)
  | .relu a => a.backward arr i (up * (if a.eval arr > 0.0 then 1.0 else 0.0))
  | .tanh a => a.backward arr i (up * (1.0 - Float.tanh (a.eval arr) * Float.tanh (a.eval arr)))

@[simp] theorem backward_var_match (i : Nat) (arr : Array Float) (up : Float) :
    (var i).backward arr i up = up := by
  show (if i = i then up else 0.0) = up
  simp

@[simp] theorem backward_var_no_match {i j : Nat} (h : j ≠ i) (arr : Array Float) (up : Float) :
    (var j).backward arr i up = 0.0 := by
  show (if j = i then up else 0.0) = 0.0
  simp [h]

@[simp] theorem backward_const (c : Float) (arr : Array Float) (i : Nat) (up : Float) :
    (Expr.const c).backward arr i up = 0.0 := rfl

@[simp] theorem backward_add (a b : Expr) (arr : Array Float) (i : Nat) (up : Float) :
    (a + b).backward arr i up = a.backward arr i up + b.backward arr i up := rfl

@[simp] theorem backward_mul (a b : Expr) (arr : Array Float) (i : Nat) (up : Float) :
    (a * b).backward arr i up = a.backward arr i (up * b.eval arr) + b.backward arr i (up * a.eval arr) := rfl

/-- gradient vector of the same length as arr whose i-th entry is ∂(eval e arr)/∂arr[i] -/
def grad (e : Expr) (arr : Array Float) : Array Float :=
  (Array.range arr.size).map λi => e.backward arr i

theorem grad_size (e : Expr) (arr : Array Float) :
    (e.grad arr).size = arr.size := by
  simp [grad]

private def allclose (e : Expr) (arr : Array Float)
    (h : Float := 1e-4) (atol : Float := 1e-3) : Bool :=
  let g := e.grad arr
  (Array.range arr.size).all λi =>
    (g[i]! - (e.eval (arr.set! i (arr[i]! + h)) - e.eval (arr.set! i (arr[i]! - h))) / (2.0 * h)).abs < atol

example : allclose (.const 5.0)                                            #[1.0]           := by native_decide
example : allclose (.var 0 + .var 1)                                       #[1.0, 2.0]      := by native_decide
example : allclose (.var 0 + .const 5.0)                                   #[2.0]           := by native_decide
example : allclose (.var 0 * .var 1)                                       #[3.0, 4.0]      := by native_decide
example : allclose (.var 0 * .var 0 + .var 1 * .var 1)                     #[1.5, -2.5]     := by native_decide
example : allclose ((.var 0 + .var 1) * (- .var 1))                        #[1.0, 0.7]      := by native_decide
example : allclose (.var 0 * .var 1 * .var 2)                              #[1.0, 2.0, 0.5] := by native_decide
example : allclose ((.var 0 + .const 1) * (.var 1 + .const 2))             #[0.5, 1.5]      := by native_decide
example : allclose (.relu (.var 0))                                        #[2.0]           := by native_decide
example : allclose (.relu (- .var 0))                                      #[2.0]           := by native_decide
example : allclose (.tanh (.var 0))                                        #[0.5]           := by native_decide
example : allclose (.tanh (.tanh (.var 0)))                                #[0.3]           := by native_decide
example : allclose (.tanh (.var 0 * .var 1 + .var 2))                      #[0.3, 0.4, 0.1] := by native_decide
example : allclose (.relu (.tanh (.var 0)))                                #[0.4]           := by native_decide
example : allclose (.relu (.var 0 * .var 1) + .tanh (.var 0 + .var 1))     #[0.6, -0.2]     := by native_decide

end Expr

end Autograd.Scalar
