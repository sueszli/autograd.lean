namespace Autograd.Scalar

/--
algebraic data type
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
overload infix operators
e.g. `x + y` equiv to `.add x y`
-/
instance : Add Expr := ⟨.add⟩
instance : Mul Expr := ⟨.mul⟩
instance : Neg Expr := ⟨fun a => .mul (.const (-1.0)) a⟩

namespace Expr

def eval : Expr → Array Float → Float
  | .var i, arr => arr[i]!
  | .const c, _ => c
  | .add a b, arr => a.eval arr + b.eval arr
  | .mul a b, arr => a.eval arr * b.eval arr
  | .relu a, arr => if a.eval arr > 0.0 then a.eval arr else 0.0
  | .tanh a, arr => Float.tanh (a.eval arr)

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

/--
gradient vector of the same length as arr whose i-th entry is ∂(eval e arr)/∂arr[i]
-/
def grad (e : Expr) (arr : Array Float) : Array Float :=
  (Array.range arr.size).map fun i => e.backward arr i

/-!
Proofs
-/

@[simp] theorem eval_var (i : Nat) (arr : Array Float) :
    (var i).eval arr = arr[i]! := rfl

@[simp] theorem eval_const (c : Float) (arr : Array Float) :
    (const c).eval arr = c := rfl

@[simp] theorem eval_add (a b : Expr) (arr : Array Float) :
    (add a b).eval arr = a.eval arr + b.eval arr := rfl

@[simp] theorem eval_mul (a b : Expr) (arr : Array Float) :
    (mul a b).eval arr = a.eval arr * b.eval arr := rfl

@[simp] theorem eval_tanh (a : Expr) (arr : Array Float) :
    (tanh a).eval arr = Float.tanh (a.eval arr) := rfl

@[simp] theorem backward_const (c : Float) (arr : Array Float) (i : Nat) (up : Float) :
    (const c).backward arr i up = 0.0 := rfl

@[simp] theorem backward_var_self (i : Nat) (arr : Array Float) (up : Float) :
    (var i).backward arr i up = up := by
  show (if i = i then up else 0.0) = up
  simp

theorem backward_var_ne {i j : Nat} (h : j ≠ i) (arr : Array Float) (up : Float) :
    (var j).backward arr i up = 0.0 := by
  show (if j = i then up else 0.0) = 0.0
  simp [h]

@[simp] theorem backward_add (a b : Expr) (arr : Array Float) (i : Nat) (up : Float) :
    (add a b).backward arr i up = a.backward arr i up + b.backward arr i up := rfl

@[simp] theorem backward_mul (a b : Expr) (arr : Array Float) (i : Nat) (up : Float) :
    (mul a b).backward arr i up =
      a.backward arr i (up * b.eval arr) + b.backward arr i (up * a.eval arr) := rfl

@[simp] theorem backward_relu (a : Expr) (arr : Array Float) (i : Nat) (up : Float) :
    (relu a).backward arr i up =
      a.backward arr i (up * (if a.eval arr > 0.0 then 1.0 else 0.0)) := rfl

theorem grad_size (e : Expr) (arr : Array Float) :
    (e.grad arr).size = arr.size := by
  simp [grad]

/-!
Tests
-/

private def fdGrad (e : Expr) (arr : Array Float) (i : Nat) (h : Float := 1e-4) : Float :=
  let plus  := arr.set! i (arr[i]! + h)
  let minus := arr.set! i (arr[i]! - h)
  (e.eval plus - e.eval minus) / (2.0 * h)

private def closeEnough (a b : Float) : Bool :=
  (a - b).abs < 1e-3

private def gradMatchesFd (e : Expr) (arr : Array Float) : Bool :=
  (Array.range arr.size).all fun i =>
    closeEnough (e.backward arr i) (fdGrad e arr i)

example : (const 3.14).backward #[1.0, 2.0] 0 = 0.0 := rfl
example : (var 0).backward #[5.0, 6.0] 0 = 1.0 := backward_var_self 0 _ _
example : (var 1).backward #[5.0, 6.0] 0 = 0.0 := backward_var_ne (by decide) _ _
example : ((var 0).grad #[7.0, 8.0]).size = 2 := grad_size _ _

private def x : Expr := .var 0
private def y : Expr := .var 1
private def z : Expr := .var 2

example : gradMatchesFd (x + y)          #[1.0, 2.0]         = true := by native_decide
example : gradMatchesFd (x * y)          #[3.0, 4.0]         = true := by native_decide
example : gradMatchesFd (x*x + y*y)      #[1.5, -2.5]        = true := by native_decide
example : gradMatchesFd ((x + y) * (-y)) #[1.0, 0.7]         = true := by native_decide
example : gradMatchesFd (.relu x)        #[2.0]              = true := by native_decide
example : gradMatchesFd (.relu (-x))     #[2.0]              = true := by native_decide
example : gradMatchesFd (.tanh x)        #[0.5]              = true := by native_decide
example : gradMatchesFd (.tanh (x*y + z)) #[0.3, 0.4, 0.1]   = true := by native_decide
example : gradMatchesFd (.relu (x*y) + .tanh (x + y)) #[0.6, -0.2] = true := by native_decide

end Expr

end Autograd.Scalar
