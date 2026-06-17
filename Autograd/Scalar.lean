namespace Autograd.Scalar

-- algebraic data type
-- e.g. `x*y + 1` equivalent to `.add (.mul (.var 0) (.var 1)) (.const 1.0)`
inductive Expr where
  | var : Nat → Expr
  | const : Float → Expr
  | add : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | relu : Expr → Expr
  | tanh : Expr → Expr
  deriving Repr, Inhabited

-- allows us to write `e.eval args` instead of `Expr.eval e args`
namespace Expr

def eval : Expr → Array Float → Float
  | .var i,  inputs => inputs[i]!
  | .const c,  _      => c
  | .add a b,  inputs => a.eval inputs + b.eval inputs
  | .mul a b,  inputs => a.eval inputs * b.eval inputs
  | .relu a,   inputs => if a.eval inputs > 0.0 then a.eval inputs else 0.0
  | .tanh a,   inputs => (a.eval inputs).tanh

def bwd : Expr → Array Float → Nat → Float → Float
  | .var j,  _,      i, up => if j = i then up else 0.0
  | .const _,  _,      _, _  => 0.0
  | .add a b,  inputs, i, up => a.bwd inputs i up + b.bwd inputs i up
  | .mul a b,  inputs, i, up =>
      a.bwd inputs i (up * b.eval inputs) + b.bwd inputs i (up * a.eval inputs)
  | .relu a,   inputs, i, up =>
      a.bwd inputs i (up * (if a.eval inputs > 0.0 then 1.0 else 0.0))
  | .tanh a,   inputs, i, up =>
      let t := (a.eval inputs).tanh
      a.bwd inputs i (up * (1.0 - t * t))

def backward (e : Expr) (inputs : Array Float) (i : Nat) : Float :=
  e.bwd inputs i 1.0

def grad (e : Expr) (inputs : Array Float) : Array Float :=
  (Array.range inputs.size).map (e.backward inputs)

instance : Add Expr := ⟨.add⟩
instance : Mul Expr := ⟨.mul⟩
instance : Neg Expr := ⟨fun a => .mul (.const (-1.0)) a⟩

/-
Proofs
-/

@[simp] theorem eval_var (i : Nat) (inputs : Array Float) :
    (Expr.var i).eval inputs = inputs[i]! := rfl

@[simp] theorem eval_const (c : Float) (inputs : Array Float) :
    (Expr.const c).eval inputs = c := rfl

@[simp] theorem eval_add (a b : Expr) (inputs : Array Float) :
    (Expr.add a b).eval inputs = a.eval inputs + b.eval inputs := rfl

@[simp] theorem eval_mul (a b : Expr) (inputs : Array Float) :
    (Expr.mul a b).eval inputs = a.eval inputs * b.eval inputs := rfl

@[simp] theorem eval_tanh (a : Expr) (inputs : Array Float) :
    (Expr.tanh a).eval inputs = (a.eval inputs).tanh := rfl

@[simp] theorem bwd_const (c : Float) (inputs : Array Float) (i : Nat) (up : Float) :
    (Expr.const c).bwd inputs i up = 0.0 := rfl

@[simp] theorem bwd_var_self (i : Nat) (inputs : Array Float) (up : Float) :
    (Expr.var i).bwd inputs i up = up := by
  show (if i = i then up else 0.0) = up
  simp

theorem bwd_var_ne {i j : Nat} (h : j ≠ i) (inputs : Array Float) (up : Float) :
    (Expr.var j).bwd inputs i up = 0.0 := by
  show (if j = i then up else 0.0) = 0.0
  simp [h]

@[simp] theorem bwd_add (a b : Expr) (inputs : Array Float) (i : Nat) (up : Float) :
    (Expr.add a b).bwd inputs i up = a.bwd inputs i up + b.bwd inputs i up := rfl

@[simp] theorem bwd_mul (a b : Expr) (inputs : Array Float) (i : Nat) (up : Float) :
    (Expr.mul a b).bwd inputs i up =
      a.bwd inputs i (up * b.eval inputs) + b.bwd inputs i (up * a.eval inputs) := rfl

@[simp] theorem bwd_relu (a : Expr) (inputs : Array Float) (i : Nat) (up : Float) :
    (Expr.relu a).bwd inputs i up =
      a.bwd inputs i (up * (if a.eval inputs > 0.0 then 1.0 else 0.0)) := rfl

theorem grad_size (e : Expr) (inputs : Array Float) :
    (e.grad inputs).size = inputs.size := by
  simp [grad]

theorem backward_const (c : Float) (inputs : Array Float) (i : Nat) :
    (Expr.const c).backward inputs i = 0.0 := rfl

theorem backward_var_self (i : Nat) (inputs : Array Float) :
    (Expr.var i).backward inputs i = 1.0 := by
  simp [backward]

theorem backward_var_ne {i j : Nat} (h : j ≠ i) (inputs : Array Float) :
    (Expr.var j).backward inputs i = 0.0 := by
  simp [backward, bwd_var_ne h]

/-
Tests
-/

private def fdGrad (e : Expr) (inputs : Array Float) (i : Nat) (h : Float := 1e-4) : Float :=
  let plus  := inputs.set! i (inputs[i]! + h)
  let minus := inputs.set! i (inputs[i]! - h)
  (e.eval plus - e.eval minus) / (2.0 * h)

private def closeEnough (a b : Float) : Bool :=
  (a - b).abs < 1e-3

private def gradMatchesFd (e : Expr) (inputs : Array Float) : Bool :=
  (Array.range inputs.size).all fun i =>
    closeEnough (e.backward inputs i) (fdGrad e inputs i)

example : (Expr.const 3.14).backward #[1.0, 2.0] 0 = 0.0 := rfl
example : (Expr.var 0).backward #[5.0, 6.0] 0 = 1.0 := backward_var_self 0 _
example : (Expr.var 1).backward #[5.0, 6.0] 0 = 0.0 := backward_var_ne (by decide) _
example : ((Expr.var 0).grad #[7.0, 8.0]).size = 2 := grad_size _ _

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
