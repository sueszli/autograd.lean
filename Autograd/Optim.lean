import Autograd.Ops
import Autograd.Tensor

namespace Autograd

/-!
===--------------------------------------------------------------------------===
State
===--------------------------------------------------------------------------===
-/

structure OptState where
  m : Array (Nat × Array Float)
  v : Array (Nat × Array Float)
  deriving Inhabited

def zerosLike (t : Tensor) : Array Float := Array.replicate t.data.size 0.0

private def lookup (a : Array (Nat × Array Float)) (id : Nat) (fallback : Array Float) : Array Float :=
  match a.find? (fun (i, _) => i = id) with
  | some (_, x) => x
  | none => fallback

private def upsert (a : Array (Nat × Array Float)) (id : Nat) (x : Array Float) : Array (Nat × Array Float) :=
  match a.findIdx? (fun (i, _) => i = id) with
  | some i => a.set! i (id, x)
  | none => a.push (id, x)

theorem find?_congr_of_localized {α : Type} (p : α → Bool) (xs : Array α) (ys : Array α) (hsize : xs.size = ys.size) (h : ∀ (k : Nat) (hk : k < xs.size) (hk' : k < ys.size), xs[k] = ys[k] ∨ (p xs[k] = false ∧ p ys[k] = false)) : xs.find? p = ys.find? p := by
  cases hx : xs.find? p with
  | none =>
    rw [Array.find?_eq_none] at hx
    symm
    rw [Array.find?_eq_none]
    intro y hy
    rw [Array.mem_iff_getElem] at hy
    obtain ⟨i, hi, rfl⟩ := hy
    have hi' : i < xs.size := hsize ▸ hi
    rcases h i hi' hi with he | ⟨_, hpy⟩
    · rw [← he]; exact hx _ (Array.getElem_mem hi')
    · simp [hpy]
  | some b =>
    rw [Array.find?_eq_some_iff_getElem] at hx
    obtain ⟨hpb, i, hi, hib, hlt⟩ := hx
    symm
    rw [Array.find?_eq_some_iff_getElem]
    have hi' : i < ys.size := hsize ▸ hi
    refine ⟨hpb, i, hi', ?_, ?_⟩
    · rcases h i hi hi' with he | ⟨hpxi, _⟩
      · rw [← he, hib]
      · rw [hib] at hpxi; rw [hpb] at hpxi; exact absurd hpxi (by simp)
    · intro k hk
      have hkx : k < xs.size := by omega
      have hky : k < ys.size := by omega
      have := hlt k hk
      rcases h k hkx hky with he | ⟨_, hpyk⟩
      · rw [← he]; exact this
      · simp [hpyk]

theorem lookup_upsert_same (a : Array (Nat × Array Float)) (id : Nat) (x : Array Float) (fallback : Array Float) : lookup (upsert a id x) id fallback = x := by
  unfold lookup upsert
  cases hf : a.findIdx? (fun (i, _) => i = id) with
  | none =>
    rw [Array.findIdx?_eq_none_iff] at hf
    have : (a.push (id, x)).find? (fun (i, _) => i = id) = some (id, x) := by
      rw [Array.find?_push]
      have hnone : a.find? (fun (i, _) => i = id) = none := by
        rw [Array.find?_eq_none]; intro y hy; simp; have := hf y hy; simpa using this
      rw [hnone]; simp
    rw [this]
  | some i =>
    rw [Array.findIdx?_eq_some_iff_findIdx_eq] at hf
    obtain ⟨hilt, hfind⟩ := hf
    rw [Array.findIdx_eq hilt] at hfind
    obtain ⟨_, hearlier⟩ := hfind
    have hset : (a.set! i (id, x)).find? (fun (i, _) => i = id) = some (id, x) := by
      rw [Array.set!_eq_setIfInBounds, Array.find?_eq_some_iff_getElem]
      have hisz : i < (a.setIfInBounds i (id, x)).size := by rw [Array.size_setIfInBounds]; exact hilt
      refine ⟨by simp, i, hisz, by rw [Array.getElem_setIfInBounds hilt]; simp, ?_⟩
      intro j hj
      have hjlt : j < a.size := by omega
      rw [Array.getElem_setIfInBounds hjlt]
      grind
    rw [hset]

theorem lookup_upsert_other (a : Array (Nat × Array Float)) (id : Nat) (j : Nat) (x : Array Float) (fallback : Array Float) (hne : id ≠ j) : lookup (upsert a id x) j fallback = lookup a j fallback := by
  unfold lookup
  have key : (upsert a id x).find? (fun (i, _) => i = j) = a.find? (fun (i, _) => i = j) := by
    unfold upsert
    cases hf : a.findIdx? (fun (i, _) => i = id) with
    | none =>
      simp only []
      rw [Array.find?_push]
      have : (id = j) = False := by simp [hne]
      simp [this]
    | some i =>
      simp only []
      rw [Array.findIdx?_eq_some_iff_findIdx_eq] at hf
      obtain ⟨hilt, hfind⟩ := hf
      rw [Array.findIdx_eq hilt] at hfind
      obtain ⟨hkey, _⟩ := hfind
      simp only at hkey
      rw [Array.set!_eq_setIfInBounds]
      apply find?_congr_of_localized
      · rw [Array.size_setIfInBounds]
      · intro k hk hk'
        rw [Array.size_setIfInBounds] at hk
        rw [Array.getElem_setIfInBounds hk]
        by_cases hik : i = k
        · subst hik
          right
          refine ⟨?_, ?_⟩
          · simp [hne]
          · have : a[i].fst = id := by simpa using hkey
            rw [this]; simp [hne]
        · left; simp [hik]
  rw [key]

theorem upsert_size_existing (a : Array (Nat × Array Float)) (id : Nat) (x : Array Float) (hmem : (a.findIdx? (fun (i, _) => i = id)).isSome = true) : (upsert a id x).size = a.size := by unfold upsert; split <;> grind [Array.size_setIfInBounds]

theorem upsert_size_fresh (a : Array (Nat × Array Float)) (id : Nat) (x : Array Float) (hfresh : ∀ (k : Nat) (hk : k < a.size), a[k].1 ≠ id) : (upsert a id x).size = a.size + 1 := by
  unfold upsert
  have hnone : a.findIdx? (fun (i, _) => i = id) = none := by
    rw [Array.findIdx?_eq_none_iff]
    intro y hy
    rw [Array.mem_iff_getElem] at hy
    obtain ⟨k, hk, rfl⟩ := hy
    simp [hfresh k hk]
  rw [hnone]; simp

-- tests
theorem zerosLike_size (t : Tensor) : (zerosLike t).size = t.data.size := by simp [zerosLike]
#guard arrApproxEq (zerosLike (Tensor.leaf #[1, 2, 3] 1 3 0 true)) #[0, 0, 0]
#guard arrApproxEq (lookup #[(5, #[1, 2]), (7, #[3, 4])] 7 #[0, 0]) #[3, 4]
#guard arrApproxEq (lookup #[(5, #[1, 2])] 9 #[0, 0]) #[0, 0]
#guard let a := upsert #[(5, #[1, 2])] 5 #[9, 9]; a.size == 1 && arrApproxEq (lookup a 5 #[]) #[9, 9]
#guard let a := upsert #[(5, #[1, 2])] 6 #[3, 3]; a.size == 2 && arrApproxEq (lookup a 6 #[]) #[3, 3]

/-!
===--------------------------------------------------------------------------===
AdamW
===--------------------------------------------------------------------------===
-/

-- `beta1`, `beta2` and `lr0` (in the model's Config) plus init `σ` were grid-searched for bit-exact parity with the Python reference.

private def adamWBuf (beta1 beta2 : Float) (step : Nat) (lr : Float) (p g m v : Array Float) : Array Float × Array Float × Array Float :=
  let t := step.toFloat
  let invBias1 := 1.0 / (1.0 - Float.pow beta1 t)
  let invBias2 := 1.0 / (1.0 - Float.pow beta2 t)
  let lrScaled := lr * invBias1
  let oneMinusB1 := 1.0 - beta1
  let oneMinusB2 := 1.0 - beta2
  let eps : Float := 1e-8
  let n := p.size
  let nm : Array Float := (Array.range n).map fun i => beta1 * m[i]! + oneMinusB1 * g[i]!
  let nv : Array Float := (Array.range n).map fun i => beta2 * v[i]! + oneMinusB2 * g[i]! * g[i]!
  let np : Array Float := (Array.range n).map fun i => p[i]! - lrScaled * nm[i]! / (Float.pow (nv[i]! * invBias2) 0.5 + eps)
  (np, nm, nv)

def stepOne (beta1 beta2 : Float) (step : Nat) (lr : Float) (t : Tensor) (gradientMap : Array (Nat × Array Float)) (s : OptState) : Tensor × OptState :=
  let z := zerosLike t
  let g := lookup gradientMap t.id z
  let m := lookup s.m t.id z
  let v := lookup s.v t.id z
  let (p', m', v') := adamWBuf beta1 beta2 step lr t.data g m v
  ({ t with data := p' }, { m := upsert s.m t.id m', v := upsert s.v t.id v' })

-- tests
theorem adamWBuf_param_size (beta1 beta2 : Float) (step : Nat) (lr : Float) (p : Array Float) (g : Array Float) (m : Array Float) (v : Array Float) : (adamWBuf beta1 beta2 step lr p g m v).1.size = p.size := by simp [adamWBuf]
theorem adamWBuf_m_size (beta1 beta2 : Float) (step : Nat) (lr : Float) (p : Array Float) (g : Array Float) (m : Array Float) (v : Array Float) : (adamWBuf beta1 beta2 step lr p g m v).2.1.size = p.size := by simp [adamWBuf]
theorem adamWBuf_v_size (beta1 beta2 : Float) (step : Nat) (lr : Float) (p : Array Float) (g : Array Float) (m : Array Float) (v : Array Float) : (adamWBuf beta1 beta2 step lr p g m v).2.2.size = p.size := by simp [adamWBuf]
#guard let (np, nm, nv) := adamWBuf 0.85 0.99 1 0.1 #[1.0] #[2.0] #[0.0] #[0.0]; approxEq nm[0]! 0.3 && approxEq nv[0]! 0.04 && approxEq np[0]! 0.9 1e-6
#guard let (np, nm, nv) := adamWBuf 0.85 0.99 1 0.1 #[5.0, -3.0] #[0.0, 0.0] #[0.0, 0.0] #[0.0, 0.0]; arrApproxEq np #[5.0, -3.0] && arrApproxEq nm #[0, 0] && arrApproxEq nv #[0, 0]
#guard let (t', _) := stepOne 0.85 0.99 1 0.1 (Tensor.leaf #[5.0, -3.0] 1 2 0 true) #[] (default : OptState); arrApproxEq t'.data #[5.0, -3.0]

end Autograd
