# Port autograd.c → autograd.lean (microGPT target)

## Context

The repo `https://github.com/sueszli/autograd.lean` (already cloned at
`/Users/sueszli/dev/autograd.lean/` with Lean 4.31.0 + Lake skeleton + AGPL
license) will host a Lean 4 port of the autograd engine from
`https://github.com/sueszli/autograd.c`.

Original plan targeted full feature parity with the C repo (conv2d, pooling,
batchnorm, CIFAR-10). User then narrowed the scope: **the end-to-end demo will
be the microGPT in `sueszli/microgpt-benchmarks-game/sueszli_plain.py`**, not a
CNN. That dramatically shrinks the op surface and the data plumbing.

User constraints (unchanged):
- **Pure functional** Lean — no `IO.Ref` in the autograd hot path. `IO` only at
  the outer training loop and dataset load.
- **Pure Lean numerics** — Lean's `FloatArray` with naive loops. No FFI, no BLAS.
- **Proofs of gradient correctness** for each op — ℝ-typed mirror in Mathlib +
  `HasFDerivAt` chain-rule payoff theorem. Float impl proven structurally
  identical to the ℝ mirror; Float-vs-ℝ rounding error explicitly out of scope.

Outcome: a small (~1500 LOC Lean) autograd library, plus a parallel
`AutogradProofs` library (~2500 LOC) carrying the Mathlib-backed correctness
theorems, plus an end-to-end character-level transformer that bit-matches the
Python reference at low precision.

## What microGPT actually needs (vs. autograd.c)

`sueszli_plain.py` is ~250 LOC. Constants: `N_LAYER=1`, `N_EMBED=16`,
`BLOCK_SIZE=16`, `N_HEAD=4`, `NUM_STEPS=1000`. Everything is 2D Python
`list[list[float]]`; per-head tensors are `list[list[list[float]]]`. No
broadcasting machinery, no strides, no ndim — every shape is statically known.

Required ops (mirroring the Python function names):
- `linear_fwd / linear_bwd_x / linear_bwd_w` — three flavors of matmul
- `softmax / softmax_bwd` — vector softmax + fused-with-grad backward (uses scale)
- `rmsnorm_fwd / rmsnorm_bwd` — row-wise scale by `1/√(mean(x²)+ε)`
- `attn_fwd / attn_bwd` — multi-head causal attention with manual head slicing
- `mlp_fwd / mlp_bwd` — RMSNorm → linear → ReLU → linear → residual
- `forward / backward` — embedding lookup + N_LAYER blocks + lm_head + masked CE
- `madd` — matrix add (used for residual + bias-free skip)
- `merge_heads` — concat heads back to embed dim
- AdamW step with linear LR decay (`lr = 0.01 * (1 - step/NUM_STEPS)`)
- `tokenize` — char-level, with BOS sentinel + fixed BLOCK_SIZE padding + loss mask

**Dropped from the original plan:** Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D,
general broadcasting/unbroadcast, im2col/col2im, CIFAR-10 data loading,
PyTorch-shaped Tensor with ndim/strides. Saves ~3000 LOC of C-equivalent work.

## Module layout

All under `/Users/sueszli/dev/autograd.lean/`. Two Lake libraries:

```
Autograd.lean                          -- umbrella, re-exports
Autograd/Basic.lean                    -- Float helpers, fold helpers
Autograd/Matrix.lean                   -- core type: Matrix = Array (Array Float)
                                       --   constructors (zeros, ofFn, gaussian),
                                       --   shape accessors, transpose, slice/concat,
                                       --   row/col helpers
Autograd/Ops/Linear.lean               -- linearFwd, linearBwdX, linearBwdW, madd
Autograd/Ops/Softmax.lean              -- softmax (vector), softmaxBwd (matrix form
                                       --   with `scale` for attention)
Autograd/Ops/RMSNorm.lean              -- rmsnormFwd : Matrix → Matrix × Array Float
                                       -- rmsnormBwd : (dout x rms) → Matrix
Autograd/Ops/Activation.lean           -- relu (elementwise, no autograd-graph node;
                                       --   mlp_bwd uses h_pre cache directly)
Autograd/Ops/Attention.lean            -- attnFwd / attnBwd; uses Linear + Softmax;
                                       --   head_dim = N_EMBED / N_HEAD
Autograd/Ops/MLP.lean                  -- mlpFwd / mlpBwd; uses Linear + RMSNorm + relu
Autograd/Ops/CrossEntropy.lean         -- maskedCrossEntropy with sum_mask normalization
                                       --   + fused backward: dlogits = (probs - onehot)/sm
Autograd/Model.lean                    -- Params record (Std.HashMap String Matrix),
                                       --   FwdCache / AttnCache / MlpCache records,
                                       --   forward, backward
Autograd/Optim.lean                    -- AdamW with linear LR schedule;
                                       --   pure (Params, OptState, step) → (Params, OptState)
Autograd/Random.lean                   -- StdGen wrapper, Gaussian via Box–Muller,
                                       --   deterministic with seed 42 for parity with Python
Autograd/Tokenizer.lean                -- charset → token map, BOS = |charset|,
                                       --   tokenize : String → InputIds × TargetIds × Mask
Examples/Scalar.lean                   -- Phase 1 demo: a-la-micrograd
Examples/Micrograd.lean                -- Phase 2 demo: 1-layer MLP on synthetic
Examples/MicroGPT.lean                 -- Phase 4 demo: full training loop on input.txt
Tests/                                 -- LSpec specs per module + golden tests
                                       --   from a Python oracle script
```

`AutogradProofs` (Mathlib-dependent, separate lake_lib, optional dep):

```
AutogradProofs/Ideal/Matrix.lean       -- M m n = Matrix (Fin m) (Fin n) ℝ
                                       --   (use Mathlib's Matrix type directly)
AutogradProofs/Ideal/Linear.lean       -- linear_ideal, softmax_ideal, rmsnorm_ideal, ...
AutogradProofs/Ideal/Attention.lean    -- attention_ideal (causal masked)
AutogradProofs/Proofs/Linear.lean      -- linear_backward_correct (bilinear ⇒ trivial)
AutogradProofs/Proofs/Softmax.lean     -- softmax Jacobian; quotient rule
AutogradProofs/Proofs/RMSNorm.lean     -- rmsnorm derivative; smooth composition
AutogradProofs/Proofs/Attention.lean   -- attention as composition (uses Linear + Softmax)
AutogradProofs/Proofs/CrossEntropy.lean -- fused softmax+CE lemma
AutogradProofs/Proofs/ChainRule.lean   -- payoff: per-op correctness ⇒ Model.backward = ∇Model.forward
AutogradProofs/Bridge/FloatReal.md     -- documents the Float/ℝ gap (no theorems)
```

## Core types

```lean
-- Autograd/Matrix.lean
abbrev Matrix := Array (Array Float)

namespace Matrix
def rows (m : Matrix) : Nat := m.size
def cols (m : Matrix) : Nat := if m.size = 0 then 0 else m[0]!.size
def zeros (r c : Nat) : Matrix := Array.replicate r (Array.replicate c 0.0)
def ofFn (r c : Nat) (f : Nat → Nat → Float) : Matrix
def transpose (m : Matrix) : Matrix
def madd (a b : Matrix) : Matrix              -- elementwise
def gaussian (gen : StdGen) (r c : Nat) (σ : Float) : Matrix × StdGen
```

Rationale for `Array (Array Float)` over `FloatArray + shape`: faithful to the
Python reference (which uses `list[list[float]]`); avoids stride arithmetic the
microGPT path doesn't need; pattern-matches one-to-one with the Python source so
porting is mechanical; row slicing for per-head views is `Array.extract`.
Trade-off: ~2× slower than packed `FloatArray` for matmul due to pointer chasing
on inner arrays, but microGPT at N_EMBED=16 is tiny so this is fine. If perf
becomes an issue, swap the type alias for a packed representation; ops are
written against the namespace so it's one-file-per-op refactor.

Computation-graph reification is **not used**. The Python reference passes
explicit cache namedtuples (`AttnCache`, `MlpCache`, `FwdCache`) into the
backward functions. Lean mirrors this: forward returns `(out, Cache)`, backward
takes `Cache`. This is the right design for a fixed-topology demo — no DAG, no
topo-sort, no `HashMap TensorId Tensor`, no `partial def`. The whole library
becomes structurally recursive and total.

Cache records:

```lean
structure AttnCache where
  xPre   : Matrix
  xn     : Matrix
  rms    : Array Float
  q k v  : Array Matrix          -- length N_HEAD, each (n × head_dim)
  attnW  : Array Matrix          -- per-head attention weights
  outFlat : Matrix
structure MlpCache where
  xPre xn : Matrix; rms : Array Float
  hPre h  : Matrix
structure FwdCache where
  inputIds targetIds : Array Nat
  lossMask : Array Float; sumMask : Float
  emb : Matrix; rmsInit : Array Float
  x : Matrix; probs : Matrix
  layerCaches : Array (AttnCache × MlpCache)
structure Params where           -- mirrors the Python state_dict
  wte wpe lmHead : Matrix
  blocks : Array TransformerBlock
structure TransformerBlock where
  attnWq attnWk attnWv attnWo : Matrix
  mlpFc1 mlpFc2 : Matrix
```

Hyperparameters as a `Config` record (not module-level constants like Python) so
tests can vary them.

## Function signatures (forward/backward pairs)

```lean
def linearFwd  (x W : Matrix) : Matrix
def linearBwdX (dout W : Matrix) : Matrix          -- dout @ W
def linearBwdW (dout x : Matrix) : Matrix          -- doutᵀ @ x

def softmax     (v : Array Float) : Array Float
def softmaxBwd  (aw daw : Matrix) (scale : Float) : Matrix
                -- elementwise: scale · aw_ij · (daw_ij − ⟨aw_i, daw_i⟩)

def rmsnormFwd (x : Matrix) : Matrix × Array Float
def rmsnormBwd (dout x : Matrix) (rms : Array Float) : Matrix

def attnFwd (cfg : Config) (x : Matrix) (wq wk wv wo : Matrix)
            : Matrix × AttnCache
def attnBwd (cfg : Config) (dx : Matrix) (wq wk wv wo : Matrix) (c : AttnCache)
            : Matrix × (Matrix × Matrix × Matrix × Matrix)   -- dx, (dwq dwk dwv dwo)

def mlpFwd  (x fc1 fc2 : Matrix) : Matrix × MlpCache
def mlpBwd  (dx fc1 fc2 : Matrix) (c : MlpCache) : Matrix × (Matrix × Matrix)

def maskedCrossEntropy (probs : Matrix) (targetIds : Array Nat) (mask : Array Float) (sumMask : Float) : Float
def maskedCrossEntropyBwd (probs : Matrix) (targetIds : Array Nat) (mask : Array Float) (sumMask : Float) : Matrix

def forward  (p : Params) (cfg : Config) (input target : Array Nat) (mask : Array Float) : Float × FwdCache
def backward (p : Params) (cfg : Config) (cache : FwdCache) : Grads

def adamWStep (cfg : Config) (step : Nat) (p : Params) (s : OptState) (g : Grads) : Params × OptState
```

## Idealised semantics & per-op proofs

`AutogradProofs/Ideal/*.lean` uses Mathlib's `Matrix (Fin m) (Fin n) ℝ` directly.
Key theorem shells (these are the payoff per op):

```lean
theorem linear_bwdX_correct (W : Matrix (Fin out) (Fin k) ℝ) (x : Matrix (Fin n) (Fin k) ℝ) :
    HasFDerivAt (fun x' => linear_ideal x' W) ((linear_ideal · W) : _ →L[ℝ] _) x

theorem softmax_bwd_correct (v : EuclideanSpace ℝ (Fin n)) :
    HasFDerivAt softmax_ideal (softmaxJacobian v) v
    -- Jᵢⱼ = s_i(δᵢⱼ − s_j)

theorem rmsnorm_bwd_correct (x : EuclideanSpace ℝ (Fin n)) (hx : ∑ i, x i ^ 2 ≠ 0) :
    HasFDerivAt rmsnorm_ideal (rmsnormJacobian x) x

theorem ce_softmax_fused (logits : EuclideanSpace ℝ (Fin V)) (y : Fin V) :
    (fun ℓ => -Real.log (softmax_ideal ℓ y)) has fderiv (softmax_ideal logits − e_y) at logits

-- payoff:
theorem backward_eq_grad (p : Params) (input target : List Nat) (mask : List ℝ) :
    Model.backward p cache = fderiv (fun p' => Model.forward p' input target mask) p
```

Mathlib deps: `Mathlib.Analysis.Calculus.FDeriv.{Basic,Add,Mul,Comp,Prod,Pi}`,
`Mathlib.Analysis.SpecialFunctions.{Exp,Log}`,
`Mathlib.Analysis.InnerProductSpace.{EuclideanDist,Calculus}`,
`Mathlib.Analysis.NormedSpace.BoundedLinearMaps` (for linear's bilinearity),
`Mathlib.LinearAlgebra.Matrix.{Basic,Adjoint}`.

Hard cases (smaller list now that conv/BN are gone):
- **ReLU at 0**: prove `HasFDerivAt` only on `{x | x ≠ 0}`; cite subgradient
  convention; the impl matches.
- **Softmax + CE fusion**: prove unfused softmax Jacobian + unfused CE; then
  separately prove `softmax_logits − onehot = (∇CE) ∘ Jsoftmax`. The fused impl
  matches by `simp`.
- **RMSNorm denominator ≠ 0**: parametrize by `hx : ∑ x_i² + ε > 0`. The `+ε` in
  the impl guarantees this; lift as a non-zero hypothesis.
- **Attention as composition**: just chain Linear + reshape + softmax + masked
  Linear. The causal mask is a finite-support modification — prove on the
  unmasked side, mask is constant-additive (so derivative pass-through).

## Phased delivery

| Phase | Deliverable | Effort |
|---|---|---|
| **0** | Already done: repo, lakefile, lean-toolchain, AGPL, README. | done |
| **1** | `Basic`, `Matrix`, `Random` (Box–Muller w/ seed parity to Python), LSpec wired. `Examples/Scalar.lean` for a-la-micrograd (Value type, +, *, relu, tanh, recursive backward, finite-difference grad check). | 1 wk |
| **2** | `Ops/Linear`, `Ops/Softmax`, `Ops/RMSNorm`, `Ops/Activation`, `Ops/CrossEntropy`. Forward+backward functions; LSpec unit tests per op against finite-difference grads. | 2 wk |
| **3** | `Ops/Attention`, `Ops/MLP`, `Model.forward`/`backward`, `Optim.adamWStep`, `Tokenizer`. `Examples/MicroGPT.lean` with the same hyperparams; train on `input.txt`. | 2 wk |
| **4** | Golden tests: a Python oracle script `scripts/gen_golden.py` calls `sueszli_plain.py` op-by-op with fixed seed, dumps to `Tests/Golden/*.json` (or `.lean` literal). LSpec asserts Lean impl ≈ Python to ~1e-6. | 1 wk |
| **5** | `AutogradProofs` setup: lakefile `[[lean_lib]]` for proofs lib, Mathlib dep, `Ideal/*` mirrors. | 1 wk |
| **6** | Per-op proofs in order: Linear → RMSNorm → Softmax → CE fusion → Attention composition → MLP composition. | 6 wk |
| **7** | `Proofs/ChainRule.lean` payoff theorem tying `Model.backward = fderiv Model.forward`. | 2 wk |
| **8** | Optional perf pass (FloatArray-backed Matrix swap, profile-guided). | 1 wk |

**Realistic MVP** (Phases 1–4, no proofs): ~6 weeks, ships a trainable microGPT
in pure Lean that matches Python loss curves.

**Full deliverable** including proofs: ~3.5 months for one Lean+Mathlib-fluent
engineer. Float-vs-ℝ rounding proofs explicitly **out of scope**.

## Verification

- **Per-op finite-difference tests** (`Tests/FiniteDiff.lean`): for each op,
  pick a random small input, compare hand-coded backward to symmetric
  difference quotient at `h = 1e-4`. Pass if `|backward − fd| < 1e-3`.
- **Python oracle golden tests** (`Tests/Golden/*.lean`): regenerate via
  `python scripts/gen_golden.py` (runs `sueszli_plain.py` with seed 42 for K
  steps, dumps state_dict + per-step loss). Lean asserts post-K-step weights
  match Python to ~1e-5 elementwise. This is the bit-equivalence anchor.
- **End-to-end training run** on `input.txt` (Tiny Shakespeare): Lean loss
  curve over 1000 steps matches Python's `step_times.json` reference. Final
  loss within 1% of Python.
- **Proof CI**: `lake build AutogradProofs` runs on nightly cron only
  (Mathlib cold-build is slow). On main, only `lake build Autograd` + `lake
  test` (LSpec).

## Critical files for implementation

- `/Users/sueszli/dev/autograd.lean/lakefile.toml` — add `AutogradProofs`
  lean_lib, add `batteries` and (proofs-only) `mathlib` deps, declare
  `Examples/MicroGPT` lean_exe.
- `/Users/sueszli/dev/autograd.lean/Autograd/Matrix.lean` — central data type;
  every op file uses it.
- `/Users/sueszli/dev/autograd.lean/Autograd/Ops/Linear.lean` — bottom of the
  call graph; everything depends on it.
- `/Users/sueszli/dev/autograd.lean/Autograd/Ops/Attention.lean` — biggest
  single op; mirrors the Python `attn_fwd`/`attn_bwd` closely.
- `/Users/sueszli/dev/autograd.lean/Autograd/Model.lean` — the `Params`,
  `FwdCache`, `forward`/`backward` definitions that Phase 7's chain-rule
  payoff theorem talks about.
- `/Users/sueszli/dev/autograd.lean/AutogradProofs/Proofs/ChainRule.lean` —
  the headline theorem.

## Known risks

1. **Box–Muller / random parity with Python**: matching Python's
   `random.gauss(0, σ)` bit-for-bit may be infeasible (different RNG, different
   `log`/`sqrt` rounding). Mitigation: read the Python RNG state into Lean once
   per run via a generated `init_weights.json`; bypass Lean RNG for the parity
   tests.
2. **AdamW step bit-equivalence**: `**0.5` (Python) vs `Float.sqrt` (Lean) may
   diverge in last bits over 1000 steps. Mitigation: define golden tolerance
   per Phase 4 at `1e-5`, not bitwise; if user wants stricter, add a Bridge
   lemma documenting the IEEE-754 caveat.
3. **Mathlib build time**: cold ~10 min, warm ~30 s incremental. Keeps proofs
   library separate so the impl library stays fast.
