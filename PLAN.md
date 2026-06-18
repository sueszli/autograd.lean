# Port autograd.c → autograd.lean (microGPT target)

## Context

The repo `https://github.com/sueszli/autograd.lean` (already cloned at
`/Users/sueszli/dev/autograd.lean/` with Lean 4.31.0 + Lake skeleton + AGPL
license) hosts a Lean 4 port of `https://github.com/sueszli/autograd.c`.

The single deliverable is a trainable character-level transformer that mirrors
`sueszli/microgpt-benchmarks-game/sueszli_plain.py`. No other targets, no
proofs, no broader engine ambitions. Anything in autograd.c that isn't on the
microGPT call graph is dropped.

User constraints:
- **Pure functional** Lean. No `IO.Ref` in the hot path. `IO` only at the outer
  training loop and dataset load.
- **Pure Lean numerics**. Lean's `FloatArray`/`Array Float` with naive loops.
  No FFI, no BLAS.
- **No proofs**. The previous plan included a parallel `AutogradProofs`
  Mathlib-backed library. Dropped.

Outcome: ~1500 LOC Lean that loads `input.txt` (Tiny Shakespeare), runs the
identical 1-layer 16-dim 4-head model for 1000 AdamW steps, and produces a
loss curve within 1% of Python's reference.

## What microGPT needs (vs. autograd.c)

`sueszli_plain.py` is ~250 LOC. Constants: `N_LAYER=1`, `N_EMBED=16`,
`BLOCK_SIZE=16`, `N_HEAD=4`, `NUM_STEPS=1000`. Everything is 2D Python
`list[list[float]]`; per-head tensors are `list[list[list[float]]]`. No
broadcasting, no strides, no ndim. Every shape is statically known.

Required ops (mirroring the Python function names):
- `linear_fwd / linear_bwd_x / linear_bwd_w` — three flavors of matmul
- `softmax / softmax_bwd` — vector softmax + fused-with-grad backward (uses scale)
- `rmsnorm_fwd / rmsnorm_bwd` — row-wise scale by `1/√(mean(x²)+ε)`
- `attn_fwd / attn_bwd` — multi-head causal attention with manual head slicing
- `mlp_fwd / mlp_bwd` — RMSNorm → linear → ReLU → linear → residual
- `forward / backward` — embedding lookup + N_LAYER blocks + lm_head + masked CE
- `madd` — matrix add (residuals)
- `merge_heads` — concat heads back to embed dim
- AdamW step with linear LR decay (`lr = 0.01 * (1 - step/NUM_STEPS)`)
- `tokenize` — char-level, with BOS sentinel + fixed BLOCK_SIZE padding + loss mask

**Dropped from autograd.c entirely**: convolutions (`ops/convolutions*`), pooling,
batchnorm, CIFAR-10 (`utils/cifar10*`, `utils/augment*`), `aligned_alloc.h`
(Lean has GC), all `_autograd.c` tape-based tests (we don't reify a graph),
the entire `AutogradProofs` library and Mathlib dep, and `Examples/Micrograd.lean`
(synthetic MLP demo from the old plan).

## Module layout

Mirrors `autograd.c/src/` and `autograd.c/test/` one-to-one for the
microGPT-relevant subset. Forward and backward are split into sibling files,
matching the C convention.

```
autograd.c                                  autograd.lean
─────────────────────                       ──────────────────────────────
src/tensor.{c,h}                            Autograd/Tensor.lean
src/layers.{c,h}                            Autograd/Layers.lean
src/optimizers.{c,h}                        Autograd/Optimizers.lean
src/main.c                                  Main.lean
src/autograd.{c,h}                          Autograd.lean  (umbrella re-export;
                                              no tape engine — caches replace it)

src/ops/activations.{c,h}                   Autograd/Ops/Activations.lean
src/ops/activations_backward.{c,h}          Autograd/Ops/ActivationsBackward.lean
src/ops/arithmetic.{c,h}                    Autograd/Ops/Arithmetic.lean
src/ops/arithmetic_backward.{c,h}           Autograd/Ops/ArithmeticBackward.lean
src/ops/losses.{c,h}                        Autograd/Ops/Losses.lean
src/ops/losses_backward.{c,h}               Autograd/Ops/LossesBackward.lean
src/ops/reductions.{c,h}                    Autograd/Ops/Reductions.lean
src/ops/reductions_backward.{c,h}           Autograd/Ops/ReductionsBackward.lean
src/ops/reshapes.{c,h}                      Autograd/Ops/Reshapes.lean
src/ops/reshapes_backward.{c,h}             Autograd/Ops/ReshapesBackward.lean
src/ops/convolutions*.{c,h}                 (dropped)

src/utils/types.h                           (folded into Autograd/Tensor.lean)
src/utils/metrics.h                         Autograd/Utils/Metrics.lean
src/utils/tqdm.h                            Autograd/Utils/Tqdm.lean
src/utils/aligned_alloc.h                   (dropped — Lean GC)
src/utils/augment.{c,h}                     (dropped — CIFAR-only)
src/utils/cifar10.{c,h}                     (dropped — CIFAR-only)
                                            Autograd/Utils/Random.lean   (new: Box–Muller w/ seed 42)
                                            Autograd/Utils/Tokenizer.lean (new: char-level + BOS + mask)

test/test_tensor.c                          Test/TestTensor.lean
test/test_layers.c                          Test/TestLayers.lean
test/test_optimizers.c                      Test/TestOptimizers.lean
test/test_activations.c                     Test/TestActivations.lean
test/test_activations_backward.c            Test/TestActivationsBackward.lean
test/test_arithmetic.c                      Test/TestArithmetic.lean
test/test_arithmetic_backward.c             Test/TestArithmeticBackward.lean
test/test_losses.c                          Test/TestLosses.lean
test/test_losses_backward.c                 Test/TestLossesBackward.lean
test/test_reductions.c                      Test/TestReductions.lean
test/test_reductions_backward.c             Test/TestReductionsBackward.lean
test/test_reshapes.c                        Test/TestReshapes.lean
test/test_reshapes_backward.c               Test/TestReshapesBackward.lean
test/test_*_autograd.c                      (dropped — no tape engine)
test/test_convolutions*.c                   (dropped)
test/test_augment.c                         (dropped)
                                            Test/Golden/                  (Python parity)
```

Plus, already in tree and unchanged:
- `Autograd/Scalar.lean` — pedagogical scalar `Expr` engine (`var/const/add/mul/relu/tanh`)
  with recursive backward and FD checks. Not on the microGPT call graph; kept
  as a learning artifact.

## Per-op contents (microGPT scope)

The C source splits one category per file, forward and backward separated.
Each Lean file is a single namespace mirroring the C file.

| File | Contents |
|---|---|
| `Ops/Arithmetic.lean` | `linearFwd` (matmul), `madd` (elementwise add) |
| `Ops/ArithmeticBackward.lean` | `linearBwdX`, `linearBwdW`, `maddBwd` (identity pair) |
| `Ops/Activations.lean` | `relu` (elementwise), `softmax` (vector) |
| `Ops/ActivationsBackward.lean` | `reluBwd` (mask by `hPre > 0`), `softmaxBwd` (matrix form, takes `scale`) |
| `Ops/Reductions.lean` | `sum`, `mean`, `meanOfSquares` (only what RMSNorm/softmax need) |
| `Ops/ReductionsBackward.lean` | gradient broadcasts for the above |
| `Ops/Reshapes.lean` | `transpose`, `splitHeads`, `mergeHeads` |
| `Ops/ReshapesBackward.lean` | inverse rearrangements |
| `Ops/Losses.lean` | `maskedCrossEntropy` — `−Σ mask_i · log probs[i, target_i] / sumMask` |
| `Ops/LossesBackward.lean` | fused `(probs − onehot) * mask / sumMask` |

No `tanh` in the microGPT op set (`Scalar.lean` keeps it for its own demo).

## Core types

```lean
-- Autograd/Tensor.lean
abbrev Matrix := Array (Array Float)

namespace Matrix
def rows (m : Matrix) : Nat := m.size
def cols (m : Matrix) : Nat := if m.size = 0 then 0 else m[0]!.size
def zeros (r c : Nat) : Matrix := Array.replicate r (Array.replicate c 0.0)
def ofFn (r c : Nat) (f : Nat → Nat → Float) : Matrix
def transpose (m : Matrix) : Matrix
def madd (a b : Matrix) : Matrix              -- elementwise
```

Rationale for `Array (Array Float)` over `FloatArray + shape`: faithful to
`sueszli_plain.py` (which uses `list[list[float]]`); avoids stride arithmetic
microGPT doesn't need; pattern-matches one-to-one with the Python source so
porting is mechanical; row slicing for per-head views is `Array.extract`.
Trade-off: ~2× slower than packed `FloatArray` for matmul due to pointer
chasing. microGPT at N_EMBED=16 is tiny, so this is fine. If perf matters
later, swap the type alias and recompile; ops are written against the
namespace.

**No computation-graph reification.** The Python reference passes explicit
cache namedtuples (`AttnCache`, `MlpCache`, `FwdCache`) into backward. Lean
mirrors that: forward returns `(out, Cache)`, backward takes `Cache`. No DAG,
no topo-sort, no `HashMap TensorId Tensor`, no `partial def`. Whole library
is structurally recursive and total. The `test_*_autograd.c` tests in
autograd.c exercise its tape; we have no tape, so those tests have no
analogue.

Cache records (in `Autograd/Layers.lean`):

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
structure TransformerBlock where
  attnWq attnWk attnWv attnWo : Matrix
  mlpFc1 mlpFc2 : Matrix
structure Params where           -- mirrors Python state_dict
  wte wpe lmHead : Matrix
  blocks : Array TransformerBlock
```

Hyperparameters live in a `Config` record (not module-level constants) so
tests can vary them.

## Function signatures

```lean
-- Ops/Arithmetic.lean
def linearFwd  (x W : Matrix) : Matrix
def madd       (a b : Matrix) : Matrix

-- Ops/ArithmeticBackward.lean
def linearBwdX (dout W : Matrix) : Matrix          -- dout @ W
def linearBwdW (dout x : Matrix) : Matrix          -- doutᵀ @ x

-- Ops/Activations.lean
def relu    (m : Matrix) : Matrix
def softmax (v : Array Float) : Array Float

-- Ops/ActivationsBackward.lean
def reluBwd    (dout hPre : Matrix) : Matrix
def softmaxBwd (aw daw : Matrix) (scale : Float) : Matrix
                -- scale · aw_ij · (daw_ij − ⟨aw_i, daw_i⟩)

-- Layers.lean
def rmsnormFwd (x : Matrix) : Matrix × Array Float
def rmsnormBwd (dout x : Matrix) (rms : Array Float) : Matrix

def attnFwd (cfg : Config) (x : Matrix) (wq wk wv wo : Matrix)
            : Matrix × AttnCache
def attnBwd (cfg : Config) (dx : Matrix) (wq wk wv wo : Matrix) (c : AttnCache)
            : Matrix × (Matrix × Matrix × Matrix × Matrix)

def mlpFwd  (x fc1 fc2 : Matrix) : Matrix × MlpCache
def mlpBwd  (dx fc1 fc2 : Matrix) (c : MlpCache) : Matrix × (Matrix × Matrix)

def forward  (p : Params) (cfg : Config) (input target : Array Nat) (mask : Array Float)
             : Float × FwdCache
def backward (p : Params) (cfg : Config) (cache : FwdCache) : Grads

-- Ops/Losses.lean / LossesBackward.lean
def maskedCrossEntropy    (probs : Matrix) (targetIds : Array Nat)
                          (mask : Array Float) (sumMask : Float) : Float
def maskedCrossEntropyBwd (probs : Matrix) (targetIds : Array Nat)
                          (mask : Array Float) (sumMask : Float) : Matrix

-- Optimizers.lean
def adamWStep (cfg : Config) (step : Nat) (p : Params) (s : OptState) (g : Grads)
              : Params × OptState
```

## Phased delivery

| Phase | Deliverable | Effort |
|---|---|---|
| **0** | Already done: repo, lakefile, lean-toolchain, AGPL, README. `Autograd/Scalar.lean` (pedagogical scalar engine with FD checks) lives here too. | done |
| **1** | `Tensor.lean` (Matrix type + helpers), `Utils/Random.lean` (Box–Muller + seed 42), `Utils/Metrics.lean`, `Utils/Tqdm.lean`. LSpec wired in `lakefile.toml`. `Test/TestTensor.lean`. | 1 wk |
| **2** | All five `Ops/*` pairs (`Arithmetic`, `Activations`, `Reductions`, `Reshapes`, `Losses`) + matching `Test/Test*.lean` (per-op FD tests against symmetric difference quotient at `h=1e-4`, pass if `|backward − fd| < 1e-3`). | 2 wk |
| **3** | `Layers.lean` (RMSNorm + Attention + MLP + Model `forward`/`backward`), `Optimizers.lean` (AdamW + linear LR decay), `Utils/Tokenizer.lean`. `Test/TestLayers.lean`, `Test/TestOptimizers.lean`. `Main.lean` runs the full training loop on `input.txt`. | 2 wk |
| **4** | Golden tests: `scripts/gen_golden.py` runs `sueszli_plain.py` at seed 42 for K steps, dumps op-level intermediates + state_dict to `Test/Golden/*.json`. `Test/Golden/*.lean` asserts Lean ≈ Python to ~1e-5 elementwise. End-to-end: Lean loss curve over 1000 steps within 1% of Python's reference. | 1 wk |

**Total**: ~6 weeks to a trainable microGPT in pure Lean matching the Python loss curve.

## Verification

- **Per-op finite-difference tests** (`Test/Test*Backward.lean`): for each op,
  pick a small random input, compare hand-coded backward to symmetric
  difference quotient at `h = 1e-4`. Pass if `|backward − fd| < 1e-3`. Same
  technique as `Scalar.lean`'s existing `allclose` helper.
- **Python oracle golden tests** (`Test/Golden/*.lean`): regenerate via
  `python scripts/gen_golden.py` (runs `sueszli_plain.py` at seed 42 for K
  steps, dumps state_dict + per-step loss). Lean asserts post-K-step weights
  match Python to ~1e-5 elementwise. This is the bit-equivalence anchor.
- **End-to-end training run** on `input.txt` (Tiny Shakespeare): Lean loss
  curve over 1000 steps within 1% of Python's reference at the final step.

## Critical files for implementation

- `lakefile.toml` — add `batteries` dep, add LSpec dep, declare `Main` lean_exe.
- `Autograd/Tensor.lean` — central data type; every op file uses it.
- `Autograd/Ops/Arithmetic.lean` — bottom of the call graph; everything depends on `linearFwd`.
- `Autograd/Layers.lean` — biggest file; mirrors `attn_fwd`/`attn_bwd`/`mlp_fwd`/`mlp_bwd`/`forward`/`backward` closely. If it grows past ~600 LOC, split into `Layers/{RMSNorm,Attention,MLP,Model}.lean`.
- `Main.lean` — training loop driver; mirrors `autograd.c/src/main.c`.

## Known risks

1. **Box–Muller / Python random parity**: matching Python's `random.gauss(0, σ)`
   bit-for-bit may be infeasible (different RNG, different `log`/`sqrt` rounding).
   Mitigation: have the Python oracle script dump initial weights to
   `init_weights.json`; Lean loads that for the parity tests and skips its own RNG.
2. **AdamW bit-equivalence over 1000 steps**: `**0.5` (Python) vs `Float.sqrt`
   (Lean) may diverge in last bits. Mitigation: golden tolerance is `1e-5`
   elementwise, not bitwise.
3. **`Array (Array Float)` matmul perf**: pointer chasing on inner arrays.
   Acceptable at N_EMBED=16. If profiling shows it dominates, swap the type
   alias in `Tensor.lean` to a `FloatArray`-packed representation — ops only
   touch the namespace so the change is local.
