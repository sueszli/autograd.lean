import Autograd.Model
import Autograd.Rng
import Lean.Data.Json

namespace Autograd

open Lean (Json)

/-! ## initParams — random init (σ=0.08, per original.py) -/

def initParams (cfg : Config) (rng : RngState) : Params × RngState := Id.run do
  let σ : Float := 0.08
  let (wte, r1) := rngGaussMat cfg.vocabSize cfg.nEmbed σ rng
  let (wpe, r2) := rngGaussMat cfg.blockSize cfg.nEmbed σ r1
  let (lmHead, r3) := rngGaussMat cfg.nEmbed cfg.vocabSize σ r2
  let mut r := r3
  let mut blocks : Array TransformerBlock := #[]
  for h in [0:cfg.nLayer] do
    let (wq, r') := rngGaussMat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (wk, r') := rngGaussMat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (wv, r') := rngGaussMat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (wo, r') := rngGaussMat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (fc1, r') := rngGaussMat cfg.nEmbed (4 * cfg.nEmbed) σ r; r := r'
    let (fc2, r') := rngGaussMat (4 * cfg.nEmbed) cfg.nEmbed σ r; r := r'
    blocks := blocks.push {
      attnWq := Tensor.leaf wq (ParamIds.attnWq h) true,
      attnWk := Tensor.leaf wk (ParamIds.attnWk h) true,
      attnWv := Tensor.leaf wv (ParamIds.attnWv h) true,
      attnWo := Tensor.leaf wo (ParamIds.attnWo h) true,
      mlpFc1 := Tensor.leaf fc1 (ParamIds.mlpFc1 h) true,
      mlpFc2 := Tensor.leaf fc2 (ParamIds.mlpFc2 h) true
    }
  return ({
    wte := Tensor.leaf wte ParamIds.wte true,
    wpe := Tensor.leaf wpe ParamIds.wpe true,
    lmHead := Tensor.leaf lmHead ParamIds.lmHead true,
    blocks := blocks
  }, r)

/-! ## JSON-loaded init — for parity against original.py's dumped weights -/

private def asArr (j : Json) : IO (Array Json) :=
  match j.getArr? with
  | .ok a => pure a
  | .error e => throw (IO.userError s!"expected array: {e}")

private def asFloat (j : Json) : IO Float :=
  match j.getNum? with
  | .ok n => pure n.toFloat
  | .error e => throw (IO.userError s!"expected number: {e}")

private def getObj (j : Json) (k : String) : IO Json :=
  match j.getObjVal? k with
  | .ok v => pure v
  | .error e => throw (IO.userError s!"missing key '{k}': {e}")

private def asMatrix (j : Json) : IO Matrix := do
  let rows ← asArr j
  rows.mapM fun row => do (← asArr row).mapM asFloat

-- Python stores linear weights [out × in]; my linearFwd wants [in × out].
private def asMatrixT (j : Json) : IO Matrix := do
  return Matrix.transpose (← asMatrix j)

-- build a Params record from original.py's weights.json layout.
def paramsFromJson (j : Json) : IO Params := do
  return {
    wte := Tensor.leaf (← asMatrix (← getObj j "wte")) ParamIds.wte true,
    wpe := Tensor.leaf (← asMatrix (← getObj j "wpe")) ParamIds.wpe true,
    lmHead := Tensor.leaf (← asMatrixT (← getObj j "lm_head")) ParamIds.lmHead true,
    blocks := #[{
      attnWq := Tensor.leaf (← asMatrixT (← getObj j "layer0.attn_wq")) (ParamIds.attnWq 0) true,
      attnWk := Tensor.leaf (← asMatrixT (← getObj j "layer0.attn_wk")) (ParamIds.attnWk 0) true,
      attnWv := Tensor.leaf (← asMatrixT (← getObj j "layer0.attn_wv")) (ParamIds.attnWv 0) true,
      attnWo := Tensor.leaf (← asMatrixT (← getObj j "layer0.attn_wo")) (ParamIds.attnWo 0) true,
      mlpFc1 := Tensor.leaf (← asMatrixT (← getObj j "layer0.mlp_fc1")) (ParamIds.mlpFc1 0) true,
      mlpFc2 := Tensor.leaf (← asMatrixT (← getObj j "layer0.mlp_fc2")) (ParamIds.mlpFc2 0) true
    }]
  }

end Autograd
