import Autograd.Tensor
import Autograd.Model
import Autograd.Rng
import Lean.Data.Json

namespace Autograd

open Lean (Json)

def initParams (cfg : Config) (rng : RngState) : Params × RngState := Id.run do
  let σ : Float := 0.08
  let mkLeaf (data : Array Float) (rows cols id : Nat) : Tensor :=
    Tensor.leaf data rows cols id true
  let (wte, r1) := rngGaussFlat cfg.vocabSize cfg.nEmbed σ rng
  let (wpe, r2) := rngGaussFlat cfg.blockSize cfg.nEmbed σ r1
  let (lmHead, r3) := rngGaussFlat cfg.nEmbed cfg.vocabSize σ r2
  let mut r := r3
  let mut blocks : Array TransformerBlock := #[]
  for h in [0:cfg.nLayer] do
    let (wq, r') := rngGaussFlat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (wk, r') := rngGaussFlat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (wv, r') := rngGaussFlat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (wo, r') := rngGaussFlat cfg.nEmbed cfg.nEmbed σ r; r := r'
    let (fc1, r') := rngGaussFlat cfg.nEmbed (4 * cfg.nEmbed) σ r; r := r'
    let (fc2, r') := rngGaussFlat (4 * cfg.nEmbed) cfg.nEmbed σ r; r := r'
    blocks := blocks.push {
      attnWq := mkLeaf wq cfg.nEmbed cfg.nEmbed (ParamIds.attnWq h),
      attnWk := mkLeaf wk cfg.nEmbed cfg.nEmbed (ParamIds.attnWk h),
      attnWv := mkLeaf wv cfg.nEmbed cfg.nEmbed (ParamIds.attnWv h),
      attnWo := mkLeaf wo cfg.nEmbed cfg.nEmbed (ParamIds.attnWo h),
      mlpFc1 := mkLeaf fc1 cfg.nEmbed (4 * cfg.nEmbed) (ParamIds.mlpFc1 h),
      mlpFc2 := mkLeaf fc2 (4 * cfg.nEmbed) cfg.nEmbed (ParamIds.mlpFc2 h)
    }
  return ({
    wte := mkLeaf wte cfg.vocabSize cfg.nEmbed ParamIds.wte,
    wpe := mkLeaf wpe cfg.blockSize cfg.nEmbed ParamIds.wpe,
    lmHead := mkLeaf lmHead cfg.nEmbed cfg.vocabSize ParamIds.lmHead,
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

-- read a 2D nested array and flatten to (rows × cols) Array Float + dims
private def asRows (j : Json) : IO (Array Float × Nat × Nat) := do
  let rows ← asArr j
  let r := rows.size
  let c : Nat := ← if r = 0 then pure 0
                   else do let row0 ← asArr rows[0]!; pure row0.size
  let flat ← rows.foldlM (init := (#[] : Array Float)) fun acc row => do
    let cols ← asArr row
    return acc ++ (← cols.mapM asFloat)
  return (flat, r, c)

-- Python stores linear weights [out × in]; my linearFwd wants [in × out]. Transpose.
private def asRowsT (j : Json) : IO (Array Float × Nat × Nat) := do
  let (flat, r, c) ← asRows j
  return (transposeFlat flat r c, c, r)

private def leafFrom (triple : Array Float × Nat × Nat) (id : Nat) : Tensor :=
  Tensor.leaf triple.1 triple.2.1 triple.2.2 id true

def paramsFromJson (j : Json) : IO Params := do
  return {
    wte    := leafFrom (← asRows  (← getObj j "wte"))    ParamIds.wte,
    wpe    := leafFrom (← asRows  (← getObj j "wpe"))    ParamIds.wpe,
    lmHead := leafFrom (← asRowsT (← getObj j "lm_head")) ParamIds.lmHead,
    blocks := #[{
      attnWq := leafFrom (← asRowsT (← getObj j "layer0.attn_wq")) (ParamIds.attnWq 0),
      attnWk := leafFrom (← asRowsT (← getObj j "layer0.attn_wk")) (ParamIds.attnWk 0),
      attnWv := leafFrom (← asRowsT (← getObj j "layer0.attn_wv")) (ParamIds.attnWv 0),
      attnWo := leafFrom (← asRowsT (← getObj j "layer0.attn_wo")) (ParamIds.attnWo 0),
      mlpFc1 := leafFrom (← asRowsT (← getObj j "layer0.mlp_fc1")) (ParamIds.mlpFc1 0),
      mlpFc2 := leafFrom (← asRowsT (← getObj j "layer0.mlp_fc2")) (ParamIds.mlpFc2 0)
    }]
  }

end Autograd
