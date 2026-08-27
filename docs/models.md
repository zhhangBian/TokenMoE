# vLLM model support

TokenMoE trace collection requires a routed MoE Hugging Face configuration and
the local `vllm/` submodule. Dense models are rejected rather than routed
through a synthetic fallback.

The collector derives and records:

- model type and architecture;
- number of decoder layers;
- real MoE layer IDs;
- Router top-k;
- logical Expert count;
- Router score semantics.

The current score-semantics table covers Qwen2/3-MoE and DeepSeek-V2/V3. A new
model family must add a capture hook in the vLLM fork and document the exact
score tensor captured after top-k selection and model-specific normalization.

## Required launch profile

Every run declares tensor and Expert parallelism, dtype, maximum model length,
and GPU memory utilization. Pipeline and context parallelism must be one, and KV
transfer must be disabled until their routed-output indexing is validated.

The vLLM submodule is pinned by Git. Build and import that checkout before the
root package:

```bash
export TOKENMOE_ROOT=/path/to/TokenMoE
PYTHONPATH="$TOKENMOE_ROOT/vllm:$TOKENMOE_ROOT" python -c \
  'import vllm; print(vllm.__file__)'
```

The printed path must point to the intended TokenMoE vLLM build. A successful
Python import is not sufficient evidence; each new configuration also requires
finite/aligned score checks and capture-on/off generated-token equivalence.
