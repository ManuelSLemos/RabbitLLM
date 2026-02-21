# Compatibility

## Transformers version

- **Supported**: `transformers>=5.0,<5.3` (5.0.x–5.2.x).
- **Qwen2/Qwen2.5 with transformers 5.1+**:
  - **RoPE 14 vs 64**: In layer-streaming with Qwen2/Qwen2.5 there may be an error on incremental forward (second forward with `past_key_values`): `RuntimeError: The size of tensor a (14) must match the size of tensor b (64) at non-singleton dimension 3` in `apply_rotary_pos_emb`. Cause: incorrect `head_dim` in attention. **Workaround**: use transformers 5.0.x for Qwen2/Qwen2.5; other architectures work fine on 5.1+.
  - **KV cache**: In 5.0+ decoder layers for Qwen2 and similar do not return the cache in the output tuple; they update `DynamicCache` in-place. The engine uses a fallback reading from the cache object (`.layers[0].keys`/`.values` or legacy `.key_cache`/`.value_cache`) and passes `cache_position` and the same object for reading. On 5.1+, if the "KV cache was not filled" warning appears, re-validate the DynamicCache API and the fallback. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#kv-cache-not-filled--no-incremental-decoding).

The codebase uses `GenerationMixin` from `transformers.generation.utils` (with fallback from `transformers`) and `Cache`/`DynamicCache` from `transformers.cache_utils`.

Previous 4.x support (reference). Upgrading to 4.47 from 4.46 brought:

- Better support for Qwen2.5, Llama 3.x, and modern configs (e.g. rope_scaling).
- Native SDPA and FlashAttention 2 integration.
- Cache utilities (`DynamicCache`) as the standard; our code uses them when available.
- In 4.47+, `GenerationMixin` remains available via `from transformers import GenerationMixin`; no code change required for the base model.

## Gated models

Some repos (e.g. Meta Llama, certain Gemma variants) are gated. Use a Hugging Face token: pass `token="hf_..."` (preferred; required in transformers v5) or `hf_token="hf_..."` for backward compatibility, or set the `HF_TOKEN` environment variable. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#gated-models-hugging-face).

## Dependencies

- **accelerate** ≥ 1.1.0 (required for transformers 5.x).
- **sentencepiece** (required for Baichuan tokenizer; add to project dependencies if you use Baichuan).
- **flash-attn** (optional): for Flash Attention 2; requires Ampere+ GPU (compute capability ≥ 8.0) and fp16/bf16.

## Attention implementation (Flash Attention)

The default **`attn_implementation="auto"`** selects the best implementation automatically:

- **Flash Attention 2** is used when:
  - The optional package `flash-attn` is installed (`pip install flash-attn` or `uv sync --extra flash`),
  - CUDA is available and the GPU has compute capability ≥ 8.0 (Ampere or newer),
  - The model dtype is fp16 or bf16,
  - A minimal runtime check passes (avoids broken installs or ABI mismatches).
- Otherwise **SDPA** (PyTorch scaled dot-product attention) is used, with fallback to **eager** if the model does not support SDPA.

You do not need to set `attn_implementation="flash_attention_2"` manually on compatible machines; leave the default `"auto"` so the engine enables Flash when possible. To force a specific implementation, pass `"flash_attention_2"`, `"sdpa"`, or `"eager"` (the model creation will still try the fallback chain if the requested one is unavailable).

### How to check if Flash is active and if it helps

- **Compatibility**: Run `from rabbitllm.utils.platform import is_flash_attention_available; print(is_flash_attention_available())` to see whether your system can use Flash (and why not if it returns `(False, "...")`).
- **What the model uses**: After loading with `AutoModel.from_pretrained(..., attn_implementation="auto")`, check `model.active_attention_implementation` — it will be `"flash_attention_2"`, `"sdpa"`, or `"eager"`.
- **Logs**: With `logging` at INFO level, you will see either `Attention: Flash Attention 2 available on <GPU name>` or `Flash Attention not available: ... Using SDPA` when the model is created, and `Model initialized with attn_implementation='...'`.
- **Benchmark**: Use `uv run python scripts/check_attention_and_benchmark.py --benchmark` to print compatibility, the active implementation, and a short throughput comparison (auto vs sdpa vs eager). Higher tokens/s with `auto` (when it uses Flash) means you are getting a benefit.

## Requirements for transformers 5.x

This project targets `transformers>=5.0`. Ensure: Python 3.10+, **PyTorch ≥ 2.4** (transformers 5.0 uses APIs that require 2.4+), **accelerate** ≥ 1.1.0, **peft** ≥ 0.18.0 (if using PEFT), **bitsandbytes** ≥ 0.46.1 (if using quantization). See [TRANSFORMERS_UPGRADE_PLAN.md](TRANSFORMERS_UPGRADE_PLAN.md).

### PyTorch 2.5 and Flash Attention

**Transformers 5.0** effectively requires PyTorch ≥ 2.4 (e.g. `torch.is_autocast_enabled(device_type)`). Going **down** to PyTorch 2.5 (from 2.10) is supported and often **recommended** if you want Flash Attention with **prebuilt wheels** (2.5 has better wheel coverage than 2.10 for many CUDA/Python combinations). What it involves:

1. **Constraint in the project**: This project uses `torch>=2.5,<2.6` so that `uv sync` / `pip install` resolves to PyTorch 2.5.x exactly, improving the chance of finding a prebuilt flash-attn wheel. No code changes in RabbitLLM are required for 2.5.
2. **Recreate the environment**: After changing the constraint, run `uv sync` (or `pip install -e .` / reinstall). The lockfile will resolve to PyTorch 2.5.x (and matching CUDA variant if you use a PyTorch index).
3. **Flash-attn**: With PyTorch 2.5, use [flashattn.dev](https://flashattn.dev) to get a prebuilt wheel for your Python/CUDA, or try `uv sync --extra flash` again (wheels for 2.5 are more commonly available).
4. **Risks**: None for 2.5 vs 2.10 for this project; we do not rely on 2.10-specific APIs. You only “lose” very new PyTorch features if any; for inference with transformers 5.0, 2.5 is sufficient.

## Model compatibility matrix

| Model / family           | Layer-streaming | Tied lm_head handling | Cache (past_key_value) | Registry mapping   |
|--------------------------|-----------------|------------------------|-------------------------|--------------------|
| **Llama2 / Llama3 / 3.2**| Yes             | Yes                    | Standard                | RabbitLLMLlama2    |
| **Qwen2 / Qwen2.5 / Qwen3** | Yes          | Yes                    | Standard                | RabbitLLMQWen2     |
| **Mistral / Mixtral**    | Yes             | Yes                    | Standard                | RabbitLLMMistral/Mixtral |
| **InternLM**             | Yes             | Yes                    | Standard                | RabbitLLMInternLM  |
| **Baichuan**             | Yes*            | Yes                    | Standard                | RabbitLLMBaichuan  |
| **Gemma2 / Gemma3**      | Yes**           | Yes                    | Standard                | Llama-like         |
| **DeepSeek V2 / V3**     | Yes**           | Yes                    | Standard                | Llama-like         |
| **Phi2 / Phi3 / Phi4**   | Yes**           | Yes                    | Standard                | Llama-like         |
| **QWen v1**              | Yes             | N/A                    | Uses `layer_past`       | RabbitLLMQWen      |
| **ChatGLM**              | Yes             | N/A                    | Uses `kv_cache`         | RabbitLLMChatGLM   |

\* Baichuan uses a custom tokenizer (sentencepiece); ensure the dependency is installed.

\*\* Gemma, DeepSeek, Phi are routed to the Llama-based implementation; layer layout is compatible. If a model fails (e.g. different layer names), a dedicated subclass may be needed.

### Qwen2 / Qwen2.5 with transformers 4.47+

- Decoder layers expect **`position_embeddings`** (cos, sin tuple) from RoPE; `RabbitLLMQWen2` overrides `get_pos_emb_args()` to compute and pass them.
- **RoPE head_dim**: Some configs set `head_dim` to `num_attention_heads` (e.g. 14) instead of `hidden_size // num_attention_heads` (e.g. 64), causing a shape mismatch in `apply_rotary_pos_emb`. The engine applies several fixes: (1) set `config.head_dim` to the canonical value in `__init__` and at the start of `init_model()`; (2) `_fix_attention_head_dim()` forces the same value on all decoder `self_attn` modules after creating the model and at the start of the layer loop; (3) Qwen2’s `get_pos_emb_args()` uses the canonical head_dim and treats `head_dim == num_attention_heads` as wrong and uses the canonical value for cos/sin. With **transformers 5.1+** a runtime mismatch (14 vs 64) occurs in layer-streaming; use **transformers 5.0.x** for Qwen2/Qwen2.5 until a fix is available.
- **KV cache en layer-streaming**: En 4.47+ las capas decoder no devuelven el cache en la tupla; actualizan el `DynamicCache` in-place. El motor pasa un cache vacío, llama a la capa y, si la salida no trae KV, lee del mismo objeto (nueva API `.layers[0].keys`/`.values` o legacy `.key_cache`/`.value_cache`) para rellenar `past_key_values`. Hay que usar `return_dict=True` para que la salida tenga `past_key_values`. Si el cache no se rellena, cada paso re-ejecuta el forward completo (throughput bajo). En **versiones superiores de transformers** (5.1+), comprobar que la API de Cache y este fallback sigan siendo válidos; anotado para resolver al subir de versión.

### Fixes that apply to all models (base engine)

- **`config.head_dim`**: If the config has `hidden_size` and `num_attention_heads`, the engine sets `config.head_dim = hidden_size // num_attention_heads` before creating the model so RoPE and attention use the correct dimension (avoids wrong values from hub or older configs).
- **`get_sequence_len(seq)`**: Handles both 3D tensors `(batch, seq_len, hidden)` and 2D `(seq_len, hidden)` so the sequence length used for position embeddings is correct (avoids using `hidden_size` as length).
- **`_reset_model()`**: After re-creating the model skeleton, the engine calls `set_layers_from_layer_names()` so `self.layers` always refers to the current model’s layers.
- **`_fix_attention_head_dim()`**: For any model with `hidden_size` and `num_attention_heads`, the engine sets each decoder layer’s `self_attn.head_dim` to the canonical value. This is required when the config or transformers creates attention with a wrong `head_dim` (e.g. Qwen2.5-0.5B).

### Cache compatibility note

Models that use a **non-standard** cache keyword (QWen v1: `layer_past`, ChatGLM: `kv_cache`) are compatible only as long as the code path uses the **legacy tuple** format. If `_uses_cache_objects` is True for all models (DynamicCache), those two would receive `past_key_value` instead of their expected key and generation could break. Future work may make `_make_layer_past_kv_arg()` respect per-model overrides so both legacy and cache-object paths work for all supported architectures.

## Single-file checkpoints

Checkpoints that ship as a single `model.safetensors` (no `model.safetensors.index.json`) are supported. The split logic detects this and loads keys from the single file; `find_or_create_local_splitted_path` ensures the file is downloaded and the split directory is created correctly.
