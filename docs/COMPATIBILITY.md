# Compatibility

## Transformers version

- **Supported**: `transformers>=5.0,<5.1` (5.0.x only).
- **5.1.x / 5.2.x**: Not supported for Qwen2/Qwen2.5 layer-streaming due to a RoPE/head_dim mismatch in the attention layer (`apply_rotary_pos_emb` 14 vs 64). A fix is pending; use 5.0.x for Qwen2/Qwen2.5 until then.

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
- **flash-attn** (optional): for `attn_implementation="flash_attention_2"`; requires Ampere+ GPU and fp16/bf16.

## Requirements for transformers 5.x

This project targets `transformers>=5.0`. Ensure: Python 3.10+, PyTorch 2.0+ (2.4+ recommended), **accelerate** ≥ 1.1.0, **peft** ≥ 0.18.0 (if using PEFT), **bitsandbytes** ≥ 0.46.1 (if using quantization). See [TRANSFORMERS_UPGRADE_PLAN.md](TRANSFORMERS_UPGRADE_PLAN.md).

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
- When using layer-streaming, the decoder may not fill the `DynamicCache` we pass; the engine then returns `past_key_values=None` and logs a warning once. Generation still works but each step re-runs the full forward (no incremental decoding), so throughput is lower. This is a known limitation with Qwen2 in 4.47+ under streaming; using a non-streaming run or a future fix will restore KV cache.

### Fixes that apply to all models (base engine)

- **`config.head_dim`**: If the config has `hidden_size` and `num_attention_heads`, the engine sets `config.head_dim = hidden_size // num_attention_heads` before creating the model so RoPE and attention use the correct dimension (avoids wrong values from hub or older configs).
- **`get_sequence_len(seq)`**: Handles both 3D tensors `(batch, seq_len, hidden)` and 2D `(seq_len, hidden)` so the sequence length used for position embeddings is correct (avoids using `hidden_size` as length).
- **`_reset_model()`**: After re-creating the model skeleton, the engine calls `set_layers_from_layer_names()` so `self.layers` always refers to the current model’s layers.
- **`_fix_attention_head_dim()`**: For any model with `hidden_size` and `num_attention_heads`, the engine sets each decoder layer’s `self_attn.head_dim` to the canonical value. This is required when the config or transformers creates attention with a wrong `head_dim` (e.g. Qwen2.5-0.5B).

### Cache compatibility note

Models that use a **non-standard** cache keyword (QWen v1: `layer_past`, ChatGLM: `kv_cache`) are compatible only as long as the code path uses the **legacy tuple** format. If `_uses_cache_objects` is True for all models (DynamicCache), those two would receive `past_key_value` instead of their expected key and generation could break. Future work may make `_make_layer_past_kv_arg()` respect per-model overrides so both legacy and cache-object paths work for all supported architectures.

## Single-file checkpoints

Checkpoints that ship as a single `model.safetensors` (no `model.safetensors.index.json`) are supported. The split logic detects this and loads keys from the single file; `find_or_create_local_splitted_path` ensures the file is downloaded and the split directory is created correctly.
