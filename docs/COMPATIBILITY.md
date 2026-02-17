# Compatibility

## Transformers version

- **Supported**: `transformers>=4.47,<4.49` (e.g. 4.47.x or 4.48.x).
- **Recommended**: Use the latest patch in that range for new model support (Qwen3, DeepSeek V3, Gemma3, etc.).

Upgrading to 4.47 from 4.46 brings:

- Better support for Qwen2.5, Llama 3.x, and modern configs (e.g. rope_scaling).
- Native SDPA and FlashAttention 2 integration.
- Cache utilities (`DynamicCache`) as the standard; our code uses them when available.
- In 4.47+, `GenerationMixin` remains available via `from transformers import GenerationMixin`; no code change required for the base model.

## Dependencies

- **accelerate** ≥ 0.26 (needed for transformers 4.46).
- **sentencepiece** (required for Baichuan tokenizer; add to project dependencies if you use Baichuan).
- **flash-attn** (optional): for `attn_implementation="flash_attention_2"`; requires Ampere+ GPU and fp16/bf16.

## Model compatibility matrix

| Model / family        | Layer-streaming | Tied lm_head handling | Cache (past_key_value) |
|-----------------------|-----------------|------------------------|-------------------------|
| **Llama2 / Llama3**   | Yes             | Yes                    | Standard                |
| **Qwen2 / Qwen2.5**   | Yes             | Yes                    | Standard                |
| **Mistral / Mixtral** | Yes             | Yes                    | Standard                |
| **InternLM**          | Yes             | Yes                    | Standard                |
| **Baichuan**          | Yes*            | Yes                    | Standard                |
| **QWen v1**           | Yes             | N/A                    | Uses `layer_past`       |
| **ChatGLM**           | Yes             | N/A                    | Uses `kv_cache`         |

\* Baichuan uses a custom tokenizer (sentencepiece); ensure the dependency is installed.

### Qwen2 / Qwen2.5 with transformers 4.47+

- Decoder layers expect **`position_embeddings`** (cos, sin tuple) from RoPE; `RabbitLLMQWen2` overrides `get_pos_emb_args()` to compute and pass them.
- When using layer-streaming, the decoder may not fill the `DynamicCache` we pass; the engine then returns `past_key_values=None` and logs a warning once. Generation still works but each step re-runs the full forward (no incremental decoding), so throughput is lower. This is a known limitation with Qwen2 in 4.47+ under streaming; using a non-streaming run or a future fix will restore KV cache.

### Cache compatibility note

Models that use a **non-standard** cache keyword (QWen v1: `layer_past`, ChatGLM: `kv_cache`) are compatible only as long as the code path uses the **legacy tuple** format. If `_uses_cache_objects` is True for all models (DynamicCache), those two would receive `past_key_value` instead of their expected key and generation could break. Future work may make `_make_layer_past_kv_arg()` respect per-model overrides so both legacy and cache-object paths work for all supported architectures.

## Single-file checkpoints

Checkpoints that ship as a single `model.safetensors` (no `model.safetensors.index.json`) are supported. The split logic detects this and loads keys from the single file; `find_or_create_local_splitted_path` ensures the file is downloaded and the split directory is created correctly.
