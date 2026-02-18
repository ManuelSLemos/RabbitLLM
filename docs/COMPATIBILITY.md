# Compatibility

## Transformers version

- **Supported**: `transformers>=4.47,<4.57` (e.g. 4.47.x through 4.56.x).
- **Recommended**: Use the latest patch (e.g. 4.56.x) for the widest model support (Qwen3, DeepSeek V3, Gemma2/3, Phi3, Llama 3.2, etc.).

In 4.50+, `GenerationMixin` may need to be imported from `transformers.generation.utils`; the codebase tries both import paths.

Upgrading to 4.47 from 4.46 brings:

- Better support for Qwen2.5, Llama 3.x, and modern configs (e.g. rope_scaling).
- Native SDPA and FlashAttention 2 integration.
- Cache utilities (`DynamicCache`) as the standard; our code uses them when available.
- In 4.47+, `GenerationMixin` remains available via `from transformers import GenerationMixin`; no code change required for the base model.

## Gated models

Some repos (e.g. Meta Llama, certain Gemma variants) are gated. Use a Hugging Face token: pass `hf_token="hf_..."` to `from_pretrained()` or set the `HF_TOKEN` environment variable. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#gated-models-hugging-face).

## Dependencies

- **accelerate** ≥ 0.26 (needed for transformers 4.46).
- **sentencepiece** (required for Baichuan tokenizer; add to project dependencies if you use Baichuan).
- **flash-attn** (optional): for `attn_implementation="flash_attention_2"`; requires Ampere+ GPU and fp16/bf16.

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
- When using layer-streaming, the decoder may not fill the `DynamicCache` we pass; the engine then returns `past_key_values=None` and logs a warning once. Generation still works but each step re-runs the full forward (no incremental decoding), so throughput is lower. This is a known limitation with Qwen2 in 4.47+ under streaming; using a non-streaming run or a future fix will restore KV cache.

### Cache compatibility note

Models that use a **non-standard** cache keyword (QWen v1: `layer_past`, ChatGLM: `kv_cache`) are compatible only as long as the code path uses the **legacy tuple** format. If `_uses_cache_objects` is True for all models (DynamicCache), those two would receive `past_key_value` instead of their expected key and generation could break. Future work may make `_make_layer_past_kv_arg()` respect per-model overrides so both legacy and cache-object paths work for all supported architectures.

## Single-file checkpoints

Checkpoints that ship as a single `model.safetensors` (no `model.safetensors.index.json`) are supported. The split logic detects this and loads keys from the single file; `find_or_create_local_splitted_path` ensures the file is downloaded and the split directory is created correctly.
