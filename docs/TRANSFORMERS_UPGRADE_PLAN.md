# Transformers upgrade plan

This document describes a phased plan to upgrade the `transformers` version from the current range (`>=4.47,<4.57`) to the latest stable release that adds value to the project (v5.x), with the benefits and risks of each step.

## Current status

- **Constraint in `pyproject.toml`**: `transformers>=4.47,<4.57`
- **Recommendation in COMPATIBILITY.md**: use the latest 4.56.x patch for maximum model support (Qwen3, DeepSeek V3, Gemma2/3, Phi3, Llama 3.2, etc.)
- **Latest stable version (February 2025)**: **v5.2.0** (5.x branch with weekly releases)

---

## Executive summary

| Phase | Target version | Effort | Main benefit |
|------|----------------|--------|---------------|
| 1 | 4.56.x / 4.57.x | Low | Maximum 4.x support without API changes |
| 2 | v5 preparation | Medium | Code ready for v5 (token, rope, cache) |
| 3 | 5.0.x | High | New weights API, unified tokenization, better loading |
| 4 | 5.1 / 5.2 | Low–Medium | Qwen3.5, GLM-5, Voxtral, fixes and improvements |

---

## Phase 1: Maximize 4.x (4.56 → 4.57 if available)

**Goal**: Use the latest patch of the 4.x branch within the already supported range or by raising the ceiling slightly.

### Actions

1. **Check availability of 4.57.x**  
   GitHub Releases has tags such as `v4.57.6`, `v4.57.5`, etc. If the project wants to stay on 4.x a bit longer:
   - In `pyproject.toml`, change to: `transformers>=4.47,<4.58` (or `<5.0` if preferred).
   - Run `uv sync` and the test suite.

2. **Pin recommended version in documentation**  
   In `docs/COMPATIBILITY.md`, state explicitly "recommended: 4.56.x or 4.57.x" depending on what is validated.

### Benefits

- **Model support**: Better coverage of recent models (Qwen3, DeepSeek V3, Gemma2/3, Phi3, Llama 3.2) that may depend on 4.56/4.57 configs or behavior.
- **Bug fixes and security**: Patches and possible transitive dependency updates.
- **No API changes**: Current code (GenerationMixin, `cache_utils`, configs) remains valid.

### Risks

- Very low: the project is already within `>=4.47,<4.57`; extending to 4.57.x is conservative.

---

## Phase 2: Preparation for v5 (compatible with 4.x and 5.x)

**Goal**: Introduce changes that reduce the impact of the jump to v5 while keeping compatibility with 4.x.

### Actions

1. **Authentication token**  
   - Search for `use_auth_token` and `hf_token` in code and examples.
   - Replace with `token` (v5 removes `use_auth_token` in favor of `token`).  
   - In v4, `token` is already the recommended parameter; the change is compatible with both branches.

2. **RoPE access in config**  
   - In v5, `config.rope_theta` is removed; use `config.rope_parameters` (dict, e.g. `rope_theta` inside).
   - In `src/rabbitllm/engine/mlx_engine.py` (and any other use of `rope_theta`):
     - Add a helper that reads `getattr(config, 'rope_parameters', None)` and, if present, takes `rope_theta` from there; otherwise use `getattr(config, 'rope_theta', 10000)` for 4.x.
   - This keeps the same code working on 4.x and 5.x.

3. **Default cache in `generate`**  
   - In v5, if no cache is passed, the model chooses its default cache class (not always `DynamicCache`).  
   - Ensure tests and flows that assume `DynamicCache` still pass; if the project always passes `past_key_values` or an explicit cache, impact is usually low.

4. **Baichuan tokenizer**  
   - `compat/tokenization_baichuan.py` uses `PreTrainedTokenizer` / `tokenization_utils`.  
   - In v5 the base may be `PythonBackend` or similar; check the v5 migration guide for custom tokenizers and, if needed, add a conditional import (v4 vs v5) so 4.x is not broken.

5. **Document minimum dependencies for v5**  
   - v5 requires, among others: Python 3.10+, PyTorch 2.4+ (per recent docs), `accelerate` 1.1.0+, `peft` 0.18.0+, `bitsandbytes` 0.46.1+.  
   - Update `pyproject.toml` and `docs/COMPATIBILITY.md` when the 5.x branch is decided.

### Benefits

- **Shorter migration to v5**: Fewer surprises on the day of the jump.
- **Backward compatibility**: Can stay on 4.56/4.57 until tests and deployments are ready.

### Risks

- Low if helpers (e.g. RoPE) are tested on 4.x and then on 5.x.

---

## Phase 3: Migration to Transformers 5.0.x

**Goal**: Move to `transformers>=5.0,<5.1` (or `<6.0` if 5.1/5.2 should be allowed without touching deps).

### Main changes affecting RabbitLLM

1. **GenerationMixin**  
   - The project already uses a fallback: `from transformers import GenerationMixin` or `from transformers.generation.utils import GenerationMixin`.  
   - Verify in 5.0 that the correct import is from `generation.utils` and that the other is not removed.

2. **Cache (`cache_utils`)**  
   - Still exists; confirm that `Cache` and `DynamicCache` do not change module or signature in 5.0.  
   - If v5 documents a "default cache class" per model, ensure the layer-streaming flow still receives the expected cache type.

3. **Config**  
   - `rope_theta` → `config.rope_parameters` (already prepared in Phase 2).  
   - Nested configs (e.g. Qwen-VL): access via subconfigs, not direct keys on the root config.  
   - Do not load config from URL; only from local path or Hub repo (the project already uses paths/repo).

4. **Quantization**  
   - Removal of `load_in_4bit` / `load_in_8bit`; always use `quantization_config=BitsAndBytesConfig(...)`.  
   - Update any scripts or docs that use `load_in_4bit=True` to `BitsAndBytesConfig(load_in_4bit=True)`.

5. **Tokenization**  
   - `apply_chat_template` returns `BatchEncoding` (dict with `input_ids`, `attention_mask`, etc.), not just `input_ids`.  
   - Any code expecting only `input_ids` must use `outputs["input_ids"]` (or equivalent).

6. **Attention**  
   - Removal of head masking, relative position biases in Bert-like, and head pruning; RabbitLLM does not depend on them for the current architecture.

### Benefits of 5.0

- **Weight loading (WeightConverter)**: API for checkpoint transformations (reshape, merge, split). Useful for future optimizations (quantization, parallelism) without touching internal code as much.
- **Faster loading**: Device loading improvements (up to ~6x in tensor parallel scenarios) and "meta device" logic.
- **Unified tokenization**: Single backend per model (TokenizersBackend / SentencePieceBackend), less duplication and fewer bugs between "slow" and "fast".
- **Empty tokenizers**: Ability to instantiate "blank" tokenizers and train them; useful for experiments or tokenizer fine-tuning.
- **MoE**: Performance improvements for MoE models (Mixtral, etc.) with grouped implementations and `batched_mm`.
- **Dependency cleanup**: Removal of TorchScript and torch.fx; focus on dynamo/export.

### Risks

- **Breakage in 5.0**: Imports, config (rope, nested), tokenizer (Baichuan), quantization, and `apply_chat_template` may need concrete adjustments.
- **Mitigation**: Phase 2 + dedicated branch + CI tests with `transformers==5.0.x`.

---

## Phase 4: Update to 5.1.x and 5.2.x

**Goal**: Move to the latest 5.x to benefit from new models and fixes.

### Actions

1. **Update constraint**  
   - For example: `transformers>=5.0,<5.3` or `>=5.2,<6.0`, depending on project versioning policy.

2. **Review release notes**  
   - **5.1**: EXAONE-MoE, PP-DocLayoutV3, Youtu-LLM, GLM-OCR; generation cache changes (sliding window), T5Gemma2, DETR, etc.  
   - **5.2**: VoxtralRealtime, GLM-5 (GlmMoeDsa), Qwen3.5 and Qwen3.5 MoE, VibeVoice; new attention mask interface; ModernBERT changes.

3. **Watch for breaking changes**  
   - 5.2: "New attn mask interface everywhere" and ModernBERT changes.  
   - If RabbitLLM uses custom attention masks or ModernBERT-like models, run specific tests.

### Benefits by version

- **5.1**  
  - New models (EXAONE-MoE, Youtu-LLM, GLM-OCR, etc.).  
  - Generation cache fixed for sliding window.  
  - Trainer improvements, vLLM compat, RoPE, FP8/DeepSpeed, etc.

- **5.2**  
  - **Qwen3.5 and Qwen3.5 MoE**: recent vision-language and MoE models.  
  - **GLM-5 (GlmMoeDsa)**: support for models with DeepSeek Sparse Attention.  
  - **VoxtralRealtime**: real-time ASR (if the project adds audio).  
  - Bug fixes (MoE, cache, compilation, etc.).

### Risks

- Attention interface changes (5.2) may affect code that builds masks by hand; review `forward_utils` and `attention.py`.

### Known issues when upgrading to 5.1+

1. **Qwen2/Qwen2.5: 14 vs 64 in RoPE**  
   With layer-streaming and KV cache (second forward with `past_key_values`) you get  
   `RuntimeError: The size of tensor a (14) must match the size of tensor b (64) at non-singleton dimension 3` in `apply_rotary_pos_emb`.  
   **Cause**: Incorrect `head_dim` in attention. **Seek a fix** when planning 5.1/5.2. See [COMPATIBILITY.md](COMPATIBILITY.md) and [TROUBLESHOOTING.md](TROUBLESHOOTING.md#error-14-vs-64-in-apply_rotary_pos_emb-transformers-51).

2. **KV cache in layer-streaming**  
   In 4.47+ decoder layers (Qwen2, etc.) do not return the cache in the tuple; they update `DynamicCache` in-place. The engine uses a fallback that reads from the cache object (`.layers[0].keys`/`.values` or legacy `.key_cache`/`.value_cache`). When upgrading to **5.1+**, verify that the Cache API has not changed and that the KV cache is still filled; if the *"KV cache was not filled"* warning reappears, re-validate fallback, `cache_position`, `position_embeddings`, and that the same cache object is used in step and incremental. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#kv-cache-not-filled--no-incremental-decoding).

---

## Flash Attention and automatic detection

- **"auto" behavior**: With `attn_implementation="auto"` (default), the engine selects Flash Attention 2 when the system is compatible (flash-attn installed, Ampere+ GPU, fp16/bf16 dtype) and a runtime check passes; otherwise it uses SDPA. No manual configuration needed on compatible machines.
- **Detection**: `is_flash_attention_available()` in `utils/platform.py` checks: flash-attn import, CUDA available, compute capability ≥ 8.0, and a minimal test with `flash_attn_func` to detect ABI/CUDA incompatibilities at runtime.
- Documentation: `docs/COMPATIBILITY.md` (section "Attention implementation (Flash Attention)").

---

## Recommended order of work

1. **Phase 1** (quick): Extend to 4.57.x in `pyproject.toml` and docs; run tests.
2. **Phase 2** (preparation): Implement RoPE helper, replace `use_auth_token` with `token`, review Baichuan and `apply_chat_template`; tests on 4.x.
3. **Phase 3** (v5 migration): Branch with `transformers>=5.0,<5.1`; apply config, quantization, and tokenization changes; CI with 5.0.x.
4. **Phase 4** (ongoing updates): Move to 5.1 then 5.2; read release notes and run tests (and benchmarks if applicable) at each step.

---

## References

- [Transformers Releases (GitHub)](https://github.com/huggingface/transformers/releases)
- [Transformers v5.0.0 release notes](https://github.com/huggingface/transformers/releases/tag/v5.0.0)
- [v5 migration guide (main)](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Transformers v5 blog](https://huggingface.co/blog/transformers-v5)
- `docs/COMPATIBILITY.md` and `docs/ARCHITECTURE.md` in this repository
