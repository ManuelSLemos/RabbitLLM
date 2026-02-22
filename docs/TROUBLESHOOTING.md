# Troubleshooting

Common issues and how they were addressed in the codebase.

## All logits are zero → garbage or repetitive output

**Symptom**: `model.generate()` runs without errors but output is nonsense or all the same token. Debug shows logits with `min=0, max=0` (all zero).

**Cause**: The LM head weights were never loaded onto the device. This happens when:

1. **Tied weights**: The model has `tie_word_embeddings=True`. The checkpoint only stores `model.embed_tokens.weight`; there is no separate `lm_head.weight`. We used to call `tie_weights()` so that `lm_head.weight` pointed to `embed_tokens.weight`. In layer-streaming we load the embedding layer first, which **replaces** that parameter with a new tensor on GPU and **breaks** the tie. So `lm_head.weight` stayed on the meta device.
2. The split file for `lm_head` is therefore empty (no key for `lm_head.weight`). Loading that layer does nothing, and the head remains on meta.

**Fix** (in code):

- Do **not** call `self.model.tie_weights()` in `init_model()`. See [ARCHITECTURE.md](ARCHITECTURE.md#tied-weightstie_word_embeddings).
- In `forward()`, when processing the `lm_head` layer, if the loaded state_dict for that layer is empty and `config.tie_word_embeddings` is True, load the **embedding** layer again and set `lm_head.weight` from that tensor.

## Eager attention: wrong or noisy output / alignment error

**Symptom**: With `attn_implementation="eager"`, output is wrong, or you see `RuntimeError: p.attn_bias_ptr is not correctly aligned`.

**Causes and fixes**:

1. **Boolean mask**: Eager attention in HuggingFace uses **additive** masking: `attn_weights + mask`, where 0.0 = attend and a large negative value = mask out. If we pass a **boolean** mask (True/False), it is interpreted as 1/0 and does not mask; the model can attend to future positions → wrong logits.  
   **Fix**: Build a float causal mask with `torch.finfo(dtype).min` for masked positions and 0 for the lower triangle (causal). Use the model’s `running_dtype`.

2. **Model created without `attn_implementation`**: In transformers 4.44+, the default is SDPA. If we do not pass `attn_implementation="eager"` when calling `from_config()`, the created model uses SDPA internally. Our code then builds an eager-style mask and may pass it to SDPA, which can trigger alignment or numerical issues.  
   **Fix**: Always pass `attn_implementation` explicitly when creating the model (including in fallbacks and when using `"eager"`).

## KV cache: `ValueError: torch.cat(): expected a non-empty list of Tensors`

**Symptom**: During generation, after the first forward pass, the next step fails when building the returned `past_key_values` with `torch.cat(kv_cache_list[i][0], 0)`.

**Cause**: With transformers ≥ 4.36, attention layers expect a **Cache** object (e.g. `DynamicCache`) for `past_key_value`. If we pass `None`, they return `None` for the present cache, so we never append anything to `kv_cache_list[i]` and the list stays empty.

**Fix**: When `transformers.cache_utils` is available, always pass a `DynamicCache` to each layer when `use_cache=True` (and update it from the layer output). The base property `_uses_cache_objects` is True in that case; we use it in `_make_layer_past_kv_arg()` to build the cache argument. Do not restrict this to SDPA/Flash only: eager in 4.44+ also expects a Cache when caching.

## KV cache: `IndexError` or wrong behavior with DynamicCache

**Symptom**: `DynamicCache.update()` or indexing fails (e.g. list index out of range).

**Cause**: In layer-streaming we run **one** layer at a time. Each layer’s attention module has a fixed `layer_idx` (e.g. 15). When we pass a fresh `DynamicCache` that only has one entry (for the current layer), indexing by `layer_idx` can be wrong or out of range.

**Fix**: Use a context manager that temporarily sets the attention module’s `layer_idx` to `0` for the duration of the layer call, then restores it. So every streamed layer updates the cache at index 0. See `_layer_idx_as_zero()` in the base class.

## KV cache not filled / no incremental decoding

**Symptom**: A warning appears once during generation: *"KV cache was not filled by decoder layers; returning past_key_values=None. Generation will work but each step re-runs the full forward (no incremental decoding)."*

**Cause**: In **transformers 4.47+**, decoder layers (e.g. Qwen2/Qwen2.5) often **do not return** the cache in the layer output tuple; they update the `DynamicCache` **in-place**. The engine passes an empty cache, calls the layer, and if the output has no KV it uses a **fallback**: read from the same cache object (new API: `.layers[0].keys` / `.layers[0].values`; legacy: `.key_cache[-1]` / `.value_cache[-1]`) to fill `past_key_values`. You must use `return_dict=True` so the forward output includes `past_key_values`. If the cache is still not filled, each step re-runs the full forward and throughput drops.

**What to do**: Ensure `return_dict=True` when using `use_cache=True`. If the warning persists, the Cache API or layer kwargs may have changed. See [COMPATIBILITY.md](COMPATIBILITY.md) (Qwen2, KV cache).

**Nota para versiones superiores (5.1+)**: Al subir transformers, **revalidar** el flujo de KV cache en layer-streaming: que el fallback siga leyendo correctamente del `DynamicCache` (atributos `.layers` vs `.key_cache`/`.value_cache`), que `cache_position` y `position_embeddings` se pasen con las formas esperadas en el primer e incremental forward, y que el mismo objeto cache que se pasa a la capa sea el que se inspecciona después. Solución pendiente si la API de Cache cambia en 5.1+.

## Error 14 vs 64 en `apply_rotary_pos_emb` (transformers 5.1+)

**Síntoma**: Al usar **versiones superiores de transformers** (5.1.x, 5.2.x o posteriores) con Qwen2/Qwen2.5 y **KV cache** (forward incremental), falla con:

```text
RuntimeError: The size of tensor a (14) must match the size of tensor b (64) at non-singleton dimension 3
```

(en `transformers/models/qwen2/modeling_qwen2.py`, en `apply_rotary_pos_emb`: `q_embed = (q * cos) + ...`).

**Causa**: En esas versiones, la atención puede crearse con un `head_dim` incorrecto (p. ej. 14 en lugar de `hidden_size // num_attention_heads` = 64). Los cos/sin de RoPE se calculan con 64 y el tensor `q` sale con última dimensión 14, de ahí el mismatch.

**Qué hacer**: Quedarse en **transformers 5.0.x** para Qwen2/Qwen2.5 hasta tener una solución. Ver [COMPATIBILITY.md](COMPATIBILITY.md) (versiones superiores). **Solución pendiente** para 5.1+.

## CUDA error: device-side assert (inf/nan in logits or sampling)

**Symptom**: `CUDA error: device-side assert triggered`, often in sampling, with a message about probability tensor containing inf, nan, or values &lt; 0.

**Causes and fixes**:

1. **Attention mask with SDPA**: Passing a manual boolean or badly scaled mask to SDPA can cause numerical issues. SDPA handles causality natively when `attention_mask=None`.  
   **Fix**: For `attn_implementation="sdpa"`, pass `attention_mask=None` in the layer kwargs (and do not build a 4D causal mask for that path).

2. **Dtype mismatch**: Some models (e.g. Qwen2.5) are trained in bfloat16. Forcing float16 can cause overflow in attention or logits.  
   **Fix**: Auto-detect `torch_dtype` from `config.torch_dtype` when the user does not pass `dtype`, and fall back to float16 only if the config does not specify a dtype.

## Single-file model: `AssertionError: model.safetensors.index.json should exist`

**Symptom**: Splitting or loading fails because the code assumed a sharded checkpoint with an index file.

**Cause**: Some checkpoints (e.g. small models) are distributed as a single `model.safetensors` with no index.

**Fix**: In `utils.split_and_save_layers()` and `find_or_create_local_splitted_path()`, detect the single-file case (no index, or only `model.safetensors`), and use `safetensors.safe_open` / the list of keys to build the layer list and split or load accordingly.

## Async CPU→GPU transfer

**Context**: To overlap CPU→GPU copy of the next layer with the current layer’s forward pass, the engine can use a separate CUDA stream for the copy and then assign parameters on the default stream after sync. This two-phase flow is implemented in `move_layer_to_device_async` (copy only on the transfer stream) and `set_layer_params_from_tensors` (assign on the default stream after `transfer_stream.synchronize()` and `wait_stream()`).

**Symptom**: With async transfer enabled, you may see `RuntimeError: Tensor on device cuda:0 is not on the expected device meta!` in a decoder layer (e.g. in `input_layernorm`).

**Cause**: Parameters assigned via `set_module_tensor_to_device` (or direct setattr) with tensors that were created on another stream can still be treated as meta when the module runs on the default stream in some PyTorch/accelerate environments.

**How it works (implemented and enabled by default)**: The copy of layer i+1 starts on the transfer stream **before** the forward of layer i, so the two overlap. After the forward of layer i, the engine syncs the transfer stream and assigns the parameters of layer i+1 using in-place `_parameters` dict update (avoids replacing the `Parameter` object). The sync `set_module_tensor_to_device` path (used for layer 0 and as fallback) remains unchanged.

**To disable async**: In `src/rabbitllm/engine/base.py`, in `_run_layer_streaming_loop`, set `_try_async_transfer = False`. The sync prefetch path (CPU background load) will still be used.

## Speeding up inference (responses in seconds)

To get the lowest latency per token:

1. **Attention**: The fastest option depends on GPU and model size. Try `"eager"`, `"sdpa"`, or default `"auto"` (SDPA or Flash if available); on small models or some GPUs, `"eager"` can be faster. For large models on Ampere+ GPUs, `"flash_attention_2"` or `"auto"` often wins.

2. **Prefetch**: Prefetching is on by default and overlaps loading the next layer from disk with the current layer’s forward pass. Do not pass `prefetching=False` when loading the model.

3. **Compression trade-off**: If you use `compression='4bit'` or `'8bit'`, prefetching is **disabled** in the code (see `rabbitllm_base.py`). For lowest latency, try **without** compression first; prefetch often outweighs the benefit of smaller layer files. Use compression when disk I/O is the clear bottleneck and you have measured it via `profiling_mode=True`.

4. **Disk and model size**: Keep the split model on a fast **local SSD** (Hugging Face cache or `layer_shards_saving_path`). Use smaller models (e.g. 0.5B–7B) for “response in seconds”; 70B will always be slower due to layer count.

5. **Attention mask when pad_token = eos_token**: If the tokenizer has `pad_token_id == eos_token_id` (common for Qwen and other decoder-only models), pass an explicit `attention_mask` to `model.generate()` to avoid the Transformers warning and ensure reliable behavior. The example scripts in `scripts/` do this (they request the mask from the tokenizer and pass it to `generate()`; if the tokenizer does not return one, use a mask of ones with the same shape as `input_ids`).

### Finding the bottleneck with the profiler

Load the model with `profiling_mode=True`:

```python
model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", profiling_mode=True)
```

After a `generate()` call, the profiler prints total time per category:

- **load_safe_tensor** — time loading layer data from disk. If this dominates, use an SSD or consider compression to reduce I/O size.
- **create_layer_from_state_dict** — time copying layer weights from CPU to VRAM. If this dominates, ensure prefetching is on (it overlaps load-to-CPU with compute). Async CPU→GPU transfer (overlap copy with forward) is implemented but disabled by default; see "Async CPU→GPU transfer" above.
- **forward_per_layer** — time spent in the actual forward pass per layer. If this dominates, use SDPA or Flash (see point 1 above).
- **load_safe_tensor_cpu_wait** — time waiting for the prefetched layer to be ready (should be low when prefetch overlaps well with compute).

Use these to decide whether to optimize disk I/O, CPU→VRAM copy, or attention implementation.

To profile with a single command (e.g. for 70B), run:

```bash
uv run python scripts/profile_inference.py --model /path/to/70B-or-repo --max-new-tokens 20
```

For **70B/72B**, if `pin_memory_to_trigger_load` and `load_safe_tensor_cpu_wait` dominate (~180–200 s per step), disable pin_memory: pass `prefetch_pin_memory=False` when loading the model, or use the script flag `--no-prefetch-pin-memory`. Re-profile to confirm total time drops (often from ~210 s to ~30–40 s per step).

### Recommended settings for 70B and large models

For lowest latency when using 70B (or other large) models:

- **Prefetch**: Leave default `prefetching=True`; do not use `compression` if you want speed (compression disables prefetch).
- **Attention**: Use `attn_implementation="auto"` or `"flash_attention_2"` on Ampere+ GPUs (e.g. RTX 30xx/40xx).
- **Disk**: Keep the split model on a **local SSD**; avoid network or slow drives.
- **Generation**: Always pass `use_cache=True` to `model.generate()` so each token uses incremental decoding (KV cache). Ensure the cache is filled (see "KV cache not filled" above if you see the warning).
- **Length**: Use a smaller `max_new_tokens` if you do not need long replies; time grows linearly with tokens.

When using `use_cache=True` (incremental decoding), the engine keeps the small layers (embed, norm, lm_head) on GPU across generated tokens instead of reloading them every step, reducing load/transfer for those layers on the second token onward.

## Flash Attention: CUDA device-side assert (varlen_fwd)

**Symptom**: During generation (second or later forward with KV cache), you get `RuntimeError: CUDA error: device-side assert triggered` in `_flash_attn_varlen_forward` or `flash_attn_gpu.varlen_fwd`.

**Cause**: With Flash Attention 2, transformers can use the variable-length path. Passing a full-length attention mask or omitting `cache_position` in the incremental step can lead to wrong bounds and trigger the assert.

**Fix** (in code): The engine now (1) passes **cache_position** in the incremental decoding branch (position of current tokens in the full sequence) and (2) for Flash only, **slices the attention mask** to the current context length (past + present) instead of full `max_seq_len`, so the varlen path does not see inconsistent lengths. If you still see the error on an older commit, pull the latest or backport those two changes.

For debugging you can set `CUDA_LAUNCH_BLOCKING=1` to get a more accurate stack trace.

## Flash Attention: installation fails (build from source)

**Symptom**: `uv sync --extra flash` or `pip install rabbitllm[flash]` fails with "Failed to build flash-attn" or ninja/compilation errors. The log may show "Precompiled wheel not found. Building from source...".

**Cause**: `flash-attn` has no prebuilt wheel for your exact combination of PyTorch version, CUDA version, and Python version. When pip/uv tries to build from source, it often fails (CUDA toolkit, compiler, or download 404 during build).

**What to do**:

1. **Use a prebuilt wheel** (recommended): Go to [flashattn.dev](https://flashattn.dev) or [flashattn.dev/install](https://flashattn.dev/install), select your **PyTorch version**, **CUDA version**, and **Python version**. The site gives you a `pip install https://...whl` command. Run that inside your project environment: with uv, use `uv pip install https://...whl` from the project root (no pip on the host needed); otherwise activate your venv and run `pip install https://...whl`. Then the project will detect Flash and use it with `attn_implementation="auto"` without needing the `rabbitllm[flash]` extra (the extra is only to pull in the dependency; the code works the same once `flash_attn` is importable).

   **Direct wheels (v2.8.3, Linux x86_64, CUDA 12)** from [Dao-AILab/flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3). Use the wheel that matches your PyTorch and Python; `cxx11abiTRUE` is the usual build from pip/uv.

   - **PyTorch 2.5 + Python 3.12** (recommended for this project). From the project root: `uv pip install` (no pip on the host needed). PyTorch installed via uv often uses the **cxx11abiFALSE** ABI; if you get an "undefined symbol" when importing `flash_attn`, try the other variant.
   ```bash
   uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
   ```
   If you see `undefined symbol: ... c10::Error ...` at import time, you need the **other** ABI: use `cxx11abiTRUE` if the above is FALSE, or `cxx11abiFALSE` if you had TRUE.
   - **PyTorch 2.4 + Python 3.12** (fallback if you use torch 2.4): same URL with `torch2.4` instead of `torch2.5`. For other combinations (2.6, 2.7, cp310, cp311), browse the [v2.8.3 assets](https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.3).

2. **Stay on SDPA**: If you do not install `flash-attn`, the model uses **SDPA** (PyTorch scaled dot-product attention) with `attn_implementation="auto"`. Inference works normally; you only miss the extra speed/memory benefits of Flash on Ampere+ GPUs.

3. **Build from source** (advanced): Ensure you have the CUDA toolkit, `ninja`, and a C++ compiler matching your PyTorch build; see [flash-attention](https://github.com/Dao-AILab/flash-attention) and [flashattn.dev/troubleshooting](https://flashattn.dev/troubleshooting). The `[tool.uv.extra-build-dependencies]` entry for `flash-attn` in `pyproject.toml` ensures `torch` is available during the build when using `uv sync --extra flash`.

## CPU vs CUDA: when is CPU faster?

**Symptom**: Inference with `device="cuda:0"` feels slower than with `device="cpu"` for the same model and prompt.

**Cause**: In layer-streaming, **every** forward pass (each generated token) loads all layers from disk and moves them to the device. There is no persistent GPU residency of weights. So the cost per step is:

- **Disk read** → **CPU→GPU transfer** (PCIe) → **compute**

For **small models** (e.g. 0.5B parameters) and short sequences:

1. Each layer does very little compute on the GPU, so the GPU is underutilized.
2. The time to copy each layer from CPU to GPU can be **larger** than the time to run the layer on the GPU.
3. On CPU you avoid that transfer: data stays in RAM, so you only pay disk→RAM and compute. For 0.5B, CPU compute is often fast enough that **total time is lower on CPU**.

So it is **normal** for small models (e.g. Qwen2.5-0.5B) to be faster on CPU in this architecture. CUDA tends to win for larger models (e.g. 1.5B–3B+) where the compute per layer dominates over transfer.

**What to do**:

- For **small models** and low latency: use `device="cpu"` and pass inputs on CPU (e.g. `input_ids` without `.cuda()`).
- To **compare** on your machine: run the benchmark script:
  - `uv run python scripts/benchmark_cpu_vs_cuda.py` (default: Qwen2.5-0.5B, 2 runs per device).
  - Options: `--model`, `--max-new-tokens`, `--runs`, `--cpu-only`, `--cuda-only`.
- For **larger models** or longer generations, CUDA usually becomes faster; the benchmark helps you see the crossover.

## Gated models (Hugging Face)

**Symptom**: `Cannot access gated repo` or `Access to model X is restricted. You must have access to it and be authenticated.`

**Fix**: Models like `meta-llama/Llama-3.2-1B` or `meta-llama/Llama-2-7b-hf` require acceptance of the license on the Hub and a Hugging Face token.

1. Accept the model’s license on [huggingface.co](https://huggingface.co) and create a token (Settings → Access tokens).
2. Pass the token when using RabbitLLM:
   - **Code**: `AutoModel.from_pretrained("meta-llama/Llama-3.2-1B", token="hf_...")` (or `hf_token="hf_..."` for backward compatibility)
   - **Env**: `HF_TOKEN=hf_... python your_script.py` (scripts that read `os.environ.get("HF_TOKEN")` will use it).
   - **CLI** (when available): `--token hf_...` or `HF_TOKEN=hf_... rabbit ...`.

The token is forwarded to `AutoConfig.from_pretrained(..., token=...)`, model download, and tokenizer loading. Do not commit tokens; use env vars or a secrets manager.

## Debugging forward vs HuggingFace

To check whether layer-streaming matches standard HuggingFace:

1. Run one forward pass with the **same** `input_ids` with:
   - Standard: `AutoModelForCausalLM.from_pretrained(...).to(device)` then `model(**inputs, use_cache=False)`.
   - RabbitLLM: `AutoModel.from_pretrained(...)` then `model(input_ids, use_cache=False)`.
2. Compare logits (e.g. last position): cosine similarity and max difference.
3. If RabbitLLM logits are all zero, inspect buffers (e.g. `inv_freq` for RoPE) and the **lm_head** weight device and whether the tied-weights path is applied (see [ARCHITECTURE](ARCHITECTURE.md#tied-weightstie_word_embeddings)).

A small script that does (1)–(2) and optionally (3) is useful for regression testing when changing loading or attention logic.
