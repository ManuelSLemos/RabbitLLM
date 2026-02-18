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

## Speeding up inference (responses in seconds)

To get the lowest latency per token:

1. **Attention**: The fastest option depends on GPU and model size. Try `"eager"`, `"sdpa"`, or default `"auto"` (SDPA or Flash if available); on small models or some GPUs, `"eager"` can be faster. For large models on Ampere+ GPUs, `"flash_attention_2"` or `"auto"` often wins.

2. **Prefetch**: Prefetching is on by default and overlaps loading the next layer from disk with the current layer’s forward pass. Do not pass `prefetching=False` when loading the model.

3. **Compression trade-off**: If you use `compression='4bit'` or `'8bit'`, prefetching is **disabled** in the code (see `rabbitllm_base.py`). For lowest latency, try **without** compression first; prefetch often outweighs the benefit of smaller layer files. Use compression when disk I/O is the clear bottleneck and you have measured it via `profiling_mode=True`.

4. **Disk and model size**: Keep the split model on a fast **local SSD** (Hugging Face cache or `layer_shards_saving_path`). Use smaller models (e.g. 0.5B–7B) for “response in seconds”; 70B will always be slower due to layer count.

### Finding the bottleneck with the profiler

Load the model with `profiling_mode=True`:

```python
model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", profiling_mode=True)
```

After a `generate()` call, the profiler prints total time per category:

- **load_safe_tensor** — time loading layer data from disk. If this dominates, use an SSD or consider compression to reduce I/O size.
- **create_layer_from_state_dict** — time copying layer weights from CPU to VRAM. If this dominates, consider the Fase 3 optimization (second CUDA stream or non_blocking copy).
- **forward_per_layer** — time spent in the actual forward pass per layer. If this dominates, use SDPA or Flash (see point 1 above).
- **load_safe_tensor_cpu_wait** — time waiting for the prefetched layer to be ready (should be low when prefetch overlaps well with compute).

Use these to decide whether to optimize disk I/O, CPU→VRAM copy, or attention implementation.

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
   - **Code**: `AutoModel.from_pretrained("meta-llama/Llama-3.2-1B", hf_token="hf_...")`
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
