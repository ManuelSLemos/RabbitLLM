# Architecture & Design Notes

This document captures design decisions and technical details of the layer-streaming implementation.

## Relationship with HuggingFace Transformers

RabbitLLM **does not replace** HuggingFace; it **depends on it** and customizes only two aspects:

1. **Weight loading**: Instead of loading the full model into GPU with `from_pretrained()`, we use `from_config()` to create an empty skeleton (meta device), then load each layer from disk on demand during `forward()`.
2. **Forward execution**: Instead of a single model `forward()`, we run a loop: load layer → run layer → free layer.

We reuse HuggingFace for:

- Model class definitions (Llama, Qwen2, Mistral, etc.) and their layers.
- `AutoConfig`, `AutoTokenizer`, safetensors checkpoint format.
- `GenerationMixin` and the generation API (`generate()`, `prepare_inputs_for_generation()`).

So the "alternative" is a thin layer of **memory policy** (streaming) on top of standard architectures.

## Tied Weights (`tie_word_embeddings`)

Many decoder-only models share the embedding matrix and the LM head: `lm_head.weight` and `embed_tokens.weight` are the same tensor. HuggingFace saves only one copy in the checkpoint.

### Why we do not call `tie_weights()`

In `init_model()` we **must not** call `self.model.tie_weights()`. Reason:

- After `tie_weights()`, `lm_head.weight` and `embed_tokens.weight` point to the **same** meta tensor.
- During `forward()`, we load `embed_tokens` from disk; `set_module_tensor_to_device()` **replaces** that parameter with a new tensor on GPU.
- The tie is broken: `lm_head.weight` still points to the old meta tensor, which is never loaded → logits are all zeros.

So for layer-streaming, ties are harmful at init. We keep the two parameters independent.

### Handling tied checkpoints

When the checkpoint has `tie_word_embeddings=True`, the safetensors file often does **not** contain `lm_head.weight` (only `model.embed_tokens.weight`). The split for `lm_head` is then empty.

In `forward()`, when we are about to run the `lm_head` layer:

- If the loaded state_dict for that layer is empty **and** `config.tie_word_embeddings` is True, we load the **embedding** layer again and set `lm_head.weight` from that tensor.

This way models like Qwen2.5, Llama, Mistral, etc. work correctly without ever calling `tie_weights()`.

## KV Cache and Attention Implementations

### Cache format (DynamicCache vs tuples)

From transformers ≥ 4.36, attention layers expect a **Cache** object (e.g. `DynamicCache`) for `past_key_value` when `use_cache=True`. If we pass `None`, they return `None` for the cache and our list of KV tensors stays empty → `torch.cat()` on empty lists fails.

So we always pass a `DynamicCache` to each layer when `use_cache=True` (when `transformers.cache_utils` is available). The property `_uses_cache_objects` in the base class reflects this.

### Attention implementations

- **eager**: Standard HuggingFace attention. Requires a **float additive** causal mask (0.0 = attend, large negative = mask). A boolean mask does not work (it does not mask correctly in the softmax).
- **sdpa**: PyTorch scaled dot-product attention. We pass `attention_mask=None` so SDPA uses its native causal masking (`is_causal=True`); a manual mask can cause numerical issues.
- **flash_attention_2**: Requires `flash-attn`, Ampere+ GPU, and fp16/bf16. Expects a 2D mask; causality is handled internally.

We always pass `attn_implementation` explicitly when creating the model with `from_config()`. In transformers 4.44+, the default is SDPA; if we did not pass the parameter, the created model could use SDPA while our code assumed eager (e.g. building a float mask), leading to wrong behavior or alignment errors.

### Model-specific cache kwargs

Some older or custom architectures use a different keyword for the cache:

- **QWen v1**: `layer_past` (tuple).
- **ChatGLM**: `kv_cache` (tuple).

The base `_make_layer_past_kv_arg()` currently always returns `past_key_value` when using cache objects. For those models, either they must keep using the legacy tuple path (no DynamicCache), or the base logic must be extended to respect their overrides of `get_past_key_value_args()` so the correct key name is used. See [COMPATIBILITY.md](COMPATIBILITY.md).

## Layer-streaming forward loop

The logic is split across `src/rabbitllm/engine/`: `base.py` holds `RabbitLLMBaseModel` and the main `forward()` / `_run_layer_streaming_loop()`; `attention.py` and `model_init.py` handle model creation and attention fallback; `layer_loading.py` handles loading and moving layer weights; `forward_utils.py` handles attention mask/position_ids and KV extraction from layer outputs.

1. Recreate the model skeleton (`init_model()`).
2. For each layer name in order (embed → layers → norm → lm_head):
   - Load that layer’s state_dict from disk (and for tied lm_head with empty split, load embed and assign to lm_head).
   - Move parameters (and optionally buffers) to the target device.
   - Run the layer on the current batch of hidden states; append KV cache if `use_cache`.
   - Move that layer back to meta and free GPU memory.
3. Return logits (and optionally past_key_values, hidden_states, attentions).

Prefetching (when enabled) overlaps loading the next layer with the current layer’s compute.
