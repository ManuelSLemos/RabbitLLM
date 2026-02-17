# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RabbitLLM enables running large language models (70B+ parameters) on GPUs with as little as 4GB VRAM by streaming model layers one at a time through GPU memory, avoiding the need for quantization, distillation, or pruning.

## Build & Install

```bash
# Install from source (editable mode)
cd airllm  # or project root
pip install -e .

# Install from PyPI
pip install rabbitllm
```

Note: setup.py includes a PostInstallCommand that auto-upgrades transformers to avoid rope_scaling compatibility issues.

## Running Tests

Tests use unittest and live in `src/tests/`. The compression test requires a CUDA GPU.

```bash
# Run all tests
python -m pytest src/tests/

# Run a single test module
python -m unittest src.tests.test_automodel
python -m unittest src.tests.test_compression
```

## Architecture

### Core Design: Layer-Streaming Inference

The central idea is processing models **one layer at a time** to fit within constrained GPU memory:

1. **Splitting phase**: `utils.split_and_save_layers()` takes a HuggingFace sharded checkpoint and saves each transformer layer as an individual safetensors file. Optional 4-bit/8-bit block-wise compression via bitsandbytes.
2. **Inference phase**: `RabbitLLMBaseModel.forward()` creates an empty model skeleton, then for each layer: loads weights from disk → (optionally decompresses) → moves to GPU → runs forward pass → frees GPU memory. A background thread prefetches the next layer to overlap I/O with compute.

### Key Classes

**`RabbitLLMBaseModel`** (`src/rabbitllm/rabbitllm_base.py`) — Base class implementing the layer-streaming forward pass, inheriting `GenerationMixin` for text generation. All model variants extend this.

**`AutoModel`** (`src/rabbitllm/auto_model.py`) — Factory that reads HuggingFace config `architectures` to select the right model class. On macOS, always returns the MLX implementation.

**Model-specific subclasses** — Each overrides `set_layer_names_dict()` to map architecture-specific layer names and may customize positional embeddings or attention masks:
- `RabbitLLMLlama2` (Llama2/3/3.1) — `rabbitllm.py`
- `RabbitLLMQWen` / `RabbitLLMQWen2` — `rabbitllm_qwen.py` / `rabbitllm_qwen2.py`
- `RabbitLLMChatGLM` — `rabbitllm_chatglm.py`
- `RabbitLLMBaichuan` — `rabbitllm_baichuan.py`
- `RabbitLLMInternLM` — `rabbitllm_internlm.py`
- `RabbitLLMMistral` / `RabbitLLMMixtral` — `rabbitllm_mistral.py` / `rabbitllm_mixtral.py`
- `RabbitLLMLlamaMlx` — `rabbitllm_llama_mlx.py` (Apple Silicon via MLX framework)

### Platform Branching

The package uses platform detection (`sys.platform == "darwin"`) in both `__init__.py` and `auto_model.py` to switch between PyTorch (Linux/Windows) and MLX (macOS) implementations. These are completely separate code paths.

### Persistence Layer

`src/rabbitllm/persist/` contains `ModelPersister` (abstract), `SafetensorModelPersister` (default), and `MlxModelPersister` (macOS) for reading/writing split layer files.

### Other Directories

- **`training/`** — QLoRA fine-tuning scripts (`qlora.py`)
- **`rlhf/`** — DPO training implementation (`qlora_dpo.py`)
- **`eval/`** — Evaluation scripts
- **`examples/`** — Jupyter notebook examples
- **`docs/`** — Architecture, compatibility, and troubleshooting notes (see below).

## Technical Notes (see docs/)

Critical design decisions are documented in `docs/`:

- **`docs/ARCHITECTURE.md`** — Relationship with HuggingFace (we use HF for model definitions, only customize loading and forward loop). Why we **do not** call `tie_weights()` and how tied `lm_head` is handled. KV cache (DynamicCache) and attention implementations (eager float mask, SDPA with mask=None, flash).
- **`docs/COMPATIBILITY.md`** — Transformers version (4.44–4.46; avoid 4.47+). Model compatibility matrix. Single-file checkpoints.
- **`docs/TROUBLESHOOTING.md`** — Zero logits (tied weights), eager mask, KV cache empty list, SDPA/cache alignment, dtype, single-file splits. How to debug forward vs HF.

When changing loading, cache, or attention logic, check these docs to avoid regressions (e.g. QWen v1 / ChatGLM use custom cache kwargs).

## Adding a New Model

1. Create a new file `src/rabbitllm/rabbitllm_<model>.py`
2. Subclass `RabbitLLMBaseModel` and override `set_layer_names_dict()` with the model's layer naming scheme
3. Add the architecture detection branch in `AutoModel.get_module_class()`
4. Export the class in `src/rabbitllm/__init__.py`
