# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RabbitLLM enables running large language models (70B+ parameters) on GPUs with as little as 4GB VRAM by streaming model layers one at a time through GPU memory, avoiding the need for quantization, distillation, or pruning.

## Build & Install

The project uses **uv** as the package manager. Makefile targets (`make install`, `make test`, etc.) run via `uv sync` and `uv run`.

```bash
# Install from source (editable mode, with dev tools)
cd airllm  # or project root
uv sync --extra dev
# or: make install

# Install from PyPI
pip install rabbitllm
```

Note: The project supports `transformers>=4.47,<4.49`. For Qwen2/Qwen2.5 with 4.47+, see docs for position_embeddings and KV cache behavior.

## Running Tests

Tests use unittest and live in `tests/`. The compression test requires a CUDA GPU.

```bash
# Run all tests
uv run pytest tests/
# or: make test

# Run a single test module
uv run python -m unittest tests.test_model_registry
uv run python -m unittest tests.test_compression
```

## Architecture

### Core Design: Layer-Streaming Inference

The central idea is processing models **one layer at a time** to fit within constrained GPU memory:

1. **Splitting phase**: `utils.split_and_save_layers()` takes a HuggingFace sharded checkpoint and saves each transformer layer as an individual safetensors file. Optional 4-bit/8-bit block-wise compression via bitsandbytes.
2. **Inference phase**: `RabbitLLMBaseModel.forward()` creates an empty model skeleton, then for each layer: loads weights from disk → (optionally decompresses) → moves to GPU → runs forward pass → frees GPU memory. A background thread prefetches the next layer to overlap I/O with compute.

### Key Classes

**`RabbitLLMBaseModel`** (`src/rabbitllm/engine/base.py`) — Base class implementing the layer-streaming forward pass, inheriting `GenerationMixin` for text generation. All model variants extend this. The engine package is split into focused modules used by the base class:
- **`engine/attention.py`** — Attention implementation resolution and meta model creation (`resolve_attn_implementation`, `create_model_from_config`, `ATTN_FALLBACK_ORDER`).
- **`engine/model_init.py`** — Model skeleton creation with attention fallback (`create_model_with_attn_fallback`).
- **`engine/layer_loading.py`** — Loading layer state dicts from disk and moving them to device (`load_layer_to_cpu`, `move_layer_to_device`).
- **`engine/forward_utils.py`** — Helpers for the forward pass (attention mask/position_ids construction, KV extraction from layer outputs).

**`AutoModel`** (`src/rabbitllm/models/registry.py`) — Factory that reads HuggingFace config `architectures` to select the right model class. On macOS, always returns the MLX implementation.

**Model-specific subclasses** — Each overrides `set_layer_names_dict()` to map architecture-specific layer names and may customize positional embeddings or attention masks:
- `RabbitLLMLlama2` (Llama2/3/3.1) — `models/llama.py`
- `RabbitLLMQWen` / `RabbitLLMQWen2` — `models/qwen.py` / `models/qwen2.py`
- `RabbitLLMChatGLM` — `models/chatglm.py`
- `RabbitLLMBaichuan` — `models/baichuan.py`
- `RabbitLLMInternLM` — `models/internlm.py`
- `RabbitLLMMistral` / `RabbitLLMMixtral` — `models/mistral.py` / `models/mixtral.py`
- `RabbitLLMLlamaMlx` — `engine/mlx_engine.py` (Apple Silicon via MLX framework)

### Platform Branching

The package uses platform detection (`sys.platform == "darwin"`) in both `__init__.py` and `auto_model.py` to switch between PyTorch (Linux/Windows) and MLX (macOS) implementations. These are completely separate code paths.

### Persistence Layer

`src/rabbitllm/persist/` contains `ModelPersister` (abstract), `SafetensorModelPersister` (default), and `MlxModelPersister` (macOS) for reading/writing split layer files.

### Local model cache

To avoid re-downloading models and keep them out of git: the repo has `models/` and `.models/` in `.gitignore`. Users can set `HF_HOME="$(pwd)/models"` so Hugging Face and RabbitLLM use that directory; see [README.md](README.md#local-model-cache).

### Device and Docker

- If CUDA is requested but unavailable or fails to init, the engine falls back to `device="cpu"` and logs a warning (see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)).
- **Makefile**: `make bash` — CPU-only container (plain Python image). `make bash-gpu` — GPU-capable container (PyTorch + CUDA image); use when you need GPU inside Docker.

### Other Directories

- **`training/`** — QLoRA fine-tuning scripts (`qlora.py`)
- **`rlhf/`** — DPO training implementation (`qlora_dpo.py`)
- **`eval/`** — Evaluation scripts
- **`examples/`** — Jupyter notebook examples
- **`docs/`** — Architecture, compatibility, and troubleshooting notes (see below).

## Technical Notes (see docs/)

Critical design decisions are documented in `docs/`:

- **`docs/ARCHITECTURE.md`** — Relationship with HuggingFace (we use HF for model definitions, only customize loading and forward loop). Why we **do not** call `tie_weights()` and how tied `lm_head` is handled. KV cache (DynamicCache) and attention implementations (eager float mask, SDPA with mask=None, flash).
- **`docs/COMPATIBILITY.md`** — Transformers version (4.47+). Model compatibility matrix. Qwen2 4.47+ (position_embeddings, KV cache fallback). Single-file checkpoints.
- **`docs/TROUBLESHOOTING.md`** — Zero logits (tied weights), eager mask, KV cache empty list, SDPA/cache alignment, dtype, single-file splits. CPU vs CUDA: when CPU is faster (layer-streaming transfer overhead). How to debug forward vs HF. Benchmark: `scripts/benchmark_cpu_vs_cuda.py`.

When changing loading, cache, or attention logic, check these docs to avoid regressions (e.g. QWen v1 / ChatGLM use custom cache kwargs).

## Adding a New Model

1. Create a new file `src/rabbitllm/rabbitllm_<model>.py`
2. Subclass `RabbitLLMBaseModel` and override `set_layer_names_dict()` with the model's layer naming scheme
3. Add the architecture detection branch in `AutoModel.get_module_class()`
4. Export the class in `src/rabbitllm/__init__.py`
