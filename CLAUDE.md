# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AirLLM enables running large language models (70B+ parameters) on GPUs with as little as 4GB VRAM by streaming model layers one at a time through GPU memory, avoiding the need for quantization, distillation, or pruning.

## Build & Install

```bash
# Install from source (editable mode)
cd air_llm
pip install -e .

# Install from PyPI
pip install airllm
```

Note: setup.py includes a PostInstallCommand that auto-upgrades transformers to avoid rope_scaling compatibility issues.

## Running Tests

Tests use unittest and live in `air_llm/tests/`. The compression test requires a CUDA GPU.

```bash
# Run all tests
cd air_llm && python -m pytest tests/

# Run a single test module
python -m unittest air_llm.tests.test_automodel
python -m unittest air_llm.tests.test_compression
```

## Architecture

### Core Design: Layer-Streaming Inference

The central idea is processing models **one layer at a time** to fit within constrained GPU memory:

1. **Splitting phase**: `utils.split_and_save_layers()` takes a HuggingFace sharded checkpoint and saves each transformer layer as an individual safetensors file. Optional 4-bit/8-bit block-wise compression via bitsandbytes.
2. **Inference phase**: `AirLLMBaseModel.forward()` creates an empty model skeleton, then for each layer: loads weights from disk → (optionally decompresses) → moves to GPU → runs forward pass → frees GPU memory. A background thread prefetches the next layer to overlap I/O with compute.

### Key Classes

**`AirLLMBaseModel`** (`air_llm/airllm/airllm_base.py`) — Base class implementing the layer-streaming forward pass, inheriting `GenerationMixin` for text generation. All model variants extend this.

**`AutoModel`** (`air_llm/airllm/auto_model.py`) — Factory that reads HuggingFace config `architectures` to select the right model class. On macOS, always returns the MLX implementation.

**Model-specific subclasses** — Each overrides `set_layer_names_dict()` to map architecture-specific layer names and may customize positional embeddings or attention masks:
- `AirLLMLlama2` (Llama2/3/3.1) — `airllm.py`
- `AirLLMQWen` / `AirLLMQWen2` — `airllm_qwen.py` / `airllm_qwen2.py`
- `AirLLMChatGLM` — `airllm_chatglm.py`
- `AirLLMBaichuan` — `airllm_baichuan.py`
- `AirLLMInternLM` — `airllm_internlm.py`
- `AirLLMMistral` / `AirLLMMixtral` — `airllm_mistral.py` / `airllm_mixtral.py`
- `AirLLMLlamaMlx` — `airllm_llama_mlx.py` (Apple Silicon via MLX framework)

### Platform Branching

The package uses platform detection (`sys.platform == "darwin"`) in both `__init__.py` and `auto_model.py` to switch between PyTorch (Linux/Windows) and MLX (macOS) implementations. These are completely separate code paths.

### Persistence Layer

`air_llm/airllm/persist/` contains `ModelPersister` (abstract), `SafetensorModelPersister` (default), and `MlxModelPersister` (macOS) for reading/writing split layer files.

### Other Directories

- **`training/`** — QLoRA fine-tuning scripts (`qlora.py`)
- **`rlhf/`** — DPO training implementation (`qlora_dpo.py`)
- **`eval/`** — Evaluation scripts
- **`examples/`** — Jupyter notebook examples

## Adding a New Model

1. Create a new file `air_llm/airllm/airllm_<model>.py`
2. Subclass `AirLLMBaseModel` and override `set_layer_names_dict()` with the model's layer naming scheme
3. Add the architecture detection branch in `AutoModel.get_module_class()`
4. Export the class in `air_llm/airllm/__init__.py`
