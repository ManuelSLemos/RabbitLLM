# Plan: Refactor & Rebrand AirLLM

## Context

AirLLM is a discontinued project with a powerful core idea: **layer-streaming inference** that allows 70B+ LLMs to run on 4GB GPUs without quantization. The codebase works but suffers from legacy packaging, zero CI/CD, no type hints, print-based "logging", minimal tests (73 LOC), and missing features that modern tools like Ollama/vLLM offer (CLI, API server, streaming, chat templates). The goal is a full refactor + rebrand into a modern, production-grade project with both CLI and server capabilities.

**Name**: TBD (using `newproject` as placeholder — will be replaced once decided).

---

## Phase 1: Restructure + Modernize Packaging

### 1.1 Flatten directory structure

Current awkward nesting `air_llm/airllm/` → clean `src/` layout:

```
src/newproject/
  __init__.py              # Clean public API exports
  _version.py              # "3.0.0a1"
  engine/
    __init__.py
    base.py                ← air_llm/airllm/airllm_base.py
    mlx_engine.py          ← air_llm/airllm/airllm_llama_mlx.py
  models/
    __init__.py
    registry.py            ← air_llm/airllm/auto_model.py
    llama.py               ← air_llm/airllm/airllm.py
    qwen.py                ← air_llm/airllm/airllm_qwen.py
    qwen2.py               ← air_llm/airllm/airllm_qwen2.py
    chatglm.py             ← air_llm/airllm/airllm_chatglm.py
    baichuan.py            ← air_llm/airllm/airllm_baichuan.py
    internlm.py            ← air_llm/airllm/airllm_internlm.py
    mistral.py             ← air_llm/airllm/airllm_mistral.py
    mixtral.py             ← air_llm/airllm/airllm_mixtral.py
  persist/
    __init__.py
    base.py                ← air_llm/airllm/persist/model_persister.py
    safetensor.py          ← air_llm/airllm/persist/safetensor_model_persister.py
    mlx.py                 ← air_llm/airllm/persist/mlx_model_persister.py
  utils/
    __init__.py
    memory.py              ← clean_memory() etc. from utils.py
    compression.py         ← compress/uncompress from utils.py
    splitting.py           ← split_and_save_layers() etc. from utils.py
    platform.py            ← NEW: single source of truth for platform detection
  compat/
    __init__.py
    tokenization_baichuan.py ← air_llm/airllm/tokenization_baichuan.py
  profiler.py              ← air_llm/airllm/profiler.py
tests/
  conftest.py
  test_model_registry.py
  test_compression.py
```

### 1.2 Delete legacy directories

Remove entirely: `training/`, `rlhf/`, `anima_100k/`, `eval/`, `scripts/`, `data/`, `air_llm/` (after moving core files), `requirements.txt`, `README_ja.md`

### 1.3 Create `pyproject.toml` (replace `setup.py`)

- Build system: `hatchling`
- Python: `>=3.10`
- Dependencies with proper version ranges (no more git deps):
  - `torch>=2.0`, `transformers>=4.36`, `accelerate>=0.25`, `safetensors>=0.4`, `huggingface-hub>=0.20`, `tqdm`, `scipy`
- Optional extras: `[mlx]`, `[compression]`, `[server]`, `[cli]`, `[dev]`
- Entry point: `newproject = "newproject.cli:app"`
- Tool config: ruff (lint+format), pytest, mypy
- **Remove the `PostInstallCommand` hack** — `transformers>=4.36` eliminates the rope_scaling issue

### 1.4 Create `Makefile`

Commands: `install`, `dev`, `lint`, `format`, `test`, `test-cov`, `typecheck`, `clean`

### 1.5 Set up GitHub Actions CI

`.github/workflows/ci.yml`: lint (ruff) + test (pytest, Python 3.10/3.11/3.12, skip CUDA tests)

### 1.6 Update `.gitignore`

Comprehensive Python gitignore (venv, coverage, caches, etc.)

### 1.7 Update all imports

Rewrite every import across the codebase to use the new package paths.

### Verification

- `pip install -e ".[dev]"` installs cleanly
- `ruff check src/ tests/` passes
- `pytest tests/` — existing tests pass with updated imports
- `python -c "from newproject import AutoModel"` works

---

## Phase 2: Core Code Refactor + Tests

### 2.1 Replace print → logging

35+ print statements across `engine/base.py`, `utils/*.py`, `models/registry.py`, `persist/*.py`, `profiler.py`. Each file gets `logger = logging.getLogger(__name__)` and prints become `logger.debug/info/warning`.

### 2.2 Add type hints to all public APIs

Priority: `engine/base.py` (`__init__`, `forward`), `models/registry.py` (`from_pretrained`), all `utils/` functions, `persist/base.py` abstract methods.

### 2.3 Refactor the 247-line `forward()` method

Break `engine/base.py:forward()` (lines 396–642) into:

- `_reset_model()` — delete + reinit model skeleton
- `_prepare_batch()` — move inputs to device
- `_create_masks()` — attention mask + position IDs
- `_init_kv_cache()` — cache initialization
- `_run_layer_streaming()` — main loop with prefetching
- `_run_single_layer()` — dispatch to embed/norm/lm_head/transformer
- `_assemble_output()` — concat batch + build CausalLMOutputWithPast

### 2.4 Consolidate platform detection

New `utils/platform.py` with `is_macos()` and `is_cuda_available()`. Replace 5 duplicate inline checks in `__init__.py`, `auto_model.py`, `utils.py`, `persist/model_persister.py`, `airllm_base.py`.

### 2.5 Fix bugs

- **Uninitialized `targetpath`** in `utils.py:178-184` — causes `UnboundLocalError` when file is not a symlink. Fix: initialize to `None` before the conditional.
- **Typo** in `auto_model.py:44` — `"artichitecture"` → `"architecture"`

### 2.6 Consolidate model subclasses into config-driven approach

Most subclasses (Llama, Qwen2, InternLM, Mistral, Mixtral, Baichuan) are nearly identical — they just override `get_use_better_transformer()` and `get_generation_config()`. Create `models/configs.py` with `ModelConfig` dataclass + `MODEL_CONFIGS` registry. Only QWen and ChatGLM need actual subclasses (custom positional embeddings).

### 2.7 Fix `ModelPersister` — proper ABC + remove global singleton

Convert to `abc.ABC` with `@abstractmethod`. Replace module-level `model_persister = None` with proper dependency injection (pass persister to `BaseModel.__init__`).

### 2.8 Add comprehensive pytest tests

New test files: `test_model_registry.py`, `test_compression.py`, `test_platform.py`, `test_splitting.py`, `test_persister.py`, `test_base_model.py`, `test_profiler.py`, `conftest.py`. Target >80% coverage on utils/, persist/, models/.

### 2.9 Add docstrings to public API (Google-style)

### Verification

- `pytest tests/ --cov=newproject` — >80% coverage on targeted modules
- `mypy src/newproject/ --ignore-missing-imports` — passes
- Manual: load TinyLlama-1.1B, verify generation works through refactored code

---

## Phase 3: New Features — CLI + API Server

### 3.1 CLI with Typer (`src/newproject/cli.py`)

Commands:

- `newproject run <model> [-p prompt] [-n max_tokens] [-c 4bit|8bit] [-i interactive]`
- `newproject pull <model>` — download + prepare
- `newproject list` — show local models
- `newproject remove <model>`
- `newproject serve <model> [--host] [--port]`

### 3.2 Model manager (`src/newproject/model_manager.py`)

`ModelManager` class: `pull()`, `list_models()`, `remove()`, `get_model_path()` — manages local model cache.

### 3.3 FastAPI server with OpenAI-compatible API

New `src/newproject/server/` package:

- `app.py` — FastAPI app factory
- `routes.py` — `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health`
- `schemas.py` — Pydantic models (ChatCompletionRequest/Response, etc.)
- `deps.py` — dependency injection for model instance

### 3.4 Streaming token generation for PyTorch path

New `generate_stream()` method on `BaseModel` — yields decoded tokens one at a time. Each token requires a full layer-streaming forward pass (inherent to the architecture). Server uses SSE via `sse-starlette`.

### 3.5 Chat template support (`src/newproject/chat.py`)

`ChatFormatter` class: uses `tokenizer.apply_chat_template()` if available, falls back to generic formatting.

### 3.6 Interactive REPL mode

In `cli.py` with `--interactive` flag: Rich-based REPL with conversation history, streaming output.

### Verification

- `newproject pull meta-llama/Llama-3.2-1B` — downloads model
- `newproject run meta-llama/Llama-3.2-1B -p "Hello"` — generates text
- `newproject serve meta-llama/Llama-3.2-1B` → `curl localhost:8000/v1/chat/completions` — works
- Streaming: `"stream": true` returns SSE events
- `pytest tests/test_cli.py tests/test_server.py tests/test_chat.py` — passes

---

## Phase 4: Documentation + Branding + Docker

### 4.1 New README.md

Modern branding with badges, quick start (CLI + Python + Server), architecture diagram, feature list, model table, benchmarks.

### 4.2 Documentation (`docs/`)

- `api-reference.md`, `server-api.md`, `cli-reference.md`, `architecture.md`, `models.md`

### 4.3 CONTRIBUTING.md

Dev setup, running tests, code style, how to add a new model.

### 4.4 Docker support

`Dockerfile` (Python 3.11-slim, installs `[server,compression]`), `docker-compose.yml` (with GPU passthrough + model cache volume), `.dockerignore`.

### 4.5 Benchmarks (`benchmarks/`)

`benchmark_inference.py`, `benchmark_memory.py` — measure tokens/sec and peak VRAM.

### 4.6 Update CLAUDE.md for new structure

### Verification

- `docker build -t newproject . && docker run --gpus all newproject serve ...` — works
- All docs render on GitHub
- `make test && make lint && make typecheck` — full CI green

---

## Critical Files (current paths)

| File | LOC | What happens to it |
|------|-----|--------------------|
| `air_llm/airllm/airllm_base.py` | 642 | → `src/newproject/engine/base.py` — heaviest refactor (forward decomposition, logging, types, config-driven) |
| `air_llm/airllm/utils.py` | 403 | → split into `utils/memory.py`, `compression.py`, `splitting.py`, `platform.py` — bug fix, logging, types |
| `air_llm/airllm/airllm_llama_mlx.py` | 436 | → `src/newproject/engine/mlx_engine.py` — logging, types |
| `air_llm/airllm/auto_model.py` | 55 | → `src/newproject/models/registry.py` — config-driven rewrite, typo fix |
| `air_llm/airllm/persist/model_persister.py` | 39 | → `src/newproject/persist/base.py` — ABC conversion, remove global state |
| `air_llm/setup.py` | 49 | **Deleted** — replaced by `pyproject.toml` |
| 8 model subclass files | ~220 | Phase 1: move to `models/`. Phase 2: consolidate into `models/configs.py` |

## Phase Dependencies

```
Phase 1 (structure) → Phase 2 (quality) → Phase 3 (features) → Phase 4 (docs)
```

Each phase is independently shippable. We execute them in order since later phases build on earlier ones.
