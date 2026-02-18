# Plan: Refactor & Rebrand RabbitLLM

## Context

RabbitLLM is a discontinued project with a powerful core idea: **layer-streaming inference** that allows 70B+ LLMs to run on 4GB GPUs without quantization. The codebase works but suffers from legacy packaging, zero CI/CD, no type hints, print-based "logging", minimal tests (73 LOC), and missing features that modern tools like Ollama/vLLM offer (CLI, API server, streaming, chat templates). The goal is a full refactor + rebrand into a modern, production-grade project with both CLI and server capabilities.

**Name**: RabbitLLM (package `rabbitllm`; CLI `rabbit` from Phase 3 onward).

---

## Phase 1: Restructure + Modernize Packaging

**Review (current repo):** The project already uses `src/rabbitllm/` with `engine/`, `models/`, `persist/`, `utils/`, `compat/`, and `profiler.py`. No `rabbit_llm/` exists. `pyproject.toml` (hatchling, deps, ruff, pytest) and `.gitignore` are in place. Imports use `rabbitllm` paths. Below: what is **done** vs **pending**.

### 1.1 Directory structure — DONE

`src/rabbitllm/` layout and `tests/` already match the target. No migration needed.

### 1.2 Legacy directories — MOSTLY DONE

`training/`, `rlhf/`, `anima_100k/`, `eval/`, `data/`, `rabbit_llm/`, `requirements.txt`, `README_ja.md` are not present.

- **Pending:** Remove empty `scripts/` if no longer needed (or keep for future helper scripts).

### 1.3 `pyproject.toml` — DONE (entry point in Phase 3)

- Build system, Python ≥3.10, dependencies and optional extras (`compression`, `flash`, `server`, `dev`) are set. Transformers is `>=4.47,<4.49`.
- **Do not add** CLI entry point until Phase 3 (no `rabbitllm.cli` yet). Optionally add `[mlx]` extra when relevant.

### 1.4 Makefile — PENDING

Current Makefile only has `bash` (Docker). Add:

- `install` — `uv sync --extra dev`
- `dev` — same as install
- `lint` — `ruff check src/ tests/`
- `format` — `ruff format src/ tests/`
- `test` — `pytest tests/`
- `test-cov` — `pytest tests/ --cov=rabbitllm`
- `typecheck` — `mypy src/rabbitllm/ --ignore-missing-imports`
- `clean` — remove `build/`, `dist/`, `*.egg-info`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `htmlcov/`, `.coverage`

Keep `bash` if you use it for Docker.

### 1.5 GitHub Actions CI — PENDING

Create `.github/workflows/ci.yml`:

- Trigger on push/PR to main (or master).
- Matrix: Python 3.10, 3.11, 3.12.
- Steps: checkout, set up Python, install with `.[dev]`, `ruff check src/ tests/`, `ruff format --check src/ tests/`, `pytest tests/` (skip or mark CUDA-only tests so CI passes without GPU).

### 1.6 `.gitignore` — DONE (optional tweaks)

Current file is adequate. Optionally add: `.mypy_cache/`, `.ruff_cache/`.

### 1.7 Imports — DONE

Code already uses `rabbitllm` package paths; no `rabbit_llm` references in source.

### Phase 1 verification (after completing pending items)

- `uv sync --extra dev` (or `make install`) installs cleanly
- `make lint` and `make format` pass
- `make test` — existing tests pass
- `uv run python -c "from rabbitllm import AutoModel"` works

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

New `utils/platform.py` with `is_macos()` and `is_cuda_available()`. Replace 5 duplicate inline checks in `__init__.py`, `auto_model.py`, `utils.py`, `persist/model_persister.py`, `rabbitllm_base.py`.

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

- `pytest tests/ --cov=rabbitllm` — >80% coverage on targeted modules
- `mypy src/rabbitllm/ --ignore-missing-imports` — passes
- Manual: load TinyLlama-1.1B, verify generation works through refactored code

---

## Phase 3: New Features — CLI + API Server

### 3.1 CLI with Typer (`src/rabbitllm/cli.py`)

Commands:

- `rabbitllm run <model> [-p prompt] [-n max_tokens] [-c 4bit|8bit] [-i interactive]`
- `rabbitllm pull <model>` — download + prepare
- `rabbitllm list` — show local models
- `rabbitllm remove <model>`
- `rabbitllm serve <model> [--host] [--port]`

### 3.2 Model manager (`src/rabbitllm/model_manager.py`)

`ModelManager` class: `pull()`, `list_models()`, `remove()`, `get_model_path()` — manages local model cache.

### 3.3 FastAPI server with OpenAI-compatible API

New `src/rabbitllm/server/` package:

- `app.py` — FastAPI app factory
- `routes.py` — `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health`
- `schemas.py` — Pydantic models (ChatCompletionRequest/Response, etc.)
- `deps.py` — dependency injection for model instance

### 3.4 Streaming token generation for PyTorch path

New `generate_stream()` method on `BaseModel` — yields decoded tokens one at a time. Each token requires a full layer-streaming forward pass (inherent to the architecture). Server uses SSE via `sse-starlette`.

### 3.5 Chat template support (`src/rabbitllm/chat.py`)

`ChatFormatter` class: uses `tokenizer.apply_chat_template()` if available, falls back to generic formatting.

### 3.6 Interactive REPL mode

In `cli.py` with `--interactive` flag: Rich-based REPL with conversation history, streaming output.

### Verification

- `rabbitllm pull meta-llama/Llama-3.2-1B` — downloads model
- `rabbitllm run meta-llama/Llama-3.2-1B -p "Hello"` — generates text
- `rabbitllm serve meta-llama/Llama-3.2-1B` → `curl localhost:8000/v1/chat/completions` — works
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

- `docker build -t rabbitllm . && docker run --gpus all rabbitllm serve ...` — works
- All docs render on GitHub
- `make test && make lint && make typecheck` — full CI green

---

## Critical Files (current paths)

| File | What happens to it |
|------|--------------------|
| `src/rabbitllm/engine/base.py` | Phase 2: forward decomposition, logging, types, config-driven |
| `src/rabbitllm/utils/*.py` | Phase 2: bug fix (splitting), logging, types |
| `src/rabbitllm/engine/mlx_engine.py` | Phase 2: logging, types |
| `src/rabbitllm/models/registry.py` | Phase 2: typo fix ("artichitecture"), config-driven optional |
| `src/rabbitllm/persist/base.py` | Phase 2: ABC conversion, remove global state |
| 8 model subclass files in `models/` | Phase 2: optional consolidate into `models/configs.py` |

## Phase Dependencies

```
Phase 1 (structure) → Phase 2 (quality) → Phase 3 (features) → Phase 4 (docs)
```

Each phase is independently shippable. We execute them in order since later phases build on earlier ones.
