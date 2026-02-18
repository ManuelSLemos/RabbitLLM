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

## Phase 2.5: Model compatibility (dependencias + registro)

*Antes de seguir con planner/tools: actualizar librerías y ampliar el registro para poder ejecutar modelos actuales (Qwen3, Gemma, DeepSeek, Phi, Llama 3.2, etc.).*

### 2.5.1 Actualizar dependencias

- **transformers**: ampliar rango soportado (ej. `>=4.47,<4.57` o superior) para poder usar 4.50–4.56 y más modelos recientes.
- Ajustar import de `GenerationMixin` si en versiones nuevas pasa a `transformers.generation.utils`.
- **accelerate**, **safetensors**, **huggingface-hub**: revisar versiones mínimas recomendadas.

### 2.5.2 Ampliar registro de arquitecturas

En `models/registry.py`, añadir ramas para arquitecturas recientes (mapeando a clases existentes cuando la estructura sea compatible):

- **Qwen3** → `RabbitLLMQWen2` (o módulo qwen2 si la API es compatible).
- **Gemma2 / Gemma3** → Llama-like (`RabbitLLMLlama2`).
- **DeepSeekV2 / DeepSeekV3** → Llama-like.
- **Phi2 / Phi3** → Llama-like.
- **Llama3_2** y variantes → `RabbitLLMLlama2`.

Comprobar en la documentación de transformers los nombres exactos de `config.architectures` para cada familia.

### 2.5.3 Actualizar docs

- `docs/COMPATIBILITY.md`: nuevo rango de transformers, matriz de modelos ampliada, notas por familia (Gemma, DeepSeek, Phi, etc.).

### Verification

- `pip install` con transformers 4.50+ (o la versión objetivo).
- `AutoModel.get_module_class()` resuelve correctamente para repos de prueba (Qwen2.5, Llama-3.2, Gemma2, Phi3, etc.).
- Al menos un modelo reciente por familia carga y genera sin error.

---

## Phase 3: Planner + auto strategy + tools

*Hacer esto antes de la CLI/API para que el producto sea útil de verdad: el runtime debe poder decidir la estrategia y ofrecer diagnóstico y benchmarks.*

### 3.1 Módulo planner (`src/rabbitllm/planner/`)

- Detección de VRAM disponible (PyTorch/CUDA).
- Estimación de memoria del modelo a partir del config (sin compresión, 4bit, 8bit).
- `ExecutionPlan` (dataclass): profile (balanced/fast/tight), device, dtype, compression, prefetch.
- Función `plan(model_id_or_path, cache_dir?) -> ExecutionPlan`.
- Log claro al arrancar: `plan=tight reason="..." strategy="..."`.

### 3.2 Integración con el engine

- `AutoModel.from_pretrained(..., execution_plan=plan)` o que el plan se traduzca a kwargs (device, compression, dtype, prefetch).

### 3.3 Herramientas `rabbit doctor` y `rabbit bench`

- **doctor**: CUDA, driver, VRAM, versiones torch/CUDA, perfil recomendado (usar `utils/platform.py` y lógica del planner).
- **bench**: script reproducible por modelo (tokens/s, time-to-first-token, pico VRAM, OOM). Reutilizar o extender `profiler.py`.

### 3.4 Config opcional `rabbit.toml`

- `device`, `profile`, `max_vram_percent`, `cache_dir`. El planner lee esto si existe.

### Verification

- Planner elige perfil correcto según VRAM disponible vs estimado.
- `rabbit doctor` (cuando exista CLI) o script equivalente informa estado del sistema.
- `rabbit bench <model>` (cuando exista CLI) o script devuelve métricas.

---

## Phase 4: Documentation + Branding + Docker

### 4.1 New README.md

Modern branding with badges, quick start (Python API first; CLI/API cuando existan), architecture diagram, feature list, model table.

### 4.2 Documentation (`docs/`)

- `api-reference.md`, `architecture.md`, `models.md`; más adelante `server-api.md`, `cli-reference.md`.

### 4.3 CONTRIBUTING.md

Dev setup, uv, tests, code style, how to add a new model.

### 4.4 Docker support

`Dockerfile`, `docker-compose.yml` (GPU + model cache), `.dockerignore`.

### 4.5 Benchmarks (`benchmarks/`)

`benchmark_inference.py`, `benchmark_memory.py` — tokens/s y pico VRAM.

### 4.6 Update CLAUDE.md for new structure

### Verification

- `docker build -t rabbitllm .` — works
- Docs render on GitHub
- `make test && make lint && make typecheck` — green

---

## Phase 5: CLI + API Server (último)

*Dejar para el final: requiere planner, model manager y streaming para ser realmente útil.*

### 5.1 CLI with Typer (`src/rabbitllm/cli.py`)

Commands:

- `rabbit run <model> [-p prompt] [-n max_tokens] [-c 4bit|8bit] [-i interactive]`
- `rabbit pull <model>` — download + prepare
- `rabbit list` — show local models
- `rabbit remove <model>`
- `rabbit serve <model> [--host] [--port]`
- `rabbit doctor` — diagnostics (Phase 3)
- `rabbit bench <model>` — benchmarks (Phase 3)

### 5.2 Model manager (`src/rabbitllm/model_manager.py`)

`ModelManager` class: `pull()`, `list_models()`, `remove()`, `get_model_path()` — manages local model cache.

### 5.3 FastAPI server with OpenAI-compatible API

New `src/rabbitllm/server/` package:

- `app.py` — FastAPI app factory
- `routes.py` — `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health`
- `schemas.py` — Pydantic models (ChatCompletionRequest/Response, etc.)
- `deps.py` — dependency injection for model instance

### 5.4 Streaming token generation for PyTorch path

New `generate_stream()` method on `BaseModel` — yields decoded tokens one at a time. Server uses SSE via `sse-starlette`.

### 5.5 Chat template support (`src/rabbitllm/chat.py`)

`ChatFormatter` class: uses `tokenizer.apply_chat_template()` if available, falls back to generic formatting.

### 5.6 Interactive REPL mode

In `cli.py` with `--interactive` flag: Rich-based REPL with conversation history, streaming output.

### Verification

- `rabbit pull <model>`, `rabbit run <model> -p "Hello"`, `rabbit serve <model>`
- `curl localhost:8000/v1/chat/completions` — works; streaming con `"stream": true`
- `pytest tests/test_cli.py tests/test_server.py tests/test_chat.py` — passes

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
Phase 1 (structure) → Phase 2 (quality) → Phase 3 (planner + tools) → Phase 4 (docs + Docker) → Phase 5 (CLI + API)
```

- **Phase 3** (planner, doctor, bench) hace que el runtime sea útil antes de exponer CLI/API.
- **Phase 5** (CLI + API) va al final: depende del planner y del model manager para ser realmente útil.
