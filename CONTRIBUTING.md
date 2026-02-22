# Contributing to RabbitLLM

Thank you for your interest in contributing. This document explains how to set up
the development environment, run tests, and add support for new models.

## Development setup

RabbitLLM uses [uv](https://docs.astral.sh/uv/) as the package manager.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/manuelslemos/rabbitllm
cd rabbitllm

# Install all dependencies including dev tools
uv sync --extra dev
# or: make install
```

This creates a virtual environment at `.venv/` with ruff, mypy, pytest, and pytest-cov.

## Running tests

```bash
# All tests (skips GPU-only compression test)
make test
# or: uv run pytest tests/

# With coverage report
make test-cov
# or: uv run pytest tests/ --cov=rabbitllm --cov-report=term-missing

# Single test module
uv run python -m unittest tests.test_model_registry

# GPU tests (requires CUDA)
uv run pytest tests/test_compression.py
```

Most tests run without a GPU. The compression test (`test_compression.py`) requires CUDA
and is skipped automatically in CI.

## Code style

RabbitLLM uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
make lint      # ruff check
make format    # ruff format
make typecheck # mypy
```

Line length is 100 characters. Imports are sorted (isort via ruff). Type hints are
required on all public APIs.

## Project structure

```
src/rabbitllm/
├── engine/          # Core layer-streaming engine
│   ├── base.py          # RabbitLLMBaseModel — the main class
│   ├── attention.py     # Attention implementation resolution
│   ├── model_init.py    # Model skeleton creation
│   ├── layer_loading.py # Layer I/O and async transfer
│   ├── forward_utils.py # Attention mask, position ids, KV extraction
│   └── mlx_engine.py    # macOS/MLX implementation
├── models/          # Architecture-specific subclasses
│   ├── registry.py      # AutoModel factory
│   ├── llama.py         # Llama2/3, Gemma, Phi, DeepSeek
│   ├── qwen.py          # Qwen v1
│   ├── qwen2.py         # Qwen2/2.5/3
│   ├── mistral.py       # Mistral
│   ├── mixtral.py       # Mixtral
│   ├── chatglm.py       # ChatGLM
│   ├── baichuan.py      # Baichuan
│   └── internlm.py      # InternLM
├── persist/         # Layer serialization (safetensors / mlx)
├── utils/           # Splitting, compression, memory, platform
├── compat/          # Tokenizer compatibility shims
└── profiler.py      # Per-layer timing
```

## Adding a new model

1. **Check if the existing Llama-based class works.** Many Llama-like architectures
   (Gemma, Phi, DeepSeek) share the same layer naming convention and work without a
   dedicated subclass. Try `AutoModel.from_pretrained("your/model")` first.

2. **Create a subclass** if the architecture uses different layer names:

   ```python
   # src/rabbitllm/models/mymodel.py
   from ..engine.base import RabbitLLMBaseModel

   class RabbitLLMMyModel(RabbitLLMBaseModel):
       def set_layer_names_dict(self) -> None:
           self.layer_names_dict = {
               "embed": "transformer.wte",      # embedding layer
               "layer_prefix": "transformer.h", # decoder layers
               "norm": "transformer.ln_f",      # final norm
               "lm_head": "lm_head",            # language model head
           }
   ```

3. **Register the architecture** in `src/rabbitllm/models/registry.py`:

   ```python
   if "MyModelForCausalLM" in arch:
       return "rabbitllm.models.mymodel", "RabbitLLMMyModel"
   ```

   Check the model's `config.json` on HuggingFace to find the exact `architectures` value.

4. **Export the class** in `src/rabbitllm/__init__.py`:

   ```python
   from .models.mymodel import RabbitLLMMyModel
   ```

5. **Add a test** in `tests/test_model_registry.py`:

   ```python
   "org/my-model": "RabbitLLMMyModel",
   ```

6. **Verify with a small model** (e.g. 1B–7B parameter variant):

   ```python
   from rabbitllm import AutoModel
   model = AutoModel.from_pretrained("org/my-model-7b")
   out = model.generate(model.tokenizer(["Hello"], return_tensors="pt")["input_ids"].cuda(), max_new_tokens=10)
   print(model.tokenizer.decode(out.sequences[0]))
   ```

## Submitting a pull request

- Open an issue first for large changes to discuss the approach.
- Include tests for any new functionality.
- Run `make lint && make test` before opening the PR.
- Keep PRs focused; one feature or fix per PR.

## Reporting issues

Please include:
- Python version, PyTorch version, transformers version
- GPU model and VRAM (if applicable)
- The model repo ID being used
- Full traceback
