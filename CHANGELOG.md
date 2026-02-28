# Changelog

All notable changes to RabbitLLM are documented here.

## [1.1.0]

### Added
- **Qwen3 support** (`RabbitLLMQWen3`): full support for Qwen3ForCausalLM with `head_dim`
  correction for a transformers 5.2 bug, per-layer `head_dim` fix during streaming, and custom
  RoPE position embeddings (`get_pos_emb_args`) that compute correct (cos, sin) per layer.
- **`offload_small_layers`**: embed/norm/lm_head are offloaded to CPU each forward pass instead
  of staying on GPU. Enables 72B bfloat16 inference on 8 GB VRAM. Pairs with `kv_cache_dir`.
- **`offload_small_layers_use_cpu_cache`** (default `True`): keeps embed/norm/lm_head state_dicts
  in CPU RAM after the first forward so decode steps skip disk I/O (~2.5–5 GiB extra RAM for 72B).
- **`cache_layers`**: keeps up to N transformer layers in CPU RAM between forward passes.
  Cache hits skip disk I/O entirely; only a fast RAM→GPU copy is paid on decode steps.
- **`DiskKVCache`** (`engine/kvcache.py`): `kv_cache_dir` option to offload the KV cache to SSD.
  Supports 50k+ token contexts without OOM; only one layer's K/V stays in VRAM at a time.
- **GPU Direct Storage** (`utils/kvikio_loader.py`): when `kvikio-cu12` is installed, layers load
  directly from disk to GPU, bypassing CPU and pin_memory entirely. Install with
  `pip install rabbitllm[gds]`. Fixed to use PyTorch-managed buffers (not CuPy) to avoid a
  2× VRAM spike during parallel initial loads.
- **`use_gds`** parameter (default `True`): enable/disable kvikio when available.
- **`prefetch_pin_memory`** parameter (default `True`): set to `False` to skip OS page-locking
  during prefetch. Useful for benchmarking; Fix 2 (pinned memory pool) will make this unnecessary.
- **`--no-pin-memory`** CLI flag in `scripts/inference_example.py`: exposes `prefetch_pin_memory=False`
  without editing the script. See profiling notes — disable pin_memory alone is not recommended
  for production use until the pinned memory pool (Fix 2) is implemented.
- **`--profile` CLI flag**: enables `profiling_mode=True`; caps `--max-new-tokens` to 3 for quick
  profiling runs. Prints per-step and aggregate breakdown tables.
- **`--do-sample`, `--temperature`, `--top-p`, `--no-think`** CLI flags in inference_example.py.
- **Async transfer pipeline with dual prefetch** (`engine/pipeline.py`): Phase A/B architecture
  fully overlaps CPU load, CPU→GPU async copy, and GPU forward. `use_dual_prefetch` keeps two
  concurrent CPU-load slots to saturate the pipeline for 70B+ models.
- **Async GPU decompression** (`async_decompress`): when using 4-bit/8-bit compression with async
  transfer, raw packed weights are copied to GPU on the transfer stream and decompressed on the
  default stream (Phase B), overlapping decompression with the previous layer's forward.
- **`example.py`** in project root for quick onboarding.
- **`samples/`** directory with sample text for long-context testing.
- **Test suite**: `tests/test_kvcache.py`, `tests/test_kvikio_loader.py`, `tests/test_profiler.py`,
  `tests/test_base_model.py`, and 6 additional test modules.
- **Dockerfile** for easier deployment: build from repo with `docker build -t rabbitllm .`, run with `--gpus all` for GPU inference. Installs RabbitLLM with optional `[gds]` extra. README Docker subsection documents build, run, and env vars (`HF_TOKEN`, `HF_HOME`). Makefile targets `docker-build` and `docker-run`.
- **Benchmark section** in README: table of benchmark scripts (GDS/long-context, CPU vs CUDA, attention comparison) and link to [docs/BENCHMARK_HISTORY.md](docs/BENCHMARK_HISTORY.md) for detailed 72B results.

### Changed
- **Pipeline extracted to `engine/pipeline.py`**: three strategies — `_no_prefetch_pipeline`,
  `_sync_prefetch_pipeline`, and `_async_transfer_pipeline` — selected automatically based on
  hardware. `base.py` now delegates all transfer logic to `create_pipeline()`.
- **Profiler categories expanded** (`profiler.py`): added `Small layer cache populate`,
  `Small layer cache hit (clone)`, `RoPE position embeddings`, `Tied lm_head load`.
  Added `accumulate_into()` for cross-step aggregation and a final aggregate report after
  `generate()`. Profiler table shows `Avg/layer` column.
- `load_layer_to_cpu` now tries kvikio (GDS) first when available and compression is not used.
- `scripts/inference_example.py` expanded with all new flags; extras appear in the load-time
  summary line (e.g. `[offload_small_layers=True, prefetch_pin_memory=False]`).
- README documents `use_gds`, `kv_cache_dir`, `offload_small_layers`, `cache_layers`, and
  the optional `[gds]` extra.
- **Documentation and in-repo text translated to English**: `docs/TRANSFORMERS_UPGRADE_PLAN.md`, `docs/BENCHMARK_HISTORY.md`, `docs/COMPATIBILITY.md`, `docs/TROUBLESHOOTING.md`, and `example.py` (docstring and comments).

### Fixed
- **RoPE decode correctness**: prefill `position_embeddings_cache` is no longer reused in decode
  steps. Previously produced garbage output on models with `q_norm`/`k_norm` (Qwen3).
- **Flash Attention 2 mask slicing**: attention mask is trimmed to `[past_len + seq_len]` to avoid
  the FA2 varlen path receiving out-of-bounds indices.
- **SDPA causal masking**: SDPA attention receives `mask=None` and handles causal masking
  natively via `is_causal=True`, avoiding nan/inf from manually-constructed causal masks.
- **`_set_param_direct` stream visibility**: async-transferred parameters are assigned via
  in-place `param.data =` replacement to keep the `Parameter` object identity stable across
  streams, preventing intermittent `device mismatch` errors on the transfer stream.

## [1.0.1] — 2026-02-22

### Fixed
- `rabbitllm.models` subpackage missing from installed wheel due to `.gitignore` pattern `models/` matching `src/rabbitllm/models/` during hatchling build; anchored to `/models/` and added explicit `include = ["src/rabbitllm/**"]` in `pyproject.toml`
- `qwen3.py` not tracked in git for the same reason

### Added
- `scripts/quickstart.py` — minimal Python example (no CLI) for loading a model and generating text
- CI now runs on `develop` branch in addition to `main`

### Changed
- README quickstart updated to use `apply_chat_template`, explicit `attention_mask`, device auto-detection and new-tokens-only decoding

## [1.0.0] — 2026-02-21

Initial release of **RabbitLLM** — a complete rewrite and rebrand of the layer-streaming inference engine.

### Added
- Layer-streaming inference engine: runs 70B+ models on 4GB VRAM without quantization
- `AutoModel.from_pretrained()` — auto-detects architecture from HuggingFace config
- Optional 4-bit/8-bit block-wise compression via bitsandbytes (up to 3× speed-up)
- Async CPU→GPU transfer pipeline to overlap layer loading with compute
- KV cache support (`DynamicCache`) for incremental decoding
- Flash Attention 2 auto-detection (`attn_implementation="auto"`)
- macOS / Apple Silicon support via MLX (`RabbitLLMLlamaMlx`)
- Supported architectures: Llama 2/3/3.1/3.2, Qwen v1/2/2.5/3, Mistral, Mixtral, ChatGLM, Baichuan, InternLM, Gemma 2/3, DeepSeek V2/V3, Phi 2/3/4
- `src/` layout with `pyproject.toml`, `uv` packaging, `ruff`, `mypy`, `pytest`
- GitHub Actions CI across Python 3.10 / 3.11 / 3.12
- Technical documentation: `ARCHITECTURE.md`, `COMPATIBILITY.md`, `TROUBLESHOOTING.md`
- `scripts/`: inference examples, benchmark, attention checker, profiling

### Notes
- Requires Python ≥ 3.10, PyTorch ≥ 2.5, transformers 5.0–5.2
- For Qwen2/Qwen2.5, use transformers 5.0.x; 5.1+ has a known RoPE head_dim issue (see `docs/COMPATIBILITY.md`)
