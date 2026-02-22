# Changelog

All notable changes to RabbitLLM are documented here.

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
