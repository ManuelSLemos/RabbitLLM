#!/usr/bin/env python3
"""Check registry resolution and optionally load one small model per family.

Requires: transformers>=4.50 (uv sync / pip install -e ".[dev]")
Usage:
  uv run python scripts/check_model_families.py              # resolve only (network)
  uv run python scripts/check_model_families.py --load       # load + generate (GPU recommended)
  uv run python scripts/check_model_families.py --load cpu   # load on CPU (slow)

Gated models (e.g. meta-llama/Llama-3.2-1B): set HF_TOKEN or pass --token.
  HF_TOKEN=xxx uv run python scripts/check_model_families.py
  uv run python scripts/check_model_families.py --token xxx
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

# Ensure src is on path when run as script (e.g. from repo root)
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

import torch
from rabbitllm import AutoModel

# One small model per family (all public so no HF token required)
MODELS = {
    "Qwen2.5": "Qwen/Qwen2.5-0.5B-Instruct",
    "Llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma2": "google/gemma-2-2b-it",
}


def _hf_kwargs(token: str | None) -> dict:
    """Build kwargs for gated repos: token from --token or HF_TOKEN env."""
    t = token or os.environ.get("HF_TOKEN", "").strip()
    return {"token": t} if t else {}


def check_registry(hf_kwargs: dict):
    """Resolve module/class for each family; requires network."""
    print("Checking registry resolution (transformers config fetch)...")
    for name, repo_id in MODELS.items():
        try:
            module_name, class_name = AutoModel.get_module_class(repo_id, **hf_kwargs)
            print(f"  {name:12} {repo_id:45} -> {class_name}")
        except Exception as e:
            print(f"  {name:12} {repo_id:45} FAILED: {e}")
            return False
    return True


def load_and_generate(device: str, hf_kwargs: dict):
    """Load Qwen2.5-0.5B and generate a few tokens (sanity check)."""
    repo_id = MODELS["Qwen2.5"]
    print(f"Loading {repo_id} on {device}...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA.*", category=UserWarning)
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(repo_id, device=device, **hf_kwargs)
    load_s = time.perf_counter() - t0
    print(f"  Loaded in {load_s:.2f}s")

    messages = [{"role": "user", "content": "Say hello in one word."}]
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = model.tokenizer(
        [text], return_tensors="pt", truncation=True, max_length=128
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        attention_mask = attention_mask.to(device)
    t1 = time.perf_counter()
    out = model.generate(
        input_ids, attention_mask=attention_mask, max_new_tokens=10, use_cache=True, do_sample=False
    )
    gen_s = time.perf_counter() - t1
    reply = model.tokenizer.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Reply: {reply.strip()!r} | {gen_s:.2f}s")
    return True


def main():
    p = argparse.ArgumentParser(description="Check model family compatibility")
    p.add_argument("--load", nargs="?", const="cuda:0", default=None, metavar="DEVICE",
                   help="Load Qwen2.5-0.5B and generate (device: cuda:0 or cpu)")
    p.add_argument("--token", default=None, metavar="HF_TOKEN",
                   help="HuggingFace token for gated repos (or set HF_TOKEN)")
    args = p.parse_args()

    hf_kwargs = _hf_kwargs(args.token)

    ok = check_registry(hf_kwargs)
    if not ok:
        return 1

    if args.load is not None:
        device = args.load
        if device == "cuda" or device == "cuda:0":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available, using cpu")
            device = "cpu"
        try:
            load_and_generate(device, hf_kwargs)
        except Exception as e:
            print(f"Load/generate failed: {e}")
            return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
