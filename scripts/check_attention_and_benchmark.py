#!/usr/bin/env python3
"""
Check if Flash Attention is active and compare throughput vs other implementations.

Shows:
  1. Result of is_flash_attention_available() (compatibility check).
  2. Which attention implementation the model actually uses after loading (auto).
  3. Optional: quick benchmark comparing auto vs sdpa (and optionally eager) so you
     can see if Flash gives you a speedup on your machine.

Usage:
  # Only check compatibility and what the model uses (small model, fast)
  uv run python scripts/check_attention_and_benchmark.py

  # Check + benchmark (compare auto vs sdpa; needs a bit more time)
  uv run python scripts/check_attention_and_benchmark.py --benchmark

  # Custom model and device
  uv run python scripts/check_attention_and_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda:0 --benchmark
"""

import argparse
import logging
import time
import warnings

import torch
from rabbitllm import AutoModel
from rabbitllm.utils.platform import is_flash_attention_available

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Check Flash Attention availability and optionally benchmark implementations"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model path or HuggingFace repo ID (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (default: cuda:0 if available else cpu)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a short benchmark comparing auto vs sdpa (and eager) to see throughput difference",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Max new tokens per generate in benchmark (default 30)",
    )
    parser.add_argument("--token", default=None, help="HuggingFace token for gated repos")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA.*unknown error.*", category=UserWarning)
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 1. Compatibility check ---
    logger.info("=== Flash Attention compatibility ===")
    flash_ok, flash_msg = is_flash_attention_available()
    logger.info("  %s", flash_msg)
    if flash_ok:
        logger.info("  -> With attn_implementation='auto', the model will use Flash Attention 2.")
    else:
        logger.info("  -> With attn_implementation='auto', the model will use SDPA (or eager) instead.")
    logger.info("")

    # --- 2. What does the model actually use? ---
    logger.info("=== Loading model with attn_implementation='auto' ===")
    model = AutoModel.from_pretrained(
        args.model,
        device=device,
        attn_implementation="auto",
        token=args.token,
    )
    impl = model.active_attention_implementation
    logger.info("  Active implementation: %s", impl)
    if impl == "flash_attention_2":
        logger.info("  -> Flash Attention is ACTIVE. You should see benefits on Ampere+ GPUs (speed, sometimes memory).")
    else:
        logger.info("  -> Flash Attention is not in use. Reason above (compatibility check).")
    logger.info("")

    if not args.benchmark:
        logger.info("Tip: run with --benchmark to compare throughput (auto vs sdpa) and see if Flash helps on your machine.")
        return

    # --- 3. Optional benchmark ---
    logger.info("=== Quick benchmark (throughput: new tokens / second) ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2? Answer in one number."},
    ]
    if hasattr(model.tokenizer, "apply_chat_template"):
        input_text = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = "What is 2+2? Answer in one number."

    def run_one(m, label: str) -> float:
        inputs = m.tokenizer(
            [input_text],
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        input_ids = inputs["input_ids"].to(m.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=m.device)
        else:
            attention_mask = attention_mask.to(m.device)
        start = time.perf_counter()
        out = m.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            do_sample=False,
            return_dict_in_generate=True,
        )
        elapsed = time.perf_counter() - start
        sequences = out.sequences if hasattr(out, "sequences") else out
        n_new = sequences.shape[1] - input_ids.shape[1]
        tok_s = n_new / elapsed if elapsed > 0 else 0
        logger.info("  %s: %.2f tok/s (active impl: %s)", label, tok_s, m.active_attention_implementation)
        return tok_s

    run_one(model, "auto (recommended)")
    model_sdpa = AutoModel.from_pretrained(
        args.model, device=device, attn_implementation="sdpa", token=args.token
    )
    run_one(model_sdpa, "sdpa")
    if device.startswith("cuda"):
        model_eager = AutoModel.from_pretrained(
            args.model, device=device, attn_implementation="eager", token=args.token
        )
        run_one(model_eager, "eager")
    logger.info("")
    logger.info("Compare the tok/s above: if 'auto' uses Flash and is faster than sdpa/eager, you get a benefit.")

if __name__ == "__main__":
    main()
