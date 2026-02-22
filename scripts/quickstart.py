"""
RabbitLLM quickstart — use the library directly from Python (no CLI).

Runs on any GPU with ≥4 GB VRAM or on CPU.
"""

import warnings

import torch
from rabbitllm import AutoModel

# ── Device ────────────────────────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*CUDA.*unknown error.*", category=UserWarning)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── Load model ────────────────────────────────────────────────────────────────
# compression: "4bit" (default, fastest, ~¼ RAM), "8bit", or None (bfloat16)
# cache_layers: number of layers to keep pinned in CPU RAM — speeds up decode
#               (e.g. 30 for a 72B model on 32 GB RAM)
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device=device,
    compression="4bit",
    # cache_layers=30,
)

# ── Build prompt ──────────────────────────────────────────────────────────────
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the capital of France?"},
]

input_text = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

tokens = model.tokenizer(
    [input_text],
    return_tensors="pt",
    truncation=True,
    max_length=512,
)
input_ids = tokens["input_ids"].to(device)
attention_mask = tokens.get("attention_mask")
if attention_mask is None:
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
else:
    attention_mask = attention_mask.to(device)

# ── Generate ──────────────────────────────────────────────────────────────────
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=200,
    use_cache=True,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    return_dict_in_generate=True,
)

# Decode only the newly generated tokens
input_len = tokens["input_ids"].shape[1]
reply = model.tokenizer.decode(
    output.sequences[0][input_len:], skip_special_tokens=True
)
print(reply)
