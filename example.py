#!/usr/bin/env python3
"""
RabbitLLM example — minimal inference script.

Run: python example.py
Or:  uv run python example.py

Uses a small model (Qwen2.5-0.5B) for fast testing. For larger models or long
context, see scripts/quickstart.py and the Configuration section in README.
"""

import warnings

import torch
from rabbitllm import AutoModel

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*CUDA.*unknown error.*", category=UserWarning)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device=device,
    compression="4bit",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2 + 2? Answer briefly."},
]

input_text = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
tokens = model.tokenizer(
    [input_text], return_tensors="pt", truncation=True, max_length=512
)
input_ids = tokens["input_ids"].to(device)
attention_mask = tokens.get("attention_mask")
if attention_mask is None:
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
else:
    attention_mask = attention_mask.to(device)

output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=64,
    use_cache=True,
    do_sample=True,
    temperature=0.6,
    return_dict_in_generate=True,
)

input_len = tokens["input_ids"].shape[1]
print(model.tokenizer.decode(output.sequences[0][input_len:], skip_special_tokens=True))
