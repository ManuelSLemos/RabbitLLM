#!/usr/bin/env python3
"""
Inferencia 70B+ sin cuantización, con KV cache en disco (evita OOM en 8 GB VRAM).
"""

import tempfile
import warnings

import torch
from rabbitllm import AutoModel

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*CUDA.*unknown error.*", category=UserWarning)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Directorio para el KV cache (en disco, no en GPU)
kv_cache_dir = tempfile.mkdtemp(prefix="rabbitllm_kv_")
# Para uso persistente: kv_cache_dir = "./kv_cache"

model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    device=device,
    compression=None,           # sin cuantización, full precision
    kv_cache_dir=kv_cache_dir, # KV cache a disco → evita OOM en 8 GB
    max_seq_len=512,           # ajusta si necesitas contexto más largo
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
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