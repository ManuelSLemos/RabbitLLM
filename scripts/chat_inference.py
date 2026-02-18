import torch
from rabbitllm import AutoModel

MAX_LENGTH = 128
MAX_NEW_TOKENS = 50

model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

print("Modelo cargado. Escribe 'exit' para salir.\n")

while True:
    user_input = input(">>> ")
    if user_input.strip().lower() in ("exit", "quit"):
        break

    input_tokens = model.tokenizer(
        [user_input],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    input_ids = input_tokens["input_ids"].to(model.device)
    attention_mask = input_tokens.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model.device)
    else:
        attention_mask = attention_mask.to(model.device)

    generation_output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        return_dict_in_generate=True,
    )

    output = model.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    print(f"\n{output}\n")
