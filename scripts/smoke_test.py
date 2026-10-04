"""Быстрая проверка API генерации до полного запуска ноутбука.

Проверяет: загрузку Qwen2.5-1.5B-Instruct в fp16 на mps, батчевую генерацию через
apply_chat_template с левым паддингом, greedy-декодинг, сэмплирование с
num_return_sequences, а также дефолты generation_config.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
model.to("mps")
model.eval()
print(f"параметров: {model.num_parameters():,}")
print(f"generation_config по умолчанию: {model.generation_config}")

texts = [
    "А когда планируется награждение?",
    "Спасибо большое, всё работает отлично!",
    "Ужасный сервис, второй день жду ответа.",
]
messages = [[{"role": "user", "content": f"Определи тональность текста: {t}"}] for t in texts]

tok.padding_side = "left"
enc = tok.apply_chat_template(
    messages, tokenize=True, padding=True, return_dict=True, add_generation_prompt=True,
    return_tensors="pt",
).to("mps")
print(f"batch shape: {enc['input_ids'].shape}")

# 1) greedy
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=8, do_sample=False, num_beams=1,
                         pad_token_id=tok.pad_token_id)
gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
print("greedy:", gen)

# 2) sampling, 5 вариантов на пример
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=8, do_sample=True, temperature=0.7,
                         top_p=0.95, num_return_sequences=5,
                         pad_token_id=tok.pad_token_id)
print(f"sampling shape: {out.shape}")  # ожидаем [3*5, seq]
gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
for i in range(5):
    print(f"  вариант {i + 1} для 1-го текста: {gen[i]!r}")

# 3) beam search
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=8, num_beams=3, do_sample=False,
                         pad_token_id=tok.pad_token_id)
gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
print("beam-3:", gen)
print("SMOKE OK")
