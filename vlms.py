from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import json
import re

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

image = Image.open("data/carla.png").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": """
You are an autonomous driving scene analysis module.

Task:
Determine whether there is at least one vehicle visible in the image.

Output requirements:
- Return ONLY a valid JSON object.
- Do not return markdown.
- Do not return code fences.
- Do not return any explanation.
- The JSON object must contain exactly two keys:
  1. "answer": must be "yes" or "no"
  2. "confidence": must be a decimal number between 0 and 1

Confidence definition:
- "confidence" is your estimated probability that your answer is correct.

Required output format:
{"answer":"yes","confidence":0.82}
"""
            }
        ]
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,   # 更稳定
        temperature=0.0
    )

# 只取新生成的部分，不包含输入 prompt
new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
output_text = processor.batch_decode(
    new_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)[0].strip()

print("Raw model output:")
print(output_text)

# 先直接解析 JSON
parsed = None
try:
    parsed = json.loads(output_text)
except json.JSONDecodeError:
    # 兜底：从输出中提取第一个 {...}
    match = re.search(r"\{.*\}", output_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

if parsed is None:
    raise ValueError(f"Model output is not valid JSON: {output_text}")

# 简单校验
if "answer" not in parsed or "confidence" not in parsed:
    raise ValueError(f"Missing required keys in output: {parsed}")

if parsed["answer"] not in ["yes", "no"]:
    raise ValueError(f'Invalid answer value: {parsed["answer"]}')

parsed["confidence"] = float(parsed["confidence"])
if not (0.0 <= parsed["confidence"] <= 1.0):
    raise ValueError(f'Confidence out of range: {parsed["confidence"]}')

print("\nParsed result:")
print(parsed)