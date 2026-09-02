# training/test_inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "./checkpoints"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

SYSTEM_PROMPT = "You are a Linux command-line assistant. Given a natural language instruction, respond with ONLY the exact bash command that accomplishes it. Do not include explanations."


def generate(nl_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": nl_input},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response.strip()


# Try a mix: things close to training data, and things phrased more "in the wild"
test_prompts = [
    "compress the file report.txt using bzip2",
    "show me all files modified in the last 24 hours",
    "find all python files bigger than 1mb",
    "kill the process running on port 8080",
    "how much disk space is free",
]

for p in test_prompts:
    print(f"IN:  {p}")
    print(f"OUT: {generate(p)}")
    print()
