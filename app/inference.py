import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "./checkpoints"

SYSTEM_PROMPT = "You are a Linux command-line assistant. Given a natural language instruction, respond with ONLY the exact bash command that accomplishes it. Do not include explanations."


class InferenceEngine:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map="auto"
        )

        self.model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        self.model.eval()

    def generate(self, nl_input: str) -> str:
        assert self.tokenizer is not None and self.model is not None, (
            "call load() first"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": nl_input},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
            response = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
        return response.strip()
