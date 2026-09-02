import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./checkpoints"

# load dataset

ds = load_dataset("mecha-org/linux-command-dataset")["train"]
ds = ds.train_test_split(test_size=0.05, seed=42)
train_ds, eval_ds = ds["train"], ds["test"]
print(f"Training samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

SYSTEM_PROMPT = "You are a Linux command-line assistant. Given a natural language instruction, respond with ONLY the exact bash command that accomplishes it. Do not include explanations."

# tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_example(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


train_ds = train_ds.map(format_example)
eval_ds = eval_ds.map(format_example)

# quantisation

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# attach lora adapters

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# train

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    bf16=True,
    max_length=256,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Adapter saved to {OUTPUT_DIR}")
