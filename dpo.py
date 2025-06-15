from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import get_peft_model, LoraConfig
from huggingface_hub import login
import torch
import os
# test CUDA problem
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load dataset
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
test_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")

def compute_loss(model, tokenizer, prompt, response):
    prompt = str(prompt).strip() if prompt else ""
    response = str(response).strip() if response else ""
    input_text = prompt + response

    # Tokenize full prompt+response and compute prompt length
    full = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    prompt_only = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    labels = full["input_ids"].clone()
    prompt_len = prompt_only["input_ids"].shape[1]
    labels[:, :prompt_len] = -100  # mask prompt tokens

    with torch.no_grad():
        outputs = model(**full, labels=labels)
        loss = outputs.loss

    return loss.item()

# Flatten into prompt/chosen/rejected
def flatten_ultrafeedback(example):
    prompt_text = example["messages"][0]["content"] if "messages" in example else ""

    def format_messages(messages):
        return "\n".join([f"{m['role'].capitalize()}: {m['content'].strip()}" for m in messages])

    return {
        "prompt": prompt_text.strip(),
        "chosen": format_messages(example["chosen"]).strip(),
        "rejected": format_messages(example["rejected"]).strip(),
    }

train_dataset = dataset.map(flatten_ultrafeedback, remove_columns=dataset.column_names).select(range(100))
test_dataset = test_dataset.map(flatten_ultrafeedback, remove_columns=test_dataset.column_names).select(range(100))

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float32,
)

# Apply LoRA
# possible target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0005,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj"
    ]
)
model = get_peft_model(model, peft_config)

# DPO training config
config = DPOConfig(
    output_dir="qwen2-dpo-output",
    max_steps=100,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

# Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=config,
    tokenizer=tokenizer,
    train_dataset=train_dataset
)

trainer.train()
model.save_pretrained(config.output_dir)


# Inference method construction.
def get_model_completion(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluation
correct = 0
total = 0
for ex in test_dataset:
    prompt = ex["prompt"]
    chosen = ex["chosen"]
    rejected = ex["rejected"]
    logp_chosen = compute_loss(model, tokenizer, prompt, chosen)
    logp_rejected = compute_loss(model, tokenizer, prompt, rejected)
    print(f"Logp chosen: {logp_chosen}, rejected: {logp_rejected}, reward: {logp_rejected - logp_chosen}")
# evaluation methods:
    if logp_chosen > logp_rejected:
        correct += 1
    total += 1

print(f"Pairwise Accuracy: {correct / total:.4f}")
