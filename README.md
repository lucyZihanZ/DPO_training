# 🔧 DPO Training with Qwen2-0.5B-Instruct on UltraFeedback (Binarized)

This repository contains a training pipeline for fine-tuning a large language model (LLM) using **Direct Preference Optimization (DPO)** on binarized preference data.

We use the **Qwen2-0.5B-Instruct** model and the **Truthy-DPO** (or UltraFeedback-binarized) dataset to align the LLM with human preferences.

---

## 📌 Highlights

- 📚 **Model**: [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- 🧾 **Dataset**: [`jondurbin/truthy-dpo-v0.1`](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1) or [`stanfordnlp/UltraFeedback`](https://huggingface.co/datasets/stanfordnlp/UltraFeedback)
- 🧠 **Objective**: Train the LLM to prefer high-quality completions using binary human feedback
- 🛠️ **Method**: Direct Preference Optimization (DPO), using HuggingFace's `trl` library

---

## 🧱 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```
If you're using a GPU (e.g. H100), install PyTorch with CUDA 12.1 support:
```bash
pip install torch==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```
## 🚀 Running DPO Training
You can launch training with:
```bash
python hf_train.py \
    --epochs 10 \
    --batch_size 32 \
    --max_length 256 \
    --lr 1e-5 \
    --beta 0.1 \
    --seed 2003 \
    --model_name "Qwen/Qwen2-0.5B-Instruct" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "truthy-dpo"
```

## 🔧 Argument Descriptions
```bash
### 🧾 Training Arguments

| Argument         | Description                                                   |
|------------------|---------------------------------------------------------------|
| `--epochs`       | Number of training epochs                                     |
| `--batch_size`   | Per-device batch size                                         |
| `--lr`           | Learning rate                                                 |
| `--beta`         | DPO temperature (preference strength)                         |
| `--max_length`   | Max input token length                                        |
| `--seed`         | Random seed for reproducibility                               |
| `--model_name`   | Hugging Face model ID to load and fine-tune                   |
| `--dataset_name` | Hugging Face dataset to use (must include `chosen` & `rejected`) |
| `--wandb_project`| Project name for Weights & Biases (W&B) logging               |
```
## 📊 Logging & Monitoring
This repo integrates with Weights & Biases for experiment tracking. To use it:
```bash
wandb login
```
Logs include:

Training loss

DPO loss

Beta value

Accuracy (if implemented)

Prompt/chosen/rejected samples (optional)

## 📦 Checkpoints
Checkpoints will be saved to:
```bash
./checkpoints/{model_name}-dpo-{dataset_name}
```
You can resume or push to Hugging Face Hub if desired.

## 🧠 References
[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)

[UltraFeedback Dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

[TRL: Transformers Reinforcement Learning](https://huggingface.co/docs/trl/index)

## 📜 License
This repository is distributed under the MIT License. See LICENSE for details.

✨ Acknowledgements
Special thanks to:
```yaml
Alibaba Qwen Team

Stanford NLP Group

Hugging Face's trl and datasets libraries
```
Let me know if you'd like:

An inference script for testing your fine-tuned DPO model

Instructions to push to the HuggingFace Hub

Or a setup.sh to auto-install the environment on HPC/cluster systems.



