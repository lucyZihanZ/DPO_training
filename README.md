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
# If you're using a GPU (e.g. H100), install PyTorch with CUDA 12.1 support:
```bash
pip install torch==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```


