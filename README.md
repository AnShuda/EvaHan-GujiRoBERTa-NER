# EvaHan-GujiRoBERTa-NER

This repository contains my submission and extended experiments for the **EvaHan2025** Classical Chinese Named Entity Recognition (NER) task.  
The goal of this project is to build a domain-adapted NER system for classical and medical Chinese texts using the GujiRoBERTa language model.

---

## 🧠 Project Overview

**Task**: Named Entity Recognition for Classical Chinese  
**Entities Covered**: Person (PER), Location (LOC), Time (TIME)  
**Model**: Fine-tuned `GujiRoBERTa` with CRF decoding and knowledge-enhanced training

---

## 🔍 Repository Structure
```
EvaHan-GujiRoBERTa-NER/
├── data/             # Training and test datasets
├── notebooks/        # Jupyter notebooks for EDA, model training and analysis
├── predictions/      # Output predictions on EvaHan test sets
├── report/           # Project summary files and evaluation charts
├── src/              # Source code (model, utils, config, trainer, etc.)
├── README.md         # This file
├── requirements.txt  # Python dependencies
```

---

## 📊 Evaluation Results

| Dataset    | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| Dev Set    | 85.3%     | 84.0%  | **84.6%** |
| Test Set A | 83.1%     | 82.4%  | **82.7%** |

The model achieves competitive results on both internal validation and the official test set of EvaHan2025.

---

## 📦 Model Checkpoints Download

> 🔗 **[Download All Checkpoints (Google Drive)](https://drive.google.com/drive/folders/10dujQZrVFjaBX1ulAGc53HqN-hH5jB5q?usp=drive_link)**

The model folder contains:
- `checkpoint-1033/`
- `checkpoint-2066/`
- `checkpoint-3099/`

Each folder includes:
- `model.safetensors`
- `config.json`
- `optimizer.pt`

⚠️ Due to GitHub's 100MB file limit, model weights are hosted externally.  
For usage instructions, see the corresponding notebook or contact me directly.

---

## ▶️ Quick Start (optional)

```bash
# 1. Clone the repository
git clone https://github.com/AnShuda/EvaHan-GujiRoBERTa-NER.git
cd EvaHan-GujiRoBERTa-NER

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run prediction (example)
python src/predict.py --input data/TestSet/raw/testset_A.txt
