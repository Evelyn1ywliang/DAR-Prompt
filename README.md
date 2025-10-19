# DAR-Prompt

This repository contains the implementation of **DAR-Prompt**, a dynamic regulation framework for multi-label zero-shot learning.

---

## 🧩 Environment Setup

Please follow the environment setup from the [DualCoOp](https://github.com/sunxm2357/DualCoOp) before running the scripts.  
Make sure all dependencies (e.g., PyTorch, torchvision, and other required libraries) are properly installed.

```bash
conda activate dualcoop
```

## 🚀 Training

To train the model on the NUS-WIDE dataset, simply run:

```bash
bash scripts/train_nus_wide_zsl.sh
```

## 🧪 Evaluation

To evaluate the trained model:

```bash
bash scripts/val_nus_wide_zsl.sh
```
⚠️ Note: This script expects the path to the checkpoint folder.
