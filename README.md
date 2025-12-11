# CS566-Project-DR-KANTreeNet-for-Diabetic-Retinopathy-Classification

## **DR-KANTreeNet**
Diabetic retinopathy (DR) is a leading cause of vision loss worldwide. **DR-KANTreeNet** is an enhanced deep-learning framework that classifies retinal fundus images into DR severity levels while offering improved interpretability. This repository contains the code and scripts used in our CS566 class project.
An enhanced deep learning framework for Diabetic Retinopathy (DR) classification using:

- 🧠 **KAN-based modules** for better feature representation  
- 🌳 **Vessel-tree modeling** to capture vascular structures  
- 🔗 **Multi-branch fusion** of CNN, ViT, and vessel features  
- ⚡ **Multi-GPU support** with efficient training tricks  

---

## 🌟 Highlights

- **KAN-based modules** for richer feature representation.
- **Vessel-tree modeling** to capture vascular structures relevant to DR severity.
- **Multi-branch fusion** combining CNN, ViT, and vessel features with graph refinement.
- **Lesion-aware attention** focusing on subtle DR lesions.
- **Robust training** with advanced augmentation, class-balanced loss, and cross-validation.
- **Multi-GPU support** for efficient training.

---

## 📊 Dataset

This project uses the **APTOS 2019 Blindness Detection** dataset (Kaggle).

1. Download the dataset from Kaggle.
2. Organize it as follows:

```text
data/
  train_images/
    000c1434d8d7.png
    001639a390f0.png
    ...
  train.csv
```

## 📦 Installation & Dependencies

Below are the required Python packages based on the project code and specifications, along with example installation commands.

1. Python version

Python 3.8+ is recommended.

2. Core packages

**Required packages:**
- `torch` – PyTorch core library (tensor computation, models)
- `torchvision` – vision models and transforms
- `timm` – Vision Transformer (ViT) and other SOTA backbones
- `opencv-python` – image I/O and basic image processing
- `matplotlib` – plotting and visualization (e.g., for probability bar charts, heatmaps)
- `numpy` – numeric operations

You can install these with pip or conda.

**Option A – Install with pip**

```bash
# (1) Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# (2) Install PyTorch + torchvision
# Choose a command from https://pytorch.org/get-started/locally/
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (example):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# (3) Install the remaining dependencies
pip install timm opencv-python matplotlib numpy
```

**Option B – Install with conda**

```bash
# (1) Create and activate environment
conda create -n dr_kantreenet python=3.10 -y
conda activate dr_kantreenet

# (2) Install PyTorch + torchvision
# Example for CUDA 11.8 (check https://pytorch.org for the latest):
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# (3) Install remaining packages
conda install -c conda-forge timm opencv matplotlib numpy
```

After installation, you can verify everything is working by running:

```bash
python -c "import torch, torchvision, timm, cv2, matplotlib, numpy as np; print('OK')"
```

---

## 🚀 Training

Example (multi-GPU, mixed precision):

```bash
python All_KANS_Sencond.py --bs 16 --ep 30 --img 384 --amp
```

**Options:**
- `--bs`: batch size
- `--ep`: number of epochs
- `--img`: input image size
- `--amp`: enable automatic mixed precision

The main training script is:

**All_KANS_Sencond.py**

Typical training command (single machine, multi-GPU if available, mixed precision):

```bash
python All_KANS_Sencond.py \
  --bs 16 \
  --ep 30 \
  --img 384 \
  --amp
```

Adjust batch size and image size based on your GPU memory.

If your dataset is in a custom path, update the relevant path variables in All_KANS_Sencond.py or the dataset configuration file.

---

## 📈 Evaluation

After training, you can evaluate the model on the validation or test set.

Example:

```bash
python All_KANS_Sencond.py --evaluate --bs 16 --img 384
```

**Metrics reported:**
- Accuracy
- Precision
- AUG
- F1-score

## 🎬 Video Demonstration

The repository includes:

- **kantree_video_demo.py** – script for generating a step-by-step analysis video for a single fundus image.

The demo typically shows:

- Image preprocessing.
- Vessel tree extraction and visualization.
- Lesion-attention heatmaps.
- DAM-enhanced local structures.
- ViT-S global context.
- Final classification result and probabilities.

**Example usage:**

```bash
python kantree_video_demo.py \
  --image_path data/train_images/000c1434d8d7.png \
  --output_video kantree_analysis_demo.mp4 \
  --weights path/to/model_checkpoint.pth \
  --fps 2 \
  --step_sec 2.0
```

**Where:**
- `--image_path` – path to a fundus image.
- `--output_video` – path for the generated .mp4 file.
- `--weights` – trained model checkpoint.
- `--fps` and `--step_sec` – control the pacing of the demo.

## 📁 Project Structure (example)

```
.
├─ All_KANS_Sencond.py          # Main training/evaluation script
├─ kantree_video_demo.py        # Video demo & visualization script
├─ KANTreeNet_Video_Presentation.md  # Project/presentation notes
├─ data/
│   ├─ train_images/            # APTOS fundus images
│   └─ train.csv                # Labels
└─ ...
```
