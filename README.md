# CS566-Project-DR-KANTreeNet-for-Diabetic-Retinopathy-Classification

## **DR-KANTreeNet**
An enhanced deep learning framework for Diabetic Retinopathy (DR) classification using:

- 🧠 **KAN-based modules** for better feature representation  
- 🌳 **Vessel-tree modeling** to capture vascular structures  
- 🔗 **Multi-branch fusion** of CNN, ViT, and vessel features  
- ⚡ **Multi-GPU support** with efficient training tricks  

---

## 🌟 Highlights
- **Lesion-aware attention**: Focuses on subtle DR lesions  
- **VesselTreeNet**: Learns tree-like vessel structures linked to disease severity  
- **Quad-modal fusion**: Combines CNN, ViT, and vessel features with graph refinement  
- **Robust training**: Advanced augmentation, class-balanced loss, mixup, and cross-validation  

---

## 📊 Dataset

This project uses the **APTOS 2019 Blindness Detection** dataset from Kaggle.

- Place training images in:  
```data/train_images/```


- Labels file should be:  
```data/train.csv```

---

## 🚀 Training

Example (multi-GPU, mixed precision):

```bash
python All_KANS_Sencond.py --bs 16 --ep 30 --img 384 --amp
Options:

--bs: batch size

--ep: number of epochs

--img: input image size

--amp: enable automatic mixed precision
```

## 📈 Evaluation

After training, you can evaluate the model on the validation or test set.

Example:

```bash
python All_KANS_Sencond.py --evaluate --bs 16 --img 384
Metrics reported:

Accuracy

Precision

AUG

F1-score
