# 🧠 Brain Tumor Classification using Deep Learning

A deep learning-based image classification system that detects and classifies brain tumors from MRI scans using Convolutional Neural Networks (CNNs).

---

## 📌 Project Overview

Brain tumors can be life-threatening if not diagnosed early. This project builds a CNN model to classify brain MRI images into four categories:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

The model is implemented using **PyTorch** with a complete training and evaluation pipeline including preprocessing, augmentation, validation, and testing.

---

## 🗂 Dataset

- **Source:** Kaggle Brain Tumor MRI Dataset  
- **Number of Classes:** 4  
- **Image Size:** 224 × 224  
- **Dataset Loader:** `torchvision.datasets.ImageFolder`

The dataset is divided into:

- Training Set  
- Testing Set  
- Validation Split (created from training data)

---

## ⚙️ Features

- End-to-end CNN training pipeline
- Dataset normalization using computed mean & standard deviation
- Data augmentation (Random Rotation + Horizontal Flip)
- Custom CNN architecture built from scratch
- Batch Normalization for stable convergence
- Dropout layers to reduce overfitting
- GPU support (CUDA auto-detection)
- Training & validation accuracy tracking

---

## 🏗 Model Architecture

The model consists of:

- 4 Convolutional Layers
- Batch Normalization
- ReLU Activation
- MaxPooling
- Dropout Layers
- Fully Connected Layers
- Output Layer (4 Classes)

---

### 🔄 Architecture Flow

Input (3×224×224)
- Conv(3 → 32) → BatchNorm → ReLU → MaxPool
- Conv(32 → 64) → BatchNorm → ReLU → MaxPool
- Conv(64 → 128) → BatchNorm → ReLU → MaxPool
- Conv(128 → 256) → BatchNorm → ReLU
- Dropout
- Fully Connected (→ 512)
- Dropout
- Fully Connected (→ 4)
---

---

## 🧪 Training Configuration

| Parameter        | Value |
|------------------|--------|
| Framework        | PyTorch |
| Loss Function    | CrossEntropyLoss |
| Optimizer        | Adam |
| Learning Rate    | 0.001 |
| Device           | CPU / GPU (Auto-detected) |

---

## 📊 Evaluation Metrics

- Training Accuracy  
- Validation Accuracy  
- Test Accuracy  

The model is evaluated on unseen MRI scans to measure generalization performance.

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision matplotlib numpy
```

## 🧠 Key Learnings

- Designing CNN architectures from scratch  
- Implementing dataset normalization and augmentation  
- Preventing overfitting using Dropout and Batch Normalization  
- Building a complete PyTorch training pipeline  

---

## 🔮 Future Improvements

- Add Confusion Matrix and Classification Report  
- Implement Transfer Learning (ResNet / EfficientNet)  
- Deploy using FastAPI  
- Build a Streamlit web interface for real-time predictions  

---

## 🛠 Tech Stack

- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  
