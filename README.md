🧠 Brain Tumor Classification using CNN
📌 Overview
This project implements a Convolutional Neural Network (CNN) to classify brain MRI images into four categories:


Glioma


Meningioma


Pituitary Tumor


No Tumor


The model is trained using PyTorch with a complete preprocessing, augmentation, and evaluation pipeline.

📂 Dataset


Source: Kaggle Brain Tumor MRI Dataset


Images resized to 224×224


Training & Testing folders used


Classes automatically extracted using ImageFolder



⚙️ Project Pipeline
1️⃣ Data Preprocessing


Resizing images to 224x224


Dataset mean & standard deviation calculation


Normalization


Data augmentation:


Random Horizontal Flip


Random Rotation (15°)




2️⃣ Train-Validation Split


80% Training


20% Validation


3️⃣ Model Architecture
Custom CNN Architecture:


4 Convolutional layers


Batch Normalization


MaxPooling


Dropout (0.25 & 0.4)


Fully Connected Layer (512 units)


Output Layer (4 classes)


Activation Function: ReLU
Loss Function: CrossEntropyLoss
Optimizer: Adam (lr=0.001)

🏗 Model Architecture Summary
Conv(3 → 32) → BN → ReLU → MaxPool
Conv(32 → 64) → BN → ReLU → MaxPool
Conv(64 → 128) → BN → ReLU → MaxPool
Conv(128 → 256) → BN → ReLU
Dropout
FC(256×14×14 → 512)
Dropout
FC(512 → 4)

📊 Evaluation


Training Accuracy


Validation Accuracy


Final Test Accuracy


GPU support (CUDA if available)



🚀 How to Run
# Install dependencies
pip install torch torchvision matplotlib kaggle

# Run notebook
jupyter notebook Brain_tumor_classification.ipynb


🛠 Tech Stack


Python


PyTorch


Torchvision


NumPy


Matplotlib



🎯 Key Learnings


Implementing CNN architecture from scratch


Dataset normalization & augmentation


Preventing overfitting using Dropout & BatchNorm


Building full training + evaluation pipeline
