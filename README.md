# MNIST Handwritten Digit Classifier 🔢

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify handwritten digits from the MNIST dataset with **99%+ test accuracy**.

-----

## 📌 Project Overview

This project implements an end-to-end deep learning pipeline for image classification:

- **Dataset:** MNIST — 70,000 grayscale images (28×28px) across 10 digit classes (0–9)
- **Architecture:** Multi-block CNN with Batch Normalization, Dropout regularization, and Data Augmentation
- **Framework:** TensorFlow 2.x / Keras
- **Task:** Multi-class image classification

-----

## 🧠 Model Architecture

```
Input (28×28×1)
    ↓
Data Augmentation (Random Rotation, Random Zoom)
    ↓
Conv2D(32) → BatchNorm → MaxPooling
    ↓
Conv2D(64) → BatchNorm → MaxPooling
    ↓
Conv2D(128) → BatchNorm
    ↓
Flatten → Dense(256) → Dropout(0.4)
    ↓
Output: Dense(10, softmax)
```

-----

## 📊 Results

|Metric         |Value                          |
|---------------|-------------------------------|
|Test Accuracy  |~99%                           |
|Optimizer      |Adam                           |
|Loss Function  |Sparse Categorical Crossentropy|
|Training Epochs|Up to 15 (Early Stopping)      |

-----

## 🛠️ Tech Stack

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Matplotlib

-----

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mnist-digit-classifier.git
cd mnist-digit-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py
```

This will:

- Download the MNIST dataset automatically
- Train the CNN model
- Save the model to `model/mnist_cnn.keras`
- Generate training curve plots in `plots/`

### 4. Predict on your own image

```bash
python predict.py --image path/to/your/digit.png
```

-----

## 📁 Project Structure

```
mnist-digit-classifier/
│
├── train.py              # Model training script
├── predict.py            # Inference on custom images
├── requirements.txt      # Python dependencies
│
├── model/                # Saved model (generated after training)
│   └── mnist_cnn.keras
│
└── plots/                # Training visualizations (generated after training)
    ├── training_curves.png
    └── sample_predictions.png
```

-----

## 🔑 Key Concepts Demonstrated

- **CNN architecture design** — convolutional feature extraction with increasing filter depth
- **Batch Normalization** — stabilizes training and accelerates convergence
- **Dropout regularization** — prevents overfitting on training data
- **Data Augmentation** — improves generalization through random transformations
- **Early Stopping** — halts training when validation accuracy plateaus
- **End-to-end pipeline** — data loading → preprocessing → training → evaluation → inference

-----

## 👤 Author

**Naveed Ahamed**
AI/ML Engineer | B.S. Information Science (AI & ML) — Trine University

- 📧 [naveedahamed117@gmail.com](mailto:naveedahamed117@gmail.com)
- 🔗 [LinkedIn](https://linkedin.com/in/naveed-khan-90b1b217b)

-----

## 📄 License

This project is open source and available under the [MIT License](LICENSE).