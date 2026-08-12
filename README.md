# 🐱🐶 Cat vs Dog Image Classifier

A binary image classifier that distinguishes cats from dogs, built with TensorFlow/Keras and transfer learning (MobileNetV2). Trained on ~2,000 images, achieving **99.57% test accuracy**.

---

## 📌 Overview

This project started as a from-scratch CNN and evolved into a transfer-learning pipeline after hitting (and fixing) three real bugs along the way — a data-shuffling issue, a BatchNormalization convergence issue, and a preprocessing mismatch at inference time. The full writeup of that debugging process is here: **[Medium article link — PLACEHOLDER]**

## 🎯 Results

| Metric | Score |
|---|---|
| Accuracy | 99.57% |
| Precision | 99.18% |
| Recall | 100% |

Kaggle dataset Link :- https://www.kaggle.com/code/ronikdedhia/cat-and-dog-classification

## 🏗️ Architecture

- **Base model:** MobileNetV2 (pretrained on ImageNet, `include_top=False`, frozen)
- **Head:** `GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)`
- **Preprocessing:** data augmentation (random flip/rotation/zoom) + `Rescaling(1./127.5, offset=-1)` applied inside the model
- **Training:** Adam optimizer, binary cross-entropy loss, EarlyStopping on `val_loss` (patience=5, restore best weights)

```
Input (256x256x3)
  → Data Augmentation
  → Rescaling [-1, 1]
  → MobileNetV2 (frozen)
  → GlobalAveragePooling2D
  → Dense(128, relu)
  → Dropout(0.3)
  → Dense(1, sigmoid)
```

## 📂 Project Structure

```
.
├── classification.ipynb        # Main notebook — data pipeline, training, evaluation
├── cat_dog_classifier.keras    # Saved trained model
├── requirements.txt
├── README.md
└── Data/                       # Not included — see Dataset section below
    ├── Cat/
    └── Dog/
```

## 📊 Dataset

~2,023 labeled images across two classes (Cat / Dog). Not included in this repo due to size — download a cats-vs-dogs dataset (e.g., from Kaggle) and place images into `Data/Cat/` and `Data/Dog/` before running.

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in `Data/Cat/` and `Data/Dog/`.

4. Open and run the notebook:
   ```bash
   jupyter notebook classification.ipynb
   ```

5. To classify your own image, update the image path in the final prediction cell and run it.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- OpenCV
- Matplotlib / NumPy

## 🐛 Key Debugging Lessons (worth reading before you copy this)

1. **`image_dataset_from_directory` reshuffles every epoch by default.** If you split into train/val/test using `.take()`/`.skip()` on the same dataset object, your "validation set" silently changes every epoch unless you call `.cache()` after loading.
2. **BatchNorm layers need enough training steps to converge.** On small datasets trained from scratch, validation performance can end up *below* random chance if BatchNorm's running statistics never stabilize — a strong signal to consider transfer learning with a frozen, pretrained backbone instead.
3. **Preprocessing must match between training and inference.** If your model has an internal `Rescaling` layer, don't also manually scale input images before calling `.predict()` — you'll double-scale and effectively feed the model a blank image.



## 📄 License

MIT License — feel free to use this for learning purposes.

## 🙋 About

Built by Moinuddin Ghare as a portfolio/learning project during an AI/Python internship.
Connect on [LinkedIn — PLACEHOLDER]
