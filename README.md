# DigitAI — Handwritten Digit Recognition

> A full-stack AI system that recognizes handwritten digits (0–9) using a Convolutional Neural Network trained on the MNIST dataset.

![Python](https://img.shields.io/badge/Python-3.14-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red) ![FastAPI](https://img.shields.io/badge/FastAPI-green) ![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen)

---

## 🏗️ Architecture

```
MNIST Dataset (70,000 images)
         ↓
  train_model.py   ←  CNN Training (PyTorch)
         ↓
  digit_model.pth  ←  Saved model weights
         ↓
     api.py        ←  FastAPI REST server
         ↓
  static/index.html ← Interactive Web UI
         ↓
  User draws/uploads → Predicted digit (0–9)
```

---

## 🚀 Quick Start

### One-Click Launch (Windows)
```bat
run.bat
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Train the CNN model (~5-10 minutes)
python train_model.py

# Start the server
python api.py

# Open in browser
# http://localhost:8000
```

---

## 📁 Project Structure

```
Bittu/
├── train_model.py     # CNN training script (MNIST)
├── api.py             # FastAPI server + prediction endpoint
├── requirements.txt   # Python dependencies
├── run.bat            # One-click Windows launcher
├── digit_model.pth    # Saved model (created after training)
├── data/              # MNIST dataset (auto-downloaded)
└── static/
    ├── index.html     # Web UI (parallax, draw canvas, upload)
    ├── style.css      # Design system (dark glassmorphism)
    ├── app.js         # Interactive JS (canvas, charts, API)
    ├── history.json   # Training history (created after training)
    ├── training_graph.png
    └── sample_predictions.png
```

---

## 🧠 CNN Model Architecture

| Layer         | Type          | Details              |
|---------------|---------------|----------------------|
| Input         | Tensor        | 1×28×28 grayscale    |
| Conv1 + ReLU  | Conv2D        | 32 filters, 3×3      |
| Pool1         | MaxPool2D     | 2×2 → 14×14          |
| Conv2 + ReLU  | Conv2D        | 64 filters, 3×3      |
| Pool2         | MaxPool2D     | 2×2 → 7×7            |
| Flatten       | —             | 3136 features        |
| Dense + ReLU  | Linear        | 128 neurons          |
| Dropout       | Regularization| rate = 0.3           |
| Output        | Linear+Softmax| 10 classes (0–9)     |

**Target Accuracy:** 99%+

---

## 🌐 API Endpoints

| Method | Endpoint    | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/`         | Serve web UI                         |
| POST   | `/predict`  | Upload image → `{digit, confidence, probabilities}` |
| GET    | `/metrics`  | Training history (accuracy, loss)    |
| GET    | `/health`   | Server health + model status         |
| GET    | `/docs`     | Interactive API docs (Swagger UI)    |

---

## 📊 Preprocessing Pipeline

1. **Load**: Accept image upload (PNG/JPG/JPEG)
2. **Grayscale**: Convert to single channel
3. **Invert**: Dark-on-white → white-on-dark (MNIST style)
4. **Resize**: Bicubic interpolation to 28×28
5. **Normalize**: Pixels 0–255 → 0.0–1.0
6. **Standardize**: `(x - 0.1307) / 0.3081` (MNIST stats)
7. **Tensor**: Shape `(1, 1, 28, 28)`

---

## 🎓 Viva Q&A

**What is CNN?**
> A Convolutional Neural Network that extracts spatial features from images using learnable filters (kernels). Each Conv layer detects different patterns: edges → curves → complex shapes.

**Why MNIST?**
> The standard benchmark dataset for digit recognition. Contains 70,000 28×28 grayscale images with balanced classes (7,000 per digit), allowing fair training and evaluation.

**Why normalization?**
> Bringing pixel values from 0–255 to 0–1 ensures stable gradient flow during backpropagation, prevents vanishing/exploding gradients, and allows faster convergence.

**What is Softmax?**
> A function that converts raw logits into a probability distribution summing to 1. Each output neuron represents the probability that the input is a specific digit (0–9).

**What is Dropout?**
> A regularization technique that randomly deactivates 30% of neurons during training, forcing the network to learn redundant features and preventing overfitting.

**What is Adam optimizer?**
> Adaptive Moment Estimation — combines momentum and RMSprop. Adapts learning rates per parameter, making it fast and effective for deep learning.

---

## 🎯 Project Checklist

- ✅ CNN model with Conv2D, MaxPool, Dense, Dropout, Softmax
- ✅ MNIST dataset (70,000 images, no manual download)
- ✅ Preprocessing pipeline (normalize + reshape)
- ✅ Training with Adam + CrossEntropy
- ✅ Accuracy ≥ 99% expected
- ✅ Model saved as `.pth`
- ✅ Prediction engine (REST API)
- ✅ Web UI with Draw Canvas
- ✅ Web UI with Image Upload
- ✅ Confidence bars (0–9)
- ✅ Training graphs (Accuracy + Loss)
- ✅ Parallax + smooth scroll
- ✅ CNN architecture visualization

---

*Built with PyTorch · FastAPI · Chart.js*
