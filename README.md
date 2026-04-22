# ASL Hand Gesture Recognition System
### A Vision-Based Hand Gesture Recognition System Using Deep Learning for Assistive Communication

---

## Project Overview

This project implements an end-to-end American Sign Language (ASL) fingerspelling recognition system using deep learning. Five model architectures are trained and compared on the Sign Language MNIST dataset, with the best-performing model (ResNet-style, 100% accuracy) deployed as a real-time web application with webcam capture and text-to-speech output.

---

## Project Structure

```
Research Project/
│
├── data/
│   ├── sign_mnist_train.csv          ← Training dataset (27,455 samples)
│   └── sign_mnist_test.csv           ← Test dataset (7,172 samples)
│
├── Sign_Language_Full.ipynb          ← Main notebook (all 5 models)
│
├── models/
│   ├── resnet_asl_model.h5           ← Saved ResNet model (100%)
│   └── efficientnet_asl_model.h5     ← Saved EfficientNetB0 model (99.05%)
│
└── gui/
    ├── app.py                        ← Flask backend (prediction API)
    ├── requirements.txt              ← Python dependencies
    └── templates/
        └── index.html                ← Frontend (webcam UI)
```

---

## Dataset

- **Name:** Sign Language MNIST
- **Source:** [Kaggle — Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **Training samples:** 27,455
- **Test samples:** 7,172
- **Classes:** 24 ASL letters (A–Y, excluding J and Z — both require motion)
- **Image size:** 28×28 grayscale pixels

---

## Models Trained

| Model | Type | Test Accuracy | Train Time | Input |
|---|---|---|---|---|
| Random Forest | Classical ML | 82.11% | ~1–2 min | 784 flat |
| SVM (RBF) | Classical ML | 83.71% | ~5–20 min | 784 flat |
| Custom CNN | Deep Learning | 94.31% | ~8 min | 28×28×1 |
| EfficientNetB0 | Transfer Learning | 99.05% | ~25 min | 64×64×3 |
| ResNet-style | Deep Learning | **100.00%** | ~12 min | 28×28×1 |

---

## Installation

### Step 1 — Clone or download the project

```bash
git clone https://github.com/your-username/asl-gesture-recognition.git
cd asl-gesture-recognition
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r gui/requirements.txt
```

Full dependency list:

```
flask>=2.3.0
tensorflow>=2.12.0
numpy>=1.23.0
Pillow>=9.0.0
pandas
matplotlib
seaborn
scikit-learn
```

---

## Running the Notebook

### Step 1 — Download the dataset

Download from Kaggle and place both CSV files in the `data/` folder:
- `data/sign_mnist_train.csv`
- `data/sign_mnist_test.csv`

### Step 2 — Open the notebook

```bash
jupyter notebook Sign_Language_Full.ipynb
```

### Step 3 — Run all cells in order

The notebook is structured in 10 sections:

| Section | Description |
|---|---|
| 1 | Install & Imports |
| 2 | Load Dataset |
| 3 | Exploratory Data Analysis |
| 4 | Preprocessing |
| 5 | Random Forest & SVM Baselines |
| 6 | Custom CNN with Grid Search |
| 7 | ResNet-style with Skip Connections |
| 8 | EfficientNetB0 with Transfer Learning |
| 9 | Full Model Comparison |
| 10 | Word Builder & TTS Output |

---

## Hyperparameter Tuning (Manual Grid Search)

No external tuning library is required. The manual grid search is built into the notebook using `itertools.product`.

### ResNet tuning grid

```python
dropout_rates  = [0.3, 0.4, 0.5]
learning_rates = [1e-2, 1e-3, 1e-4]
dense_units    = [128, 256]
# Total: 18 configurations
```

To reduce run time, narrow the grid:

```python
dropout_rates  = [0.3, 0.5]
learning_rates = [1e-3, 1e-4]
dense_units    = [256]
# Total: 4 configurations (~10 minutes)
```

---

## Saving the Trained Models

After training, save the models by adding these cells to your notebook:

```python
# Save ResNet model
resnet_model.save("models/resnet_asl_model.h5")
print("ResNet model saved.")

# Save EfficientNetB0 model
eff_model.save("models/efficientnet_asl_model.h5")
print("EfficientNetB0 model saved.")
```

---

## Running the Web Application

### Step 1 — Save your trained ResNet model

```python
resnet_model.save("gui/resnet_asl_model.h5")
```

### Step 2 — Confirm folder structure

```
gui/
├── app.py
├── resnet_asl_model.h5     ← must be here
├── requirements.txt
└── templates/
    └── index.html          ← must be inside templates/
```

> **Important:** `index.html` must be inside a `templates/` subfolder inside `gui/`. This is a Flask requirement. Placing it elsewhere will cause a `TemplateNotFound` error.

### Step 3 — Run the Flask app

```bash
cd gui
python app.py
```

### Step 4 — Open in browser

```
http://localhost:5000
```

---

## Using the Web App

1. Click **Start Camera** — allow webcam access when prompted
2. **Position your hand** in the frame showing an ASL letter gesture
3. Click **Capture & Predict** — the model predicts the letter in real time
4. The predicted letter, confidence score, and top-3 predictions are displayed
5. Letters are **automatically appended** to the Word Builder
6. Use **Space**, **Delete**, and **Clear** to manage the built word

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the frontend UI |
| POST | `/predict` | Accepts an image, returns prediction |
| GET | `/health` | Returns model loading status |

### POST /predict — Request & Response

**Request:** `multipart/form-data` with field `image` (JPEG or PNG)

**Response:**
```json
{
  "letter": "A",
  "confidence": 0.9912,
  "class_idx": 0,
  "top3": [
    { "letter": "A", "confidence": 0.9912 },
    { "letter": "B", "confidence": 0.0065 },
    { "letter": "S", "confidence": 0.0023 }
  ],
  "demo_mode": false
}
```

---

## Demo Mode

If no `.h5` model file is found in the `gui/` folder, the app runs in **demo mode** with simulated predictions. This is useful for testing the frontend UI before your model is ready. The response will include `"demo_mode": true`.

---

## Switching to EfficientNetB0

To use EfficientNetB0 instead of ResNet, save the model:

```python
eff_model.save("gui/efficientnet_asl_model.h5")
```

Set the environment variable before running:

```bash
# Windows
set MODEL_PATH=efficientnet_asl_model.h5
python app.py

# Mac/Linux
MODEL_PATH=efficientnet_asl_model.h5 python app.py
```

Note: EfficientNetB0 requires images to be resized to 64×64 RGB. The preprocessing in `app.py` will need to be updated accordingly.

---

## Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `TemplateNotFound: index.html` | `index.html` not in `templates/` folder | Move `index.html` into `gui/templates/` |
| `ModuleNotFoundError: keras_tuner` | keras-tuner not installed | Run `%pip install keras-tuner` in notebook |
| `OSError: resnet_asl_model.h5` | Model file missing | Run `resnet_model.save(...)` in notebook first |
| `Address already in use` | Port 5000 already occupied | Run `python app.py --port 5001` or kill the existing process |
| Camera not working | Browser permissions blocked | Click the camera icon in the address bar and allow access |

---

## ASL Letter Reference

The model recognises 24 static ASL letters:

```
A  B  C  D  E  F  G  H  I  K
L  M  N  O  P  Q  R  S  T  U
V  W  X  Y
```

> J and Z are excluded — both require hand motion and cannot be represented as static images.

---

## Future Work

- CNN-LSTM hybrid to extend coverage to J and Z
- Real-time webcam deployment with OpenCV hand detection
- MobileNetV2 for mobile / on-device inference
- Language model integration for word completion and sentence building
- Cross-validation against an independently collected real-world dataset

---

## References

- He, K. et al. (2016) 'Deep residual learning for image recognition', CVPR.
- Tan, M. and Le, Q.V. (2019) 'EfficientNet: rethinking model scaling', ICML.
- LeCun, Y. et al. (1998) 'Gradient-based learning applied to document recognition', IEEE.
- Bantupalli, K. and Xie, Y. (2018) 'ASL recognition using deep learning', IEEE Big Data.

---

## Project Info


- **Date:** 22 March 2026
- **Best Model:** ResNet-style — 100% test accuracy
- **Dataset:** Sign Language MNIST (Kaggle)
