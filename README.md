Great — here’s the README **fully formatted for GitHub**, clean sections, emojis, badges, proper markdown hierarchy, and polished wording.

If you want repo-specific badges (tests, deployment, Docker, etc.), I can add those too.

---

# 🎧 ACNN Audio Classification – ESC-50

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Accuracy](https://img.shields.io/badge/Accuracy-86.4%25-brightgreen)
![Status](https://img.shields.io/badge/Project-Finished-success)

A lightweight yet high-performance **Audio CNN (ACNN)** trained on the **ESC-50** dataset.
It comes with a simple frontend that lets users upload any audio file and instantly see the **Top-3 predicted sound classes**.

---

## ✨ Features

* 🎙️ CNN-based environmental sound classifier
* 📈 **86.4% validation accuracy**
* 🔊 Mel-spectrogram preprocessing for robust performance
* 💾 Checkpointing with best-model saving
* 🌐 Simple web frontend for audio upload + prediction
* 🚀 Optional cloud deployment using Modal or local GPU

---

## 📂 Project Structure

```
project/
├── model/
│   └── best_model.pth
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── inference/
│   └── inference.py
├── training/
│   ├── train.py
│   ├── dataset.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture (ACNN)

A clean ConvNet pipeline optimized for ESC-50:

* Input: **128-bin Mel Spectrogram**
* Layers:

  * Conv → BatchNorm → ReLU → MaxPool
  * (Repeat for 2–3 blocks)
* Dense classification layers
* Loss: **CrossEntropy**
* Optimizer: **Adam**
* Optional: Dropout, SpecAugment

---

## 📊 Training Summary

* **Dataset**: ESC-50 (2,000 audio clips, 50 classes)
* **Sampling Rate**: 44.1 kHz
* **Specs**: 128-bin Mel Spectrogram
* **Epochs**: 50
* **Batch Size**: 32
* **Training Hardware**: GPU recommended
* **Final Accuracy**: **86.4%**

Model checkpoint example:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "accuracy": accuracy,
    "epoch": epoch,
    "classes": train_dataset.classes
}, "/models/best_model.pth")
```

---

## 🖥️ Frontend

A minimal UI where users can:

* Upload audio files
* Trigger inference via the backend
* View **Top-3 predictions** with confidence values

Example API response:

```json
{
  "predictions": [
    {"label": "Dog Bark", "confidence": 0.91},
    {"label": "Chainsaw", "confidence": 0.05},
    {"label": "Rain", "confidence": 0.02}
  ]
}
```

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Backend

```bash
python inference/inference.py
```

### 3. Launch Frontend

Static frontend:

```bash
open frontend/index.html
```

(Or host it however your stack requires.)

### 4. Upload Audio & Get Predictions

Go to the frontend → upload file → see top-3 outputs.

---

## 🚀 Deployment Options

Choose what suits your setup:

* **Local GPU** – fastest for development
* **Modal** – easy serverless deployment
* **Docker** – production-ready containers

---

## 🔮 Future Improvements

If you decide to extend the project:

* Grad-CAM-like visualization for audio
* Real-time microphone inference
* Better frontend (spectrogram preview, waveform display)
* Try CNN-Transformer hybrid models
* Add dataset augmentation + noise robustness

---

## 📝 License

MIT (change if needed)

---

If you want:

* a **banner/logo** for the top,
* **setup screenshots**,
* **live demo GIF**,
* or a **professional “Project Overview” diagram**,

I can generate those too.
