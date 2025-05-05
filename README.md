# Soma Skin Cancer Classifier


**Soma** is a privacy-focused AI tool for classifying dermoscopic images of skin lesions as **benign** or **malignant**. It leverages a fine-tuned ResNet‑18 model and provides Grad‑CAM visualisations for interpretability.

## 🔍 Project Overview

* **Model**: ResNet‑18 (pretrained on ImageNet, final layer adjusted for 2 classes)
* **Dataset**: HAM10000 dermoscopic image dataset
* **Accuracy**: \~89% overall
* **Malignant Recall**: \~78% (prioritised for safety)
* **Explainability**: Grad‑CAM heatmap overlays
* **Privacy**: All inference runs in-memory; no images are stored or logged
* **Deployment**:

  * **Hugging Face Space**: [Try Soma Live](https://huggingface.co/spaces/Conn-Cerberus/Soma_Skin_Cancer_Classifier)
  * **PyTorch**: `skin_cancer_resnet18_version1.pt`
  * **Core ML**: `skin_cancer_resnet18_v1.mlmodel` (for iOS app integration)

## ⚙️ Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/Conn-Finnegan/Soma_Skin_Cancer.git
   cd Soma_Skin_Cancer
   ```
2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Download or train the model

* **Pretrained model**: Place `skin_cancer_resnet18_version1.pt` in the `models/` folder.
* **To train from scratch**, adjust `src/train_model.py` and run:

  ```bash
  python src/train_model.py
  ```

### 2. Run evaluation on validation set

```bash
python src/evaluate_model.py
```

### 3. Predict on custom images

```bash
python src/predict_batch.py
```

* Processes up to 20 images in `test_images/`
* Outputs predictions and saves Grad‑CAM overlays to `outputs/`

### 4. Launch Hugging Face Space locally

```bash
cd app
pip install -r requirements.txt
python app.py
```

## 📱 iOS App Integration

* Convert to Core ML:

  ```bash
  python convert_model.py
  ```
* Integrate `skin_cancer_resnet18_v1.mlmodel` into your SwiftUI Xcode project.

## 📄 File Structure

```
├── app/                    # Gradio web app for demo
│   ├── app.py
│   └── requirements.txt
├── data/                   # Optional: raw dataset or test images
├── models/                 # Model files (.pt, .mlmodel)
├── outputs/                # Saved Grad-CAM overlays and logs
├── src/                    # Training, evaluation, and prediction scripts
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict_batch.py
│   └── convert_model.py
├── thumbnail.png           # Project thumbnail
├── README.md
└── requirements.txt        # Python dependencies
```

## 🔐 Privacy Notice

> **All image processing is done locally and in-memory.**
> No images, logs, or personal data are stored or transmitted.

## 📝 License

GNU General Public License v3.0 

## 🙋‍♂️ Author

**Conn Finnegan**

* GitHub: [@Conn-Finnegan](https://github.com/Conn-Finnegan)
* LinkedIn: [Conn Finnegan](https://www.linkedin.com/in/conn-finnegan-09a98124b/)

---

> Powered by PyTorch, Gradio & Hugging Face Spaces
> © 2025 Conn Finnegan
