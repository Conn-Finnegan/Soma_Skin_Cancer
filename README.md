# Soma Skin Cancer Classifier

Soma is a privacy‑focused tool that classifies dermoscopic images of skin lesions as **benign** or **malignant** using a fine‑tuned ResNet‑18 model. The current model reaches about **92% accuracy** and produces Grad‑CAM heatmaps to show what the network sees.

## Project Overview
- Trained on the HAM10000 dataset to distinguish benign from malignant lesions.
- Fine‑tuned ResNet‑18 achieves roughly 92% accuracy on a held‑out test set.
- Generates Grad‑CAM heatmaps for transparent predictions and includes scripts for training, evaluation, and demos.

## Quick Start
Follow these steps to run predictions with the pretrained model.

1. **Clone the repository**
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
4. **Download the pretrained model**
   * Place `skin_cancer_resnet18_version1.pt` in the `models/` folder.
5. **Run predictions**
   * Add up to 20 `.jpg` or `.png` files to `test_images/`.
   * Execute:
     ```bash
     python src/predict_batch.py
     ```
   * Predictions print to the console and Grad‑CAM overlays are saved to `outputs/`.

## Train the Model Yourself
1. **Prepare the data**
   * Download the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
   * Copy `HAM10000_images_part_1`, `HAM10000_images_part_2`, and `HAM10000_metadata.csv` into a new `data/` directory.
   * Generate train/validation splits:
     ```bash
     python src/load_data.py
     ```
2. **Train**
   ```bash
   python src/train_model.py
   ```
3. **Evaluate**
   ```bash
   python src/evaluate.py
   ```

## Launch the Web Demo
```bash
cd app
pip install -r requirements.txt
python app.py
```

## iOS App Integration
* Convert the model to Core ML:
  ```bash
  python convert_model.py
  ```
* Use `skin_cancer_resnet18_v1.mlmodel` in your SwiftUI project.

## Project Structure
```text
├── app/                    # Gradio demo
├── data/                   # Dataset (ignored by git)
├── models/                 # Pretrained or trained models
├── outputs/                # Grad-CAM images and logs
├── src/                    # Training and inference scripts
└── tests/                  # Unit tests
```

## Privacy
All image processing occurs locally in memory—no images or personal data are stored or sent elsewhere.

## License
GNU General Public License v3.0

## Author
**Conn Finnegan**  
GitHub: [@Conn-Finnegan](https://github.com/Conn-Finnegan)  
LinkedIn: [Conn Finnegan](https://www.linkedin.com/in/conn-finnegan-09a98124b/)

---
Powered by PyTorch, Gradio & Hugging Face Spaces  
© 2025 Conn Finnegan
