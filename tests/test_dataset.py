import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

# Ensure the src module is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SkinLesionDataset


def _create_sample_dataset(tmp_path: Path):
    data = []
    labels = ["benign", "malignant"]
    for idx, label in enumerate(labels):
        img_path = tmp_path / f"img_{idx}.jpg"
        Image.new("RGB", (300, 300)).save(img_path)
        data.append({"image_path": str(img_path), "label": label})
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path, len(labels)


def test_skin_lesion_dataset(tmp_path):
    csv_path, num_classes = _create_sample_dataset(tmp_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = SkinLesionDataset(csv_path=str(csv_path), transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    images, labels = next(iter(loader))
    assert images.ndim == 4
    assert images.shape[1:] == (3, 224, 224)
    assert labels.min().item() >= 0
    assert labels.max().item() < num_classes
