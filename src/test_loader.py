# test_loader.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SkinLesionDataset
import matplotlib.pyplot as plt

# Define transforms
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Load dataset
train_dataset = SkinLesionDataset(csv_path="data/train.csv", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


# Show a batch of images
def show_batch(images, labels):
    grid = torch.cat([img for img in images], dim=2)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Labels: {labels.tolist()}")
    plt.axis("off")
    plt.show()


# Load one batch and display
for images, labels in train_loader:
    show_batch(images, labels)
    break
