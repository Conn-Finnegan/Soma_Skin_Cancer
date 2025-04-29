import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from dataset import SkinLesionDataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on device: {device}")

# Load validation dataset
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

val_dataset = SkinLesionDataset("data/val.csv", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/resnet18_skin.pt", map_location=device))
model.to(device)
model.eval()

# Evaluation loop
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Classification Report
print("\n--- Classification Report ---")
print(
    classification_report(all_labels, all_preds, target_names=["benign", "malignant"])
)

# Confusion Matrix (text)
cm = confusion_matrix(all_labels, all_preds)
print("\n--- Confusion Matrix ---")
print(cm)

# Visual Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["benign", "malignant"],
    yticklabels=["benign", "malignant"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()

# Save the figure
plt.savefig("outputs/confusion_matrix.png")
print("Confusion matrix plot saved to outputs/confusion_matrix.png")

# Optionally display it
plt.show()
