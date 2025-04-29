import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from dataset import SkinLesionDataset
import pandas as pd

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Output folders
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Augmented transform for training
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ]
)

# Plain transform for validation
val_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)

# Load datasets
print("Loading datasets...")
train_dataset = SkinLesionDataset("data/train.csv", transform=train_transform)
val_dataset = SkinLesionDataset("data/val.csv", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Load ResNet18 model
print("Loading ResNet18 model...")
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# --- Option A: Class-weighted CrossEntropyLoss ---
weights = torch.tensor([1.0, 3.0], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# --- Option B: Focal Loss (uncomment below to use) ---
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ce = nn.CrossEntropyLoss(reduction='none')
#     def forward(self, input, target):
#         logp = self.ce(input, target)
#         p = torch.exp(-logp)
#         loss = self.alpha * (1 - p) ** self.gamma * logp
#         return loss.mean()
# criterion = FocalLoss(alpha=1, gamma=2)

# Optimiser
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Training loop
def train(num_epochs=15):
    history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        print(f"\nEpoch {epoch + 1}/{num_epochs} starting...")

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 0:
                print(f"Batch {i}, Loss: {loss.item():.4f}")

        accuracy = 100 * correct / total
        average_loss = running_loss / len(train_loader)

        epoch_result = {"epoch": epoch + 1, "loss": average_loss, "accuracy": accuracy}
        history.append(epoch_result)

        print(
            f"Epoch [{epoch + 1}] Avg Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

    # Save training results
    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "training_log.csv"), index=False
    )
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, "resnet18_skin_weighted_augmented.pt"),
    )
    print("\nTraining complete.")
    print(f"Model saved to {MODEL_DIR}/resnet18_skin.pt")
    print(f"Training log saved to {OUTPUT_DIR}/training_log.csv")


# Run training
start = time.time()
print("Starting training...")
train(num_epochs=15)
print(f"Training finished in {time.time() - start:.2f} seconds.")
