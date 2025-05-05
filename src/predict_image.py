import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# --- CONFIG ---
model_path = "models/resnet18_skin_weighted_earlystop.pt"
input_folder = "test_images"
output_folder = "outputs"
max_images = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["benign", "malignant"]

# --- Transforms ---
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# --- Load Model ---
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# --- Predict ---
def predict(img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, 1)
        return prediction.item(), confidence.item(), probs.squeeze().cpu().numpy()


# --- Grad-CAM ---
def generate_gradcam(img_tensor, predicted_class):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    final_conv = model.layer3[1].conv2  # Earlier layer = sharper Grad-CAM
    forward_handle = final_conv.register_forward_hook(forward_hook)
    backward_handle = final_conv.register_backward_hook(backward_hook)

    output = model(img_tensor)
    class_score = output[0, predicted_class]
    model.zero_grad()
    class_score.backward()

    grads = gradients[0].squeeze().cpu().detach().numpy()
    acts = activations[0].squeeze().cpu().detach().numpy()
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))

    if np.max(cam) != 0:
        cam = cam / np.max(cam)  # Safe division
    return cam


# --- Run Batch ---
def run_batch(folder, limit):
    image_files = [
        f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    image_files = image_files[:limit]

    for filename in image_files:
        image_path = os.path.join(folder, filename)
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        pred_idx, conf, all_probs = predict(img_tensor)
        label = class_names[pred_idx]

        print(f"🖼️ {filename} → {label.upper()} ({conf * 100:.2f}%)")
        print(f"    Probabilities: {dict(zip(class_names, all_probs.round(4)))}")

        # Grad-CAM
        cam = generate_gradcam(img_tensor, pred_idx)
        img_resized = image.resize((224, 224))
        heatmap = plt.get_cmap("jet")(cam)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        overlay = Image.blend(img_resized, Image.fromarray(heatmap), alpha=0.5)

        # Save output
        base = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base}_{label}_gradcam.png")
        overlay.save(output_path)
        print(f"    🔥 Grad-CAM saved to: {output_path}\n")


# Run it
if __name__ == "__main__":
    run_batch(input_folder, max_images)
