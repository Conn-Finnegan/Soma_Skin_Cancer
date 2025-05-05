import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import coremltools as ct

# --- CONFIG ---
model_path = "models/skin_cancer_resnet18_v1.pt"
output_path = "models/skin_cancer_resnet18_v1.mlmodel"
class_labels = ["benign", "malignant"]
input_shape = (1, 3, 224, 224)

# --- Load PyTorch Model ---
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# --- Convert to TorchScript ---
example_input = torch.rand(input_shape)
traced_model = torch.jit.trace(model, example_input)

# --- Convert to Core ML ---
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.ImageType(name="input", shape=input_shape, scale=1 / 255.0, bias=[0, 0, 0])
    ],
    classifier_config=ct.ClassifierConfig(class_labels),
)

# --- Save Model ---
mlmodel.save(output_path)
print(f"✅ Core ML model saved to: {output_path}")
