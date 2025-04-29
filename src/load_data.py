# load_data.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil

# Paths
DATA_DIR = "data"
IMG_DIRS = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]
CSV_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# Load metadata
df = pd.read_csv(CSV_PATH)

# Map diagnostic labels to binary classes
label_map = {
    "mel": "malignant",  # melanoma
    "nv": "benign",  # melanocytic nevi
    "bkl": "benign",  # benign keratosis
    "bcc": "malignant",  # basal cell carcinoma
    "akiec": "malignant",  # actinic keratoses
    "vasc": "benign",  # vascular lesions
    "df": "benign",  # dermatofibroma
}
df["label"] = df["dx"].map(label_map)


# Add full image paths
def find_image_path(image_id):
    for dir_ in IMG_DIRS:
        path = os.path.join(dir_, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None


df["image_path"] = df["image_id"].apply(find_image_path)
df = df.dropna(subset=["image_path"])

# Filter to just benign vs malignant
df = df[df["label"].isin(["benign", "malignant"])]

# Split into train/val
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

print(f"Total images: {len(df)}")
print(f"Training: {len(train_df)}, Validation: {len(val_df)}")

# Optional: Save splits
train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
