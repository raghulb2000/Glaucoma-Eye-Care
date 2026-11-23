import os
import torch
import torch.nn as nn
import timm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.join(BASE_DIR, "Dataset", "val")
MODEL_PATH = os.path.join(BASE_DIR, "glaucoma_model3pytho.pth")

# -----------------------
# Validation Dataset
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

val_ds = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

val_loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False
)

print("Classes:", val_ds.classes)
print("Validation images:", len(val_ds))

# -----------------------
# Load Model
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False
)

model.classifier = nn.Linear(
    model.classifier.in_features,
    1
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()

print("Model loaded successfully")
print("Device:", device)

# -----------------------
# Evaluation
# -----------------------
true_labels = []
predicted_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs).squeeze(1)

        predictions = (
            probabilities >= 0.5
        ).long()

        true_labels.extend(labels.numpy())
        predicted_labels.extend(
            predictions.cpu().numpy()
        )

# -----------------------
# Results
# -----------------------
accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels
)

recall = recall_score(
    true_labels,
    predicted_labels
)

f1 = f1_score(
    true_labels,
    predicted_labels
)

print("\n----- MODEL RESULTS -----")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        true_labels,
        predicted_labels
    )
)

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=val_ds.classes
    )
)