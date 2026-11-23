import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import cv2
import numpy as np

# -------------------
# Load Model
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 1)
model.load_state_dict(torch.load("glaucoma_model3pytho.pth", map_location=device))
model = model.to(device)
model.eval()

# -------------------
# Transform
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------
# Predict Function
# -------------------
def predict(image_path):
    print("Image:", image_path)

    # Load image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print("❌ ERROR: Could not read image. Check path.")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image → PIL
    img_pil = Image.fromarray(img_rgb)

    # Apply transforms
    tensor_img = transform(img_pil).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(tensor_img)
        prob = torch.sigmoid(output).item()

    print(f"Probability of Glaucoma: {prob:.4f}")

    if prob > 0.5:
        print("Prediction: GLAUCOMA")
    else:
        print("Prediction: NORMAL")

# -------------------
# Test with one image
# -------------------
predict(r"C:\Users\ASUS\Desktop\Final project\Dataset_Split\train\glaucoma\drishtiGS_028.png")