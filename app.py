import os
import cv2
import torch
import torch.nn as nn
import timm

from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from werkzeug.utils import secure_filename


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "glaucoma_model3pytho.pth")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# MODEL
# =========================================================

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False
)

model.classifier = nn.Linear(
    model.classifier.in_features,
    1
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =========================================================
# CHECK FILE TYPE
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# PREDICTION
# =========================================================

def predict_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(image)

    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probability_normal = torch.sigmoid(
            output
        ).item()


    # ImageFolder class mapping:
    #
    # glaucoma = 0
    # normal   = 1

    if probability_normal >= 0.5:

        result = "Normal"
        confidence = probability_normal

    else:

        result = "Glaucoma"
        confidence = 1 - probability_normal


    return {
        "result": result,
        "confidence": round(
            confidence * 100,
            2
        )
    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image was uploaded."
        }), 400


    file = request.files["image"]


    if file.filename == "":

        return jsonify({
            "error": "Please select an image."
        }), 400


    if not allowed_file(file.filename):

        return jsonify({
            "error":
            "Please upload JPG, JPEG, PNG, BMP, TIF or TIFF."
        }), 400


    filename = secure_filename(
        file.filename
    )

    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(image_path)


    prediction = predict_image(
        image_path
    )


    if prediction is None:

        return jsonify({
            "error":
            "The uploaded image could not be read."
        }), 400


    return jsonify(prediction)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print("\nGlaucoma Detection System")
    print("Open: http://127.0.0.1:5000\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )