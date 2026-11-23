import torch
import timm
import torch.nn as nn

IMG_SIZE = 224

def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load same architecture
    model = timm.create_model("efficientnet_b0", pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device