import os
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.model import ShapeCNN

def classify_image(image_path, shape="square", models_dir="models"):
    """Classifies an input image as target shape vs non-target shape."""
    model_path = os.path.join(models_dir, f"{shape}_cnn.pth")
    if not os.path.exists(model_path):
        model_path = f"{shape}_cnn.pth"

    if not os.path.exists(model_path):
        print(f"Error: Trained model weights '{model_path}' not found!")
        return

    model = ShapeCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    img = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output_logit = model(img_tensor)
        confidence = torch.sigmoid(output_logit).item() * 100

    print(f"\n--- CNN Classification Result ({shape.upper()}) ---")
    print(f"Image File: {image_path}")
    if confidence >= 50.0:
        print(f"Result: [{shape.upper()}] Confidence: {confidence:.2f}%")
    else:
        print(f"Result: [NOT A {shape.upper()}] {shape.capitalize()} Confidence: {confidence:.2f}%")
