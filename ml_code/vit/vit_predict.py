import torch
import numpy as np
import cv2
from torchvision import transforms
from ml_code.vit.vit_model import ViTDiseaseClassifier

CLASSES = [
    "Acne",
    "Rosacea",
    "Herpes",
    "Dermatitis",
    "Psoriasis",
    "Vitiligo",
    "Basal Cell Carcinoma",
    "Melanoma"
]

MODEL_PATH = "artifacts/models/vit_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViTDiseaseClassifier(len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def vit_predict(image_np):
    img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))

    return {
        "label": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    }
