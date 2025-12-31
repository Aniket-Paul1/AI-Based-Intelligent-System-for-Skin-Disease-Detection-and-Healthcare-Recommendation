import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler

from ml_code.vit.vit_model import ViTDiseaseClassifier

# ---------------- CONFIG ----------------
DATA_DIR = "data/train"     # <-- your dataset folder
MODEL_OUT = "artifacts/models/vit_model.pth"
EPOCHS = 15
BATCH_SIZE = 16
LR = 3e-4
NUM_CLASSES = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATA ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# ---------------- MODEL ----------------
model = ViTDiseaseClassifier(NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler()

# ---------------- TRAIN ----------------
model.train()
for epoch in range(EPOCHS):
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# ---------------- SAVE ----------------
torch.save(model.state_dict(), MODEL_OUT)
print("ViT model saved:", MODEL_OUT)
