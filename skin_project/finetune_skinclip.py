import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPModel, AutoProcessor, get_scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ------------------------------------
# Dataset using Albumentations
# ------------------------------------
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.samples = []
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        for idx, cls in enumerate(self.classes):
            for f in (self.root/cls).glob("*"):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    self.samples.append((str(f), idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, label = self.samples[i]
        image = np.array(Image.open(img_path).convert("RGB"))
        image = self.transform(image=image)["image"]
        return image.float(), label


def create_transforms():
    train_tf = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ])
    return train_tf, val_tf


# ------------------------------------
# Classification head on top of SkinCLIP
# ------------------------------------
class SkinCLIPClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim, num_classes):
        super().__init__()
        self.clip_model = clip_model
        for p in self.clip_model.parameters():
            p.requires_grad = False  # Freeze backbone

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        with torch.no_grad():
            feat = self.clip_model.get_image_features(pixel_values=pixel_values)

        feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize
        return self.classifier(feat)


# ------------------------------------
# Training loop
# ------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor + model
    processor = AutoProcessor.from_pretrained(args.model_name)
    full_model = CLIPModel.from_pretrained(args.model_name).to(device)

    embed_dim = full_model.visual_projection.out_features

    # Build datasets
    train_tf, val_tf = create_transforms()
    train_ds = AlbumentationsDataset(Path(args.data_dir)/"train", train_tf)
    val_ds = AlbumentationsDataset(Path(args.data_dir)/"val", val_tf)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    model = SkinCLIPClassifier(full_model, embed_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    steps = len(train_dl) * args.epochs
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=int(0.1*steps), num_training_steps=steps)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            # Preprocess as CLIP pixel_values
            pixel_values = xb
            logits = model(pixel_values)

            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=loss.item())

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        print(f"Validation Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes
            }, args.output)
            print("Saved best model →", args.output)

    print("Training complete. Best Accuracy:", best_acc)


# ------------------------------------
# CLI
# ------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", default="models/skinclip_finetuned.pth")
    args = p.parse_args()

    train(args)
