# predict_skinclip.py
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import CLIPModel, AutoProcessor
import torch.nn.functional as F
from tqdm import tqdm

class SkinCLIPClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim, num_classes):
        super().__init__()
        self.clip_model = clip_model
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        # uses CLIPModel.get_image_features
        with torch.no_grad():
            feat = self.clip_model.get_image_features(pixel_values=pixel_values)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
        logits = self.classifier(feat)
        probs = F.softmax(logits, dim=-1)
        return probs

def load_checkpoint(checkpoint_path, model_name, device):
    ck = torch.load(checkpoint_path, map_location=device)
    classes = ck.get("classes")
    # load full clip model
    clip_full = CLIPModel.from_pretrained(model_name).to(device)
    # determine embed_dim robustly
    if hasattr(clip_full, "visual_projection"):
        embed_dim = clip_full.visual_projection.out_features
    elif hasattr(clip_full.config, "projection_dim"):
        embed_dim = clip_full.config.projection_dim
    else:
        # fallback
        embed_dim = 768
    model = SkinCLIPClassifier(clip_full, embed_dim, len(classes)).to(device)
    model.load_state_dict(ck["model_state"])  # saved earlier
    model.eval()
    return model, classes

def preprocess_image(processor, image_path, device):
    pil = Image.open(image_path).convert("RGB")
    inputs = processor(images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    return pixel_values

def predict_image(model, processor, image_path, classes, device, topk=3):
    px = preprocess_image(processor, image_path, device)
    probs = model(px)  # (1, C)
    probs = probs.detach().cpu().numpy()[0]
    idxs = np.argsort(probs)[::-1][:topk]
    return [(classes[i], float(probs[i])) for i in idxs]

def predict_folder(model, processor, folder, classes, device, topk=3, out_csv=None):
    p = Path(folder)
    imgs = [f for f in p.rglob("*") if f.suffix.lower() in (".jpg",".jpeg",".png",".webp")]
    rows = []
    for img in tqdm(imgs, desc="Predicting"):
        preds = predict_image(model, processor, img, classes, device, topk=topk)
        row = {"image": str(img)}
        for i,(lbl,prob) in enumerate(preds, start=1):
            row[f"top{i}_label"] = lbl
            row[f"top{i}_prob"] = prob
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/skinclip_finetuned.pth")
    parser.add_argument("--model_name", default="suinleelab/monet")  # same model_name used for backbone
    parser.add_argument("--image")
    parser.add_argument("--folder")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    model, classes = load_checkpoint(str(ckpt), args.model_name, device)
    processor = AutoProcessor.from_pretrained(args.model_name)

    if args.image:
        preds = predict_image(model, processor, args.image, classes, device, topk=args.topk)
        print("Predictions for:", args.image)
        for lbl, p in preds:
            print(f"  - {lbl} ({p*100:.2f}%)")
    elif args.folder:
        df = predict_folder(model, processor, args.folder, classes, device, topk=args.topk, out_csv=args.out_csv)
        print(df.head())
    else:
        print("Provide --image or --folder")
