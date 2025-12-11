# zero_shot_clip_improved.py
"""
Improved CLIP zero-shot for skin disease.
Usage:
  python zero_shot_clip_improved.py --image "path/to/img.jpg" --topk 5
  python zero_shot_clip_improved.py --folder "Datasets/raw_images" --topk 3 --out_csv out.csv
  python zero_shot_clip_improved.py --labels_file derm_labels.txt
"""

from pathlib import Path
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import argparse
from tqdm import tqdm
import pandas as pd
import json
import sys

# Default dermatology labels (you can replace or add lines in a file)
DEFAULT_LABELS = [
    "melanoma",
    "basal cell carcinoma",
    "squamous cell carcinoma",
    "nevus (mole)",
    "seborrheic keratosis",
    "psoriasis",
    "eczema",
    "dermatitis",
    "vitiligo",
    "acne",
    "herpes",
    "tinea (ringworm)",
    "impetigo",
    "urticaria (hives)",
    "rosacea",
    "bullous pemphigoid",
    "pityriasis rosea",
    "lichen planus",
    "pityriasis versicolor",
    "scabies"
]

# Templates (include dermatoscopic, clinical, and plain)
TEMPLATES = [
    "a dermatoscopic image of {}",
    "a clinical close-up photo of {}",
    "a photo of {}",
    "a close-up photo of {}",
    "a clinical photo showing {}",
    "an image showing {}",
    "{}"
]

# device
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_labels(labels_file=None):
    if labels_file:
        p = Path(labels_file)
        if not p.exists():
            raise SystemExit(f"Labels file not found: {labels_file}")
        labels = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        labels = DEFAULT_LABELS
    return labels

def build_prompts(labels):
    # For each label, produce several textual prompts using templates
    prompts = []
    label_to_prompt_idxs = []
    for lbl in labels:
        start = len(prompts)
        for t in TEMPLATES:
            prompts.append(t.format(lbl))
        end = len(prompts)
        label_to_prompt_idxs.append((start, end))
    return prompts, label_to_prompt_idxs

def compute_text_embeddings(model, processor, texts, device_str, batch=64):
    model.to(device_str); model.eval()
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            inputs = processor(text=chunk, return_tensors="pt", padding=True).to(device_str)
            t_emb = model.get_text_features(**inputs)
            t_emb = t_emb / t_emb.norm(p=2, dim=-1, keepdim=True)
            all_emb.append(t_emb.cpu().numpy())
    return np.vstack(all_emb)  # shape (N_texts, D)

def compute_image_embeddings(model, processor, images, device_str, batch=16):
    model.to(device_str); model.eval()
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i:i+batch]
            inputs = processor(images=chunk, return_tensors="pt")["pixel_values"].to(device_str)
            i_emb = model.get_image_features(inputs)
            i_emb = i_emb / i_emb.norm(p=2, dim=-1, keepdim=True)
            all_emb.append(i_emb.cpu().numpy())
    return np.vstack(all_emb)  # (N_images, D)

def cosine_sim(a, b):
    return np.dot(a, b.T)

def scores_from_sims(sims, temperature=0.07):
    # sims: (N_images, N_labels) raw cosine similarities
    # apply temperature-softmax across labels for each image
    # higher temperature -> softer ; lower -> sharper. 0.07 is CLIP default-ish
    sims = np.asarray(sims, dtype=float)
    # subtract max for numerical stability
    exp = np.exp((sims - sims.max(axis=1, keepdims=True)) / temperature)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs

def predict_single(model, processor, img_path, labels, topk=3, temperature=0.07):
    pil = Image.open(img_path).convert("RGB")
    prompts, idx_map = build_prompts(labels)
    # compute image and text embeddings
    img_emb = compute_image_embeddings(model, processor, [pil], device())[0:1]  # (1,D)
    text_emb = compute_text_embeddings(model, processor, prompts, device())      # (N_prompts, D)
    # aggregate text embeddings per label by averaging
    label_embs = []
    for (s,e) in idx_map:
        emb = text_emb[s:e].mean(axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        label_embs.append(emb)
    label_embs = np.vstack(label_embs)  # (N_labels, D)
    sims = cosine_sim(img_emb, label_embs)  # (1, N_labels)
    probs = scores_from_sims(sims, temperature=temperature)  # (1, N_labels)
    probs = probs[0]
    ranked = np.argsort(probs)[::-1]
    results = [(labels[i], float(probs[i]), float(sims[0,i])) for i in ranked[:topk]]
    return results, probs, sims

def predict_folder(model, processor, folder, labels, topk=3, out_csv=None, temperature=0.07):
    p = Path(folder)
    files = [f for f in p.rglob("*") if f.suffix.lower() in (".jpg",".jpeg",".png",".webp")]
    rows = []
    prompts, idx_map = build_prompts(labels)
    # precompute text embeddings once
    text_emb = compute_text_embeddings(model, processor, prompts, device())
    # compute label embeddings once
    label_embs = []
    for (s,e) in idx_map:
        emb = text_emb[s:e].mean(axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        label_embs.append(emb)
    label_embs = np.vstack(label_embs)
    for f in tqdm(files, desc="Predicting"):
        pil = Image.open(f).convert("RGB")
        img_emb = compute_image_embeddings(model, processor, [pil], device())[0:1]
        sims = cosine_sim(img_emb, label_embs)[0]
        probs = scores_from_sims(sims.reshape(1,-1), temperature=temperature)[0]
        ranked = np.argsort(probs)[::-1]
        row = {"image": str(f)}
        for i in range(min(len(ranked),5)):
            idx = ranked[i]
            row[f"top{i+1}_label"] = labels[idx]
            row[f"top{i+1}_score"] = float(probs[idx])
            row[f"top{i+1}_sim"] = float(sims[idx])
        rows.append(row)
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="single image path")
    parser.add_argument("--folder", help="folder of images")
    parser.add_argument("--labels_file", help="optional labels file")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--temperature", type=float, default=0.07, help="softmax temperature")
    args = parser.parse_args()

    if not args.image and not args.folder:
        print("Provide --image or --folder")
        sys.exit(1)

    labels = load_labels(args.labels_file)
    print(f"Using {len(labels)} labels. Device: {device()}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if args.image:
        res, probs, sims = predict_single(model, processor, args.image, labels, topk=args.topk, temperature=args.temperature)
        print("\nPredictions for:", args.image)
        for i, (lbl, score, sim) in enumerate(res, start=1):
            print(f"  Top{i}: {lbl} (prob={score:.4f}, sim={sim:.4f})")
    else:
        df = predict_folder(model, processor, args.folder, labels, topk=args.topk, out_csv=args.out_csv, temperature=args.temperature)
        print(df.head())

if __name__ == "__main__":
    main()
