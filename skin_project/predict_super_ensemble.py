#!/usr/bin/env python3
"""
predict_super_ensemble.py

Super-ensemble that combines:
 - Fine-tuned SkinCLIP (PyTorch .pth)
 - CNN (Keras .h5)
 - RandomForest (joblib)
 - XGBoost (joblib or Calibrated)
 - optional CLIP zero-shot (transformers)

Features:
 - Aligns class labels
 - Embedding extractor for RF/XGB (MobileNetV2)
 - Per-model smoothing (alpha) and weighted averaging
 - Optional weight tuning on validation set
 - Outputs per-image top-k and final label
"""

import argparse
from pathlib import Path
import json
import numpy as np
import joblib
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# PyTorch for SkinCLIP
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoProcessor

# TF/Keras for CNN
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# sklearn models already saved via joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# PIL for image reading
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Config / Paths
# ----------------------------
ROOT = Path(".")
MODELS_DIR = ROOT / "models"
EMB_DIR = ROOT / "embeddings"   # optional if you already have embeddings
IMG_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Helpers
# ----------------------------
def load_canonical_classes():
    # try models/classes.joblib, otherwise classes from classes.json, else fail
    p = MODELS_DIR / "classes.joblib"
    if p.exists():
        classes = joblib.load(p)
        return list(classes)
    # fallback: try classes.json
    p2 = MODELS_DIR / "classes.json"
    if p2.exists():
        d = json.load(open(p2))
        return [d[str(i)] for i in range(len(d))]
    return None

# ----------------------------
# Embedding model for RF/XGB (MobileNetV2 GAP)
# ----------------------------
def build_emb_model():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    gap = GlobalAveragePooling2D()(base.output)
    return Model(inputs=base.input, outputs=gap)

def preprocess_for_emb(path):
    arr = img_to_array(load_img(path, target_size=IMG_SIZE))
    # MobileNetV2 preprocess: using tf.keras.applications.mobilenet_v2.preprocess_input
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)

def preprocess_for_cnn(path):
    arr = img_to_array(load_img(path, target_size=IMG_SIZE))/255.0
    return np.expand_dims(arr, 0)

# ----------------------------
# SkinCLIP classifier wrapper
# ----------------------------
class SkinCLIPClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim, num_classes):
        super().__init__()
        self.clip = clip_model  # full CLIPModel object
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, pixel_values):
        with torch.no_grad():
            feats = self.clip.get_image_features(pixel_values=pixel_values)
        logits = self.head(feats)
        probs = F.softmax(logits, dim=-1)
        return probs

def load_skinclip_checkpoint(ckpt_path, model_name, device):
    ck = torch.load(ckpt_path, map_location=device)
    classes = ck.get("classes", None)
    # load CLIP full model
    clip_full = CLIPModel.from_pretrained(model_name).to(device)
    # obtain embed dim
    if hasattr(clip_full, "visual_projection"):
        embed_dim = clip_full.visual_projection.out_features
    elif hasattr(clip_full.config, "projection_dim"):
        embed_dim = clip_full.config.projection_dim
    else:
        embed_dim = 768
    model = SkinCLIPClassifier(clip_full, embed_dim, len(classes)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, classes, clip_full, embed_dim

# ----------------------------
# Utilities: normalize, smoothing, align classes
# ----------------------------
def normalize_probs(p):
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0: return np.ones_like(p)/len(p)
    return p / s

def smooth_probs(probs, alpha=1.0):
    probs = np.clip(probs, 1e-12, 1.0)
    p2 = np.power(probs, alpha)
    return normalize_probs(p2)

def align_probs_to_canonical(model_classes, canonical_classes, probs):
    """model_classes: list of labels in that model's order
       canonical_classes: list of canonical labels
       probs: 1D array len(model_classes)
       returns aligned 1D array len(canonical_classes)"""
    aligned = np.zeros(len(canonical_classes), dtype=float)
    for i, lab in enumerate(model_classes):
        if lab in canonical_classes:
            j = canonical_classes.index(lab)
            aligned[j] = float(probs[i])
        else:
            # unknown label - ignore
            pass
    return aligned

# ----------------------------
# Load available models
# ----------------------------
def load_models(config):
    models = {}
    canonical = load_canonical_classes()
    if canonical is None:
        raise SystemExit("No canonical classes found in models/classes.joblib or models/classes.json")

    # CNN
    cnn_path = MODELS_DIR / "cnn_model.h5"
    if cnn_path.exists():
        try:
            cnn = tf.keras.models.load_model(str(cnn_path))
            # load mapping if exists (cnn classes.json)
            cj = MODELS_DIR / "cnn_classes.json"
            if cj.exists():
                cnn_map = json.load(open(cj))
                cnn_class_list = [cnn_map[str(i)] for i in range(len(cnn_map))]
            else:
                # fallback assume same ordering as canonical
                cnn_class_list = canonical
            models["cnn"] = {"type":"cnn","model":cnn,"classes":cnn_class_list}
            print("Loaded CNN:", cnn_path)
        except Exception as e:
            print("Failed to load CNN:", e)

    # sklearn models (randomforest, xgboost or calibrated)
    skl_names = ["randomforest.joblib","xgboost_calib.joblib","xgboost.joblib"]
    for name in skl_names:
        p = MODELS_DIR / name
        if p.exists():
            try:
                m = joblib.load(p)
                # sklearn models rely on canonical class ordering saved in classes.joblib
                key = name.replace(".joblib","")
                models[key] = {"type":"skl","model":m,"classes":canonical}
                print("Loaded sklearn model:", name)
            except Exception as e:
                print("Failed to load", name, e)

    # SkinCLIP fine-tuned .pth
    skpth = MODELS_DIR / "skinclip_finetuned.pth"
    if skpth.exists():
        # default model_name used when fine-tuning
        model_name = config.get("skinclip_backbone","suinleelab/monet")
        try:
            sk_model, sk_classes, clip_full, embed_dim = load_skinclip_checkpoint(str(skpth), model_name, DEVICE)
            models["skinclip"] = {"type":"skinclip","model":sk_model,"classes":sk_classes,"clip_full":clip_full}
            print("Loaded SkinCLIP fine-tuned checkpoint")
        except Exception as e:
            print("Failed to load SkinCLIP checkpoint:", e)

    # optional CLIP zero-shot (we include only if user wants)
    if config.get("use_clip_zeroshot", False):
        try:
            clip_name = config.get("clip_model_name","openai/clip-vit-base-patch32")
            clip_model = CLIPModel.from_pretrained(clip_name)
            clip_processor = AutoProcessor.from_pretrained(clip_name)
            models["clip_zeroshot"] = {"type":"clip_zero","model":clip_model,"processor":clip_processor}
            print("Loaded CLIP zero-shot model:", clip_name)
        except Exception as e:
            print("Failed to load CLIP zero-shot:", e)

    return models, canonical

# ----------------------------
# Prediction helpers
# ----------------------------
def predict_cnn(cnn_info, img_path):
    cnn = cnn_info["model"]
    class_list = cnn_info["classes"]
    arr = preprocess_for_cnn(img_path)
    preds = cnn.predict(arr, verbose=0)[0]
    aligned = align_probs_to_canonical(class_list, canonical_classes, preds)
    return normalize_probs(aligned)

def predict_skl(skl_info, img_path, emb_model):
    model = skl_info["model"]
    class_list = skl_info["classes"]
    feat = emb_model.predict(preprocess_for_emb(img_path))
    try:
        probs = model.predict_proba(feat)[0]
    except Exception:
        # fallback to predict output
        pr = model.predict(feat)
        probs = np.zeros(len(class_list))
        probs[int(pr[0])] = 1.0
    aligned = align_probs_to_canonical(class_list, canonical_classes, probs)
    return normalize_probs(aligned)

def predict_skinclip(skin_info, img_path, processor=None):
    model = skin_info["model"]
    classes = skin_info["classes"]
    # prepare pixel_values using processor
    proc_name = config.get("skinclip_backbone","suinleelab/monet")
    proc = AutoProcessor.from_pretrained(proc_name)
    pil = Image.open(img_path).convert("RGB")
    inputs = proc(images=pil, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
    with torch.no_grad():
        probs = model(inputs["pixel_values"]).detach().cpu().numpy()[0]
    aligned = align_probs_to_canonical(classes, canonical_classes, probs)
    return normalize_probs(aligned)

def predict_clip_zeroshot(clip_info, img_path, labels):
    # compute image feat and text sims; return softmax over labels
    proc = clip_info["processor"]
    model = clip_info["model"]
    pil = Image.open(img_path).convert("RGB")
    inputs_img = proc(images=pil, return_tensors="pt")["pixel_values"]
    text_inputs = proc(text=labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        img_emb = model.get_image_features(inputs_img)
        text_emb = model.get_text_features(**text_inputs)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    sims = (img_emb @ text_emb.T).cpu().numpy()[0]
    # softmax with temp
    temp = 0.07
    ex = np.exp((sims - sims.max())/temp)
    probs = ex / ex.sum()
    aligned = align_probs_to_canonical(labels, canonical_classes, probs)
    return normalize_probs(aligned)

# ----------------------------
# Ensemble & weighting
# ----------------------------
def weighted_average(prob_list, weights):
    # prob_list: list of arrays length C
    # weights: list of same len
    stacked = np.stack(prob_list, axis=0)
    w = np.array(weights)[:,None]
    avg = (stacked * w).sum(axis=0) / (w.sum())
    return normalize_probs(avg)

def eval_on_val(models_dict, canonical, emb_model, val_dir, config, try_weights=None):
    # returns accuracy and optionally detailed results
    X = []
    y_true = []
    # iterate val folder structure expected: val/<class>/*.jpg
    val_path = Path(val_dir)
    for cls in sorted([p.name for p in val_path.iterdir() if p.is_dir()]):
        for f in (val_path/cls).glob("*"):
            if f.suffix.lower() not in (".jpg",".jpeg",".png",".webp"): continue
            y_true.append(canonical.index(cls))
            X.append(f)
    preds = []
    prob_preds = []
    # default weights or provided
    if try_weights is None:
        # default equal weighting across present models
        present = list(models_dict.keys())
        weights = [1.0]*len(present)
    else:
        weights = try_weights

    for img in tqdm(X, desc="Val preds"):
        per_model_probs = []
        model_keys = []
        for k,info in models_dict.items():
            try:
                if info["type"]=="cnn":
                    p = predict_cnn(info, img)
                elif info["type"]=="skl":
                    p = predict_skl(info, img, emb_model)
                elif info["type"]=="skinclip":
                    p = predict_skinclip(info, img)
                elif info["type"]=="clip_zero":
                    p = predict_clip_zeroshot(info, img, canonical)
                else:
                    continue
                # smoothing: use config alphas
                alpha = config.get("alphas",{}).get(k,1.0)
                p = smooth_probs(p, alpha)
                per_model_probs.append(p)
                model_keys.append(k)
            except Exception as e:
                print("Predict fail",k,e)
        if len(per_model_probs)==0:
            prob = np.ones(len(canonical))/len(canonical)
        else:
            # align weights order with model_keys
            if try_weights is None:
                w = [1.0]*len(per_model_probs)
            else:
                # assume order of models_dict keys; build weights accordingly
                # simple approach: use weights in same order as models_dict keys
                w = []
                for mk in models_dict.keys():
                    if mk in model_keys:
                        idx = list(models_dict.keys()).index(mk)
                        w.append(weights[idx])
                if len(w) != len(per_model_probs):
                    # fallback
                    w = [1.0]*len(per_model_probs)
            prob = weighted_average(per_model_probs, w)
        prob_preds.append(prob)
        preds.append(int(np.argmax(prob)))
    acc = accuracy_score(y_true, preds)
    return acc, y_true, preds, prob_preds

def grid_search_weights(models_dict, canonical, emb_model, val_dir, config):
    # simple grid search for up to 3 models; if more, do coarse search
    keys = list(models_dict.keys())
    n = len(keys)
    print("Grid search across models:", keys)
    # ranges
    grid = [0.0, 0.2, 0.5, 1.0, 2.0]
    best = (None, -1.0)
    if n <= 4:
        # full grid
        from itertools import product
        for comb in product(grid, repeat=n):
            if sum(comb) == 0: continue
            acc, *_ = eval_on_val(models_dict, canonical, emb_model, val_dir, config, try_weights=list(comb))
            if acc > best[1]:
                best = (comb, acc)
        return best
    else:
        # coarse: equal weights
        return ([1.0]*n, eval_on_val(models_dict, canonical, emb_model, val_dir, config, try_weights=[1.0]*n)[0])

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="single image path")
    parser.add_argument("--folder", help="folder to run predictions on")
    parser.add_argument("--out_csv", help="save predictions CSV", default=None)
    parser.add_argument("--tune_weights", action="store_true", help="grid search weights on val set")
    parser.add_argument("--val_dir", help="validation folder (data/val)", default="data/val")
    parser.add_argument("--skinclip_backbone", default="suinleelab/monet")
    parser.add_argument("--use_clip_zeroshot", action="store_true")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    # configuration
    config = {
        "skinclip_backbone": args.skinclip_backbone,
        "use_clip_zeroshot": args.use_clip_zeroshot,
        "alphas": {
            "cnn": 1.0,
            "randomforest": 0.9,
            "xgboost": 0.7,
            "xgboost_calib": 0.9,
            "skinclip": 1.0,
            "clip_zeroshot": 0.9
        }
    }

    # load models
    models_dict, canonical_classes = load_models(config)

    if len(models_dict)==0:
        print("No models loaded. Place at least one model in models/")
        sys.exit(1)

    # build embedding extractor if any sklearn models exist
    emb_model = None
    if any(info["type"]=="skl" for info in models_dict.values()):
        print("Building embedding extractor (MobileNetV2)...")
        emb_model = build_emb_model()

    # tune weights on val if requested
    if args.tune_weights:
        print("Tuning ensemble weights on validation set:", args.val_dir)
        best_comb, best_acc = grid_search_weights(models_dict, canonical_classes, emb_model, args.val_dir, config)
        print("Best weights:", best_comb, "Acc:", best_acc)
        # store chosen weights in config for prediction
        weights = best_comb
    else:
        # default equal weights across present models
        weights = [1.0]*len(models_dict)

    # helper: predict for one image
    def predict_one(img_path):
        per_model = []
        names = []
        for k, info in models_dict.items():
            try:
                if info["type"]=="cnn":
                    p = predict_cnn(info, img_path)
                elif info["type"]=="skl":
                    p = predict_skl(info, img_path, emb_model)
                elif info["type"]=="skinclip":
                    p = predict_skinclip(info, img_path)
                elif info["type"]=="clip_zero":
                    p = predict_clip_zeroshot(info, img_path, canonical_classes)
                else:
                    continue
                p = smooth_probs(p, config["alphas"].get(k,1.0))
                per_model.append(p)
                names.append(k)
                # print per-model top3
                top3 = np.argsort(p)[-args.topk:][::-1]
                print(f"=== {k.upper()} top-{args.topk} ===")
                for i in top3:
                    print(f"  - {canonical_classes[i]} ({p[i]*100:.2f}%)")
            except Exception as e:
                print("Model prediction failed:", k, e)
        if len(per_model) == 0:
            print("No model produced predictions.")
            return None
        # build weights aligned to names
        w = []
        for mk in models_dict.keys():
            if mk in names:
                # find index in models_dict keys
                idx = list(models_dict.keys()).index(mk)
                w.append(weights[idx])
        if len(w) != len(per_model):
            # fallback equal
            w = [1.0]*len(per_model)
        ensemble = weighted_average(per_model, w)
        top = int(np.argmax(ensemble))
        print("\n=====================================")
        print(f"Ensemble detected disease: {canonical_classes[top]} ({ensemble[top]*100:.2f}% confidence)")
        print("Ensemble top-3:")
        for i in np.argsort(ensemble)[-args.topk:][::-1]:
            print(f"  - {canonical_classes[i]} ({ensemble[i]*100:.2f}%)")
        print("=====================================\n")

        # >>> Clean one-line final output <<<
        print(f"Final Predicted Disease → {canonical_classes[top]}\n")

        return canonical_classes[top], float(ensemble[top]), ensemble

    # run single image or folder
    if args.img:
        predict_one(args.img)
    elif args.folder:
        files = [f for f in Path(args.folder).rglob("*") if f.suffix.lower() in (".jpg",".jpeg",".png",".webp")]
        rows = []
        for f in tqdm(files, desc="Predicting"):
            res = predict_one(f)
            if res is None:
                continue
            label, conf, probs = res
            row = {"image":str(f), "label":label, "confidence":conf}
            for i,p in enumerate(probs):
                row[f"prob_{canonical_classes[i]}"] = p
            rows.append(row)
        if args.out_csv:
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
            print("Saved to", args.out_csv)
    else:
        print("Provide --img or --folder")
