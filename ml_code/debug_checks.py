# debug_checks.py
import joblib, json, os, random
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ml_code.config import MODELS_DIR, EMB_DIR, DATA_DIR, IMG_SIZE
import numpy as np
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = Path(MODELS_DIR)
EMB_DIR = Path(EMB_DIR)
DATA_DIR = Path(DATA_DIR)

print("MODELS_DIR:", MODELS_DIR.resolve())
print("EMB_DIR:", EMB_DIR.resolve())
print("DATA_DIR:", DATA_DIR.resolve())

# 1) Show classes saved by different files
classes_json = MODELS_DIR / "classes.json"
classes_job = MODELS_DIR / "classes.joblib"
emb_train = EMB_DIR / "train_embeddings.pkl"

if classes_json.exists():
    print("\nclasses.json contents (index->label):")
    print(json.load(open(classes_json)))

if classes_job.exists():
    print("\nclasses.joblib contents (list):")
    print(joblib.load(classes_job))

if emb_train.exists():
    d = joblib.load(str(emb_train))
    print("\ntrain_embeddings.pkl classes from embeddings:", d.get("classes")[:10])
    print("train embeddings size:", d.get("feats").shape)

# 2) Check class folder counts
print("\nDataset class counts (train):")
for cls in sorted([d for d in (DATA_DIR/"train").iterdir() if d.is_dir()]):
    images = list(cls.glob("*"))
    print(f"  {cls.name}: {len(images)}")

# 3) Quick test: run model prediction on a few *training* images and see whether they predict correctly.
# Use predict.py logic but simplified for speed.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
import joblib

def load_models():
    cnn = None
    if (MODELS_DIR/"cnn_model.h5").exists():
        try:
            cnn = tf.keras.models.load_model(str(MODELS_DIR/"cnn_model.h5"))
        except Exception as e:
            print("Failed to load cnn:", e)
    rf = None
    xgb = None
    if (MODELS_DIR/"randomforest.joblib").exists():
        rf = joblib.load(str(MODELS_DIR/"randomforest.joblib"))
    if (MODELS_DIR/"xgboost.joblib").exists():
        xgb = joblib.load(str(MODELS_DIR/"xgboost.joblib"))
    return cnn, rf, xgb

cnn, rf, xgb = load_models()
# embedding model
emb_model = None
if rf is not None or xgb is not None:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0],IMG_SIZE[1],3))
    gap = GlobalAveragePooling2D()(base.output)
    emb_model = Model(inputs=base.input, outputs=gap)

# sample a few images from each class
print("\nSample predictions on few train images:")
for cls_dir in sorted((DATA_DIR/"train").iterdir()):
    imgs = list(cls_dir.glob("*"))
    if not imgs: continue
    sample = random.choice(imgs)
    print("\nClass folder:", cls_dir.name, "sample:", sample.name)
    # CNN
    if cnn is not None:
        img = load_img(str(sample), target_size=IMG_SIZE)
        arr = img_to_array(img)/255.0
        pred = cnn.predict(np.expand_dims(arr,0))[0]
        print("  CNN top:", np.argmax(pred), f"{max(pred):.3f}")
    # RF/XGB
    if emb_model is not None:
        img = load_img(str(sample), target_size=IMG_SIZE)
        arr = img_to_array(img)
        arr = preprocess_input(arr)
        feat = emb_model.predict(np.expand_dims(arr,0), verbose=0)
        if rf is not None:
            p = rf.predict_proba(feat)[0]; print("  RF top:", np.argmax(p), f"{max(p):.3f}")
        if xgb is not None:
            p = xgb.predict_proba(feat)[0]; print("  XGB top:", np.argmax(p), f"{max(p):.3f}")

print("\nDone quick checks.")
