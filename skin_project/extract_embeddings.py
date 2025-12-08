# extract_embeddings.py
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from config import DATA_DIR, EMB_DIR, IMG_SIZE

EMB_DIR.mkdir(parents=True, exist_ok=True)

def get_paths_and_labels(root_dir: Path):
    classes = sorted([p.name for p in root_dir.iterdir() if p.is_dir()])
    paths = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = root_dir / cls
        for p in cls_dir.iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                paths.append(str(p))
                labels.append(idx)
    return paths, labels, classes

def load_embedding_model():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    gap = GlobalAveragePooling2D()(base.output)
    model = Model(inputs=base.input, outputs=gap)
    return model

def extract(paths, model):
    feats = []
    for p in tqdm(paths):
        img = load_img(p, target_size=IMG_SIZE)
        arr = img_to_array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        v = model.predict(arr, verbose=0)[0]
        feats.append(v)
    return np.array(feats)

if __name__ == "__main__":
    for split in ["train", "val"]:
        root = DATA_DIR / split
        if not root.exists():
            raise SystemExit(f"Missing {root}. Run prepare_dataset.py")
        paths, labels, classes = get_paths_and_labels(root)
        if not paths:
            print(f"No images in {root}, skipping")
            continue
        print(f"{split}: found {len(paths)} images, {len(classes)} classes")
        model = load_embedding_model()
        feats = extract(paths, model)
        joblib.dump({"paths": paths, "labels": labels, "feats": feats, "classes": classes},
                    EMB_DIR / f"{split}_embeddings.pkl")
        print(f"Saved {EMB_DIR / (split + '_embeddings.pkl')}")
    print("Embedding extraction finished.")
