# prepare_dataset.py
import random
import shutil
from pathlib import Path
from config import SRC_IMAGES, DATA_DIR, RANDOM_SEED, VAL_SPLIT

random.seed(RANDOM_SEED)

def make_split():
    if not SRC_IMAGES.exists():
        raise SystemExit(f"SRC_IMAGES not found: {SRC_IMAGES}")
    classes = [p.name for p in SRC_IMAGES.iterdir() if p.is_dir()]
    if not classes:
        raise SystemExit(f"No class subfolders found under {SRC_IMAGES}")

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    for d in (train_dir, val_dir):
        d.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        cls_src = SRC_IMAGES / cls
        files = [p.name for p in cls_src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not files:
            print(f"Warning: no images for class {cls} in {cls_src}")
            continue
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_SPLIT))
        val_files = files[:n_val]
        train_files = files[n_val:]

        train_cls_dir = train_dir / cls
        val_cls_dir = val_dir / cls
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        val_cls_dir.mkdir(parents=True, exist_ok=True)

        for fname in train_files:
            shutil.copy2(cls_src / fname, train_cls_dir / fname)
        for fname in val_files:
            shutil.copy2(cls_src / fname, val_cls_dir / fname)

        print(f"{cls}: total={len(files)} train={len(train_files)} val={len(val_files)}")

    print("Done splitting. Check", train_dir, "and", val_dir)

if __name__ == "__main__":
    make_split()
