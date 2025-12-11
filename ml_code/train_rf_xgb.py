# ml_code/train_rf_xgb.py
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from ml_code.config import EMBEDDINGS_DIR, MODELS_DIR

# Ensure models dir exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_emb(split="train"):
    """
    Loads embeddings saved as joblib/pickle in EMBEDDINGS_DIR.
    Expected filenames: train_embeddings.pkl and val_embeddings.pkl
    The stored object is expected to be a dict with keys:
      - 'feats' : numpy array shape (N, D)
      - 'labels': array-like numeric labels
      - 'paths' : list of image paths (optional)
      - 'classes': list of class names (optional)
    """
    p = EMBEDDINGS_DIR / f"{split}_embeddings.pkl"
    if not p.exists():
        raise SystemExit(f"Embedding file missing: {p}")
    d = joblib.load(p)
    # Support both d being dict with keys or tuple/list
    if isinstance(d, dict):
        feats = d.get("feats") or d.get("X") or d.get("features")
        labels = d.get("labels") or d.get("y")
        paths = d.get("paths") or d.get("filenames") or d.get("files")
        classes = d.get("classes") or d.get("class_names") or None
    else:
        # if saved as tuple (X, y, paths, classes)
        try:
            feats, labels, paths, classes = d
        except Exception as e:
            raise RuntimeError("Unexpected embedding file format; expected dict or tuple (X,y,paths,classes)") from e
    return feats, labels, paths, classes

def train_rf(X, y):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf

def train_xgb(X, y):
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        verbosity=1
    )
    clf.fit(X, y)
    return clf

if __name__ == "__main__":
    X_train, y_train, _, classes = load_emb("train")
    X_val, y_val, val_paths, _ = load_emb("val")

    print("Training RF...")
    rf = train_rf(X_train, y_train)
    print("Training XGB...")
    xgb_clf = train_xgb(X_train, y_train)

    for name, model in [("randomforest", rf), ("xgboost", xgb_clf)]:
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"\n{name} validation accuracy: {acc:.4f}")
        if classes is not None:
            try:
                print(classification_report(y_val, preds, target_names=classes))
            except Exception:
                print("classification_report could not use target_names; printing plain report")
                print(classification_report(y_val, preds))
        else:
            print(classification_report(y_val, preds))
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"Saved {MODELS_DIR / (name + '.joblib')}")

    # save class mapping too (if present)
    if classes is not None:
        joblib.dump(classes, MODELS_DIR / "classes.joblib")
        print("Saved class mapping.")
