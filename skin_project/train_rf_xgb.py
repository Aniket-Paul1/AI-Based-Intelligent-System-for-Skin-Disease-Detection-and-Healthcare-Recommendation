# train_rf_xgb.py
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from config import EMB_DIR, MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_emb(split="train"):
    p = EMB_DIR / f"{split}_embeddings.pkl"
    if not p.exists():
        raise SystemExit(f"Embedding file missing: {p}")
    d = joblib.load(p)
    return d["feats"], d["labels"], d["paths"], d["classes"]

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
        print(classification_report(y_val, preds, target_names=classes))
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"Saved {MODELS_DIR / (name + '.joblib')}")

    joblib.dump(classes, MODELS_DIR / "classes.joblib")
    print("Saved class mapping.")
