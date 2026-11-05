"""
train.py - train the TF-IDF + Random Forest pipeline (the ML arm of the hybrid system).

Usage (from the project root, with .venv activated):
    python src/train.py
    python src/train.py --data data/my_dataset.csv --out models/

Pipeline:
    ColumnTransformer
      - TF-IDF (100 uni/bi-gram features, min_df=2) on symptom_text
      - passthrough: 7 binary flags + 3 continuous metrics + 2 label-encoded categoricals
    RandomForestClassifier(200 trees, min_samples_split=5, min_samples_leaf=2,
                           class_weight="balanced", random_state=42)

Evaluation (printed and saved to models/training_report.json):
    - stratified 80/20 train/test split (seed 42)
    - 5-fold stratified cross-validation on the TRAINING set
      (the original script cross-validated on the 240-sample test set only)
    - test accuracy, classification report and confusion matrix
"""
import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CATEGORICAL_FEATURES, DATASET_CSV, DROP_COLS, MODELS_DIR, NUMERIC_FEATURES,
    RANDOM_STATE, TARGET, TEST_SIZE, TEXT_FEATURE, silence_joblib_warning,
)

silence_joblib_warning()


def load_dataset(path=DATASET_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame):
    """Drop metadata columns and label-encode the categorical context features."""
    df = df.drop(columns=DROP_COLS, errors="ignore").copy()
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y, encoders


def split(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2), TEXT_FEATURE),
        ("num", "passthrough", NUMERIC_FEATURES + CATEGORICAL_FEATURES),
    ])
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("features", preprocessor), ("clf", model)])


def main():
    p = argparse.ArgumentParser(description="Train the TF-IDF + Random Forest pipeline.")
    p.add_argument("--data", type=Path, default=DATASET_CSV)
    p.add_argument("--out", type=Path, default=MODELS_DIR)
    args = p.parse_args()

    print("===== NETWORK FAULT DIAGNOSIS - TRAINING =====")
    df = load_dataset(args.data)
    X, y, encoders = preprocess(df)
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Dataset: {len(df)} samples | train {len(X_train)} | test {len(X_test)}")

    # 5-fold CV on the training set only (no test data involved)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_val_score(build_pipeline(), X_train, y_train, cv=skf)
    print(f"5-fold CV (train set): {cv.mean():.4f} +/- {cv.std():.4f}  folds={[round(float(s), 4) for s in cv]}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy : {test_acc:.4f}  ({int((y_pred != y_test).sum())} errors / {len(y_test)})")
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=pipeline.classes_),
                       index=pipeline.classes_, columns=pipeline.classes_).to_string())

    # Single-row inference (the Streamlit app) is faster without worker threads.
    # This does not change any prediction.
    pipeline.named_steps["clf"].n_jobs = 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open(out / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(Path(args.data).name),
        "n_train": len(X_train), "n_test": len(X_test),
        "cv_train_mean": float(cv.mean()), "cv_train_std": float(cv.std()),
        "cv_train_folds": [float(s) for s in cv],
        "train_accuracy": float(train_acc), "test_accuracy": float(test_acc),
        "classes": [str(c) for c in pipeline.classes_],
        "encoders": {k: [str(c) for c in v.classes_] for k, v in encoders.items()},
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved pipeline.pkl, encoders.pkl and training_report.json to {out}")


if __name__ == "__main__":
    main()
