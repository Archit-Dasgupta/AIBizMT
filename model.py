import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
)

# ================= CONFIG =================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "training_data_fake.csv"
VAL_INPUT_PATH = BASE_DIR / "validation_data.csv"
VAL_RESULTS_PATH = BASE_DIR / "validation_data_results.csv"  # optional
MODEL_PATH = BASE_DIR / "customer_centric_model.joblib"
OUTPUT_CSV_PATH = BASE_DIR / "validation_with_labels.csv"


# ============== TEXT CLEANING ==============

def clean_text(text: str) -> str:
    """
    Clean sentence:
    - lowercase
    - keep letters, digits, %, spaces
    - collapse multiple spaces
    """
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9%\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============== TRAIN MODE ==============

def train_mode():
    print(f"Loading training data from {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH)
    print("Columns:", df.columns.tolist())
    print(df.head())

    # Basic checks
    if "sentences" not in df.columns:
        raise ValueError("Expected a 'sentences' column in training_data_fake.csv")
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in training_data_fake.csv")

    # Clean + label
    df["clean_sentence"] = df["sentences"].apply(clean_text)
    df["label"] = df["label"].astype(int)

    # Label distribution plot
    df["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Label Distribution (Training Data)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    X = df["clean_sentence"].values
    y = df["label"].values

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train size:", len(X_train))
    print("Val size:", len(X_val))

    # --------- MODEL: TF-IDF + Calibrated Linear SVM ---------
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),   # uni + bigrams
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=15000,
            )
        ),
        (
            "clf",
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    class_weight="balanced",
                    C=0.5,
                    random_state=42
                ),
                method="sigmoid",
                cv=3
            )
        )
    ])

    print("Training model on train split...")
    model.fit(X_train, y_train)

    # --------- Threshold optimization for MCC ---------
    print("\nOptimizing decision threshold for MCC...")
    probs_val = model.predict_proba(X_val)[:, 1]  # P(y=1)
    best_mcc = -1.0
    best_thresh = 0.5
    thresholds = np.arange(0.3, 0.71, 0.01)

    for t in thresholds:
        preds_t = (probs_val >= t).astype(int)
        mcc_t = matthews_corrcoef(y_val, preds_t)
        if mcc_t > best_mcc:
            best_mcc = mcc_t
            best_thresh = t

    print(f"✅ Best threshold found: {best_thresh:.2f}")
    print(f"✅ Best MCC on validation split: {best_mcc:.4f}")

    # Evaluate at best threshold
    y_val_pred_best = (probs_val >= best_thresh).astype(int)

    print("\nValidation report at best threshold:")
    print(classification_report(y_val, y_val_pred_best))

    cm = confusion_matrix(y_val, y_val_pred_best)
    print("Confusion matrix (val, best threshold):")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title("Confusion Matrix (Validation, Best Threshold)")
    plt.tight_layout()
    plt.show()

    # --------- Retrain on full data with same pipeline ---------
    print("\nRetraining calibrated model on FULL dataset...")
    model.fit(X, y)
    print("Final model trained on all data.")

    # Save pipeline + chosen threshold together
    model_package = {
        "pipeline": model,
        "threshold": float(best_thresh),
    }
    joblib.dump(model_package, MODEL_PATH)
    print(f"✅ Model package saved to: {MODEL_PATH}")


# ============== PREDICT MODE ==============

def predict_mode():
    print(f"Loading model package from {MODEL_PATH}")
    model_package = joblib.load(MODEL_PATH)
    model = model_package["pipeline"]
    threshold = model_package["threshold"]
    print(f"Using optimized threshold: {threshold:.2f}")

    print(f"Loading validation data from {VAL_INPUT_PATH}")
    val_df = pd.read_csv(VAL_INPUT_PATH)
    if "sentences" not in val_df.columns:
        raise ValueError("Expected a 'sentences' column in validation_data.csv")

    # Clean and predict
    val_df["clean_sentence"] = val_df["sentences"].apply(clean_text)
    probs = model.predict_proba(val_df["clean_sentence"])[:, 1]
    val_df["label"] = (probs >= threshold).astype(int)

    # Save predictions (this is the file you submit)
    val_df[["sentences", "label"]].to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Predictions saved to {OUTPUT_CSV_PATH}")

    # Optional evaluation if you have true labels file
    if VAL_RESULTS_PATH.exists():
        print(f"\nFound {VAL_RESULTS_PATH}, evaluating MCC...")
        true_df = pd.read_csv(VAL_RESULTS_PATH)
        if "sentences" not in true_df.columns or "label" not in true_df.columns:
            raise ValueError("validation_data_results.csv must have 'sentences' and 'label' columns.")

        merged = pd.merge(
            val_df[["sentences", "label"]],
            true_df[["sentences", "label"]],
            on="sentences",
            how="inner",
            suffixes=("_pred", "_true"),
        )

        print(f"Merged {len(merged)} rows for evaluation.")
        y_true = merged["label_true"]
        y_pred = merged["label_pred"]

        print("\nEvaluation on validation_data_results.csv:")
        print(classification_report(y_true, y_pred))
        print("MCC:", matthews_corrcoef(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:")
        print(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot()
        plt.title("Confusion Matrix (validation_data_results)")
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo validation_data_results.csv found – skipping external evaluation.")


# ============== MAIN ==============

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "predict"}:
        print("Usage:")
        print("  python model.py train   # train + tune threshold + save model")
        print("  python model.py predict # load model + predict validation CSV")
        return

    mode = sys.argv[1]
    if mode == "train":
        train_mode()
    else:
        predict_mode()


if __name__ == "__main__":
    main()

