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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
)

# ---------------- CONFIG ----------------

BASE_DIR = Path(__file__).resolve().parent

# CSV files in C:\Users\USER\Desktop\AIB\Midterm
TRAIN_PATH = BASE_DIR / "training_data_fake.csv"
VAL_INPUT_PATH = BASE_DIR / "validation_data.csv"
VAL_RESULTS_PATH = BASE_DIR / "validation_data_results.csv"  # optional
MODEL_PATH = BASE_DIR / "customer_centric_model.joblib"
OUTPUT_CSV_PATH = BASE_DIR / "validation_with_labels.csv"


# -------------- TEXT CLEANING ------------

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Convert to lowercase
    - Keep letters, digits, % and spaces
    - Collapse multiple spaces
    """
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9%\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------- TRAIN MODE --------------

def train_mode():
    print(f"Loading training data from {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH)
    print("Columns in training data:", df.columns.tolist())
    print(df.head())

    # Detect text column
    if "sentences" in df.columns:
        text_col = "sentences"
    elif "sentence" in df.columns:
        text_col = "sentence"
    else:
        raise ValueError("No text column named 'sentences' or 'sentence' found.")

    label_col = "label"

    # Ensure labels are integers (0/1)
    df[label_col] = df[label_col].astype(int)

    # Clean text
    df["clean_sentence"] = df[text_col].apply(clean_text)

    # Label distribution plot
    label_counts = df[label_col].value_counts().sort_index()
    plt.figure()
    label_counts.plot(kind="bar")
    plt.title("Label distribution in training data")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    X = df["clean_sentence"].values
    y = df[label_col].values

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Train size:", len(X_train))
    print("Val size:", len(X_val))

    # Model: TF-IDF + Linear SVM
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),  # uni + bigrams
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            ("clf", LinearSVC()),
        ]
    )

    print("Training model on train split...")
    model.fit(X_train, y_train)
    print("Training done.")

    # Evaluate on validation split
    y_val_pred = model.predict(X_val)

    print("\nValidation classification report:")
    print(classification_report(y_val, y_val_pred))

    mcc = matthews_corrcoef(y_val, y_val_pred)
    print("Validation MCC:", mcc)

    cm = confusion_matrix(y_val, y_val_pred)
    print("\nConfusion matrix (val):")
    print(cm)

    # Confusion matrix plot
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Validation Split)")
    plt.tight_layout()
    plt.show()

    # Retrain on full data
    print("\nRetraining on full dataset for final model...")
    model.fit(X, y)
    print("Final model trained on all data.")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


# -------------- PREDICT MODE --------------

def predict_mode():
    # Load model
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")

    # Load validation_data (unlabeled)
    print(f"Loading validation data from {VAL_INPUT_PATH}")
    val_df = pd.read_csv(VAL_INPUT_PATH)
    print("Columns in validation_data:", val_df.columns.tolist())
    print(val_df.head())

    # Detect text column
    if "sentences" in val_df.columns:
        text_col = "sentences"
    elif "sentence" in val_df.columns:
        text_col = "sentence"
    else:
        raise ValueError(
            "No text column named 'sentences' or 'sentence' found in validation_data."
        )

    # Clean and predict
    val_df["clean_sentence"] = val_df[text_col].apply(clean_text)
    print("Predicting labels for validation_data...")
    preds = model.predict(val_df["clean_sentence"].values)
    val_df["label"] = preds

    # Save predictions (this is what you'd submit)
    val_df[[text_col, "label"]].to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_CSV_PATH}")
    print(val_df[[text_col, "label"]].head())

    # Optional evaluation if results file exists
    if VAL_RESULTS_PATH.exists():
        print(f"\nFound results file at {VAL_RESULTS_PATH}, evaluating MCC...")
        true_df = pd.read_csv(VAL_RESULTS_PATH)

        if text_col not in true_df.columns:
            raise ValueError(
                f"Column '{text_col}' not found in validation_data_results."
            )

        merged = pd.merge(
            val_df[[text_col, "label"]],
            true_df[[text_col, "label"]],
            on=text_col,
            how="inner",
            suffixes=("_pred", "_true"),
        )

        print(f"Merged {len(merged)} rows for evaluation.")
        y_true = merged["label_true"]
        y_pred = merged["label_pred"]

        print("\nEvaluation on validation_data_results:")
        print(classification_report(y_true, y_pred))

        mcc = matthews_corrcoef(y_true, y_pred)
        print("MCC:", mcc)

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:")
        print(cm)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix (validation_data_results)")
        plt.tight_layout()
        plt.show()
    else:
        print(
            "\nNo validation_data_results.csv found – skipping evaluation "
            "(this is expected on Monday with unseen data)."
        )


# -------------- MAIN --------------

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "predict"}:
        print("Usage:")
        print("  python model.py train    # train model + visualize + save")
        print("  python model.py predict  # load model + predict validation + optional eval")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "train":
        train_mode()
    else:
        predict_mode()


if __name__ == "__main__":
    main()
