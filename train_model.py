"""
Train and evaluate the complaint-classification models.

Run from the project root:

    uv run python train_model.py

The script:

1. loads the real CFPB complaint sample (and builds it from the official ZIP
   when only the ZIP is present);
2. cleans the text and removes missing/duplicate/short records;
3. keeps the strongest product categories and caps the dominant ones;
4. splits stratified train/test sets BEFORE fitting anything;
5. fits TF-IDF on the training text only (no leakage);
6. trains Logistic Regression and Multinomial Naive Bayes;
7. compares them and saves the better model plus all evaluation artefacts.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import config
import data_loader
import preprocessing


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_and_prepare() -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset, clean it, and return (text, label) pairs."""
    config.ensure_directories()

    if not config.DATASET_PATH.exists():
        print(
            "data/complaints.csv not found - building it from the best available source ..."
        )
        data_loader.prepare_dataset()

    df = data_loader.load_dataset()
    print(f"Loaded {len(df):,} rows from {config.DATASET_PATH}")

    required = [config.TEXT_COLUMN, config.TARGET_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[config.TEXT_COLUMN, config.TARGET_COLUMN])
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype("string").str.strip()
    df = df[df[config.TEXT_COLUMN].str.len() > 0]

    before = len(df)
    df = df.drop_duplicates(subset=[config.TEXT_COLUMN])
    print(f"Removed {before - len(df):,} duplicate narratives")

    df = preprocessing.preprocess_dataframe(df)
    print(f"Kept {len(df):,} usable complaints after text cleaning")

    if df.empty:
        raise ValueError("No usable complaints remain after cleaning.")

    counts = df[config.TARGET_COLUMN].value_counts()
    eligible = counts[counts >= config.MIN_CLASS_COUNT]
    if eligible.empty:
        raise ValueError(
            f"No category has at least {config.MIN_CLASS_COUNT} complaints. "
            "Lower MIN_CLASS_COUNT in config.py or supply a larger dataset."
        )
    classes = eligible.head(config.MAX_CLASSES).index.tolist()
    df = df[df[config.TARGET_COLUMN].isin(classes)].copy()

    # Cap dominant categories so the benchmark is not dominated by one product.
    balanced_parts = []
    rng = np.random.default_rng(config.RANDOM_STATE)
    for product, group in df.groupby(config.TARGET_COLUMN, observed=True):
        if len(group) > config.MAX_PER_CLASS:
            group = group.sample(
                n=config.MAX_PER_CLASS, random_state=int(rng.integers(0, 1_000_000))
            )
        balanced_parts.append(group)
    df = pd.concat(balanced_parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=config.RANDOM_STATE).reset_index(drop=True)

    print("\nClass distribution used for modelling:")
    distribution = (
        df[config.TARGET_COLUMN]
        .value_counts()
        .rename_axis("product")
        .reset_index(name="complaints")
    )
    distribution["display_name"] = distribution["product"].map(config.display_name)
    print(distribution.to_string(index=False))
    distribution.to_csv(config.CLASS_DISTRIBUTION_PATH, index=False)

    return df["clean_narrative"], df[config.TARGET_COLUMN]


def evaluate(name: str, model, X_test, y_test, labels: list[str]) -> dict:
    """Compute the standard classification metrics for one fitted model."""
    predictions = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
    }

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    metrics["_report"] = report
    metrics["_matrix"] = matrix
    return metrics


def top_model_features(
    model, feature_names: np.ndarray, labels: list[str], top_n: int = 12
) -> dict:
    """Extract the strongest positive TF-IDF terms for each class (LR only)."""
    if not hasattr(model, "coef_"):
        return {}
    result: dict[str, list[dict]] = {}
    for row, label in zip(model.coef_, labels):
        strongest = np.argsort(row)[-top_n:][::-1]
        result[label] = [
            {"term": str(feature_names[index]), "weight": round(float(row[index]), 4)}
            for index in strongest
        ]
    return result


def main() -> None:
    _print_header("ONLINE CONSUMER COMPLAINT CLASSIFICATION - MODEL TRAINING")

    texts, labels = load_and_prepare()

    label_order = sorted(labels.unique().tolist())
    print(f"\nModelling {len(label_order)} categories: {label_order}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels,
    )
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")

    # Fit the vectorizer on TRAINING text only. This is the no-leakage rule.
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(
        f"TF-IDF matrix: {X_train_tfidf.shape[0]:,} x {X_train_tfidf.shape[1]:,} features"
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1_000,
            C=1.0,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train_tfidf, y_train)
        fitted[name] = model
        metrics = evaluate(name, model, X_test_tfidf, y_test, label_order)
        results.append(metrics)
        print(
            f"  accuracy={metrics['accuracy']:.4f}  "
            f"precision={metrics['precision_weighted']:.4f}  "
            f"recall={metrics['recall_weighted']:.4f}  "
            f"f1={metrics['f1_weighted']:.4f}"
        )

    comparison = pd.DataFrame(
        [
            {
                "model": item["model"],
                "accuracy": item["accuracy"],
                "precision": item["precision_weighted"],
                "recall": item["recall_weighted"],
                "f1_score": item["f1_weighted"],
                "macro_f1": item["f1_macro"],
            }
            for item in results
        ]
    ).sort_values("f1_score", ascending=False)

    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted[best_name]
    best_metrics = next(item for item in results if item["model"] == best_name)

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {best_name}")

    # ------------------------------------------------------------------ #
    # Persist artefacts
    # ------------------------------------------------------------------ #
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(label_order, config.LABEL_ENCODER_PATH)

    comparison.to_csv(config.COMPARISON_PATH, index=False)

    payload = {
        "best_model": best_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_rows": int(len(texts)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features": int(X_train_tfidf.shape[1]),
        "labels": label_order,
        "display_names": {label: config.display_name(label) for label in label_order},
        "models": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in results
        ],
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    # Confusion matrix of the winning model, with readable row/column labels.
    matrix = pd.DataFrame(
        best_metrics["_matrix"],
        index=[f"actual: {label}" for label in label_order],
        columns=[f"predicted: {label}" for label in label_order],
    )
    matrix.to_csv(config.CONFUSION_PATH)

    report = pd.DataFrame(best_metrics["_report"]).transpose()
    report.index.name = "class_or_average"
    report.to_csv(config.REPORT_PATH)

    metadata = {
        "best_model": best_name,
        "labels": label_order,
        "display_names": {label: config.display_name(label) for label in label_order},
        "n_features": int(X_train_tfidf.shape[1]),
        "dataset_rows": int(len(texts)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "tfidf": {
            "max_features": config.TFIDF_MAX_FEATURES,
            "ngram_range": list(config.TFIDF_NGRAM_RANGE),
            "min_df": config.TFIDF_MIN_DF,
            "max_df": config.TFIDF_MAX_DF,
            "sublinear_tf": True,
        },
        "trained_at_utc": payload["generated_at_utc"],
        "python": platform.python_version(),
        "dataset_source": config.CFPB_HOME,
        "dataset_note": (
            "Real complaints from the CFPB Consumer Complaint Database, sampled from "
            "a public mirror of the same database because the current official bulk "
            "CSV export no longer carries the consumer narrative text."
        ),
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    features = top_model_features(
        best_model, vectorizer.get_feature_names_out(), label_order
    )
    with open(config.MODEL_DIR / "top_features.json", "w", encoding="utf-8") as handle:
        json.dump(features, handle, indent=2)

    # A small, human-readable snapshot of test predictions for the report.
    sample_predictions = pd.DataFrame(
        {
            "complaint": X_test.head(25).to_numpy(),
            "actual": y_test.head(25).to_numpy(),
            "predicted": best_model.predict(X_test_tfidf[:25]),
        }
    )
    sample_predictions["correct"] = (
        sample_predictions["actual"] == sample_predictions["predicted"]
    )
    sample_predictions.to_csv(config.MODEL_DIR / "sample_predictions.csv", index=False)

    _print_header("TRAINING COMPLETE")
    print(f"Best model       : {best_name}")
    print(f"Accuracy         : {best_metrics['accuracy']:.4f}")
    print(f"Weighted F1      : {best_metrics['f1_weighted']:.4f}")
    print(f"Vectorizer       : {config.VECTORIZER_PATH}")
    print(f"Model            : {config.MODEL_PATH}")
    print(f"Metrics          : {config.METRICS_PATH}")
    print("\nStart the dashboard with:  uv run streamlit run app.py")


if __name__ == "__main__":
    main()
