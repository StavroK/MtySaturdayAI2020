"""Train a reproducible TF-IDF + Logistic Regression stance classifier.

This module modernizes the 2020 notebook workflow without changing the original
notebooks. It uses only local files and open-source Python packages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 8
DEFAULT_MODEL_PATH = Path("artifacts/stance_pipeline.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/metrics.json")


def load_dataset(stances_path: str | Path, bodies_path: str | Path) -> pd.DataFrame:
    """Load and join stance/headline records to article bodies."""
    stances = pd.read_csv(stances_path)
    bodies = pd.read_csv(bodies_path)

    required_stances = {"Headline", "Body ID", "Stance"}
    required_bodies = {"Body ID", "articleBody"}

    missing_stances = required_stances.difference(stances.columns)
    missing_bodies = required_bodies.difference(bodies.columns)
    if missing_stances or missing_bodies:
        raise ValueError(
            f"Missing required columns. stances={sorted(missing_stances)}, "
            f"bodies={sorted(missing_bodies)}"
        )

    df = stances.merge(bodies, on="Body ID", how="inner", validate="many_to_one")
    df = df.dropna(subset=["Headline", "articleBody", "Stance"]).copy()
    df["text"] = (
        "HEADLINE: "
        + df["Headline"].astype(str)
        + "\nARTICLE: "
        + df["articleBody"].astype(str)
    )
    return df


def build_pipeline(max_features: int = 5000) -> Pipeline:
    """Build the modernized classical-NLP baseline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train(
    stances_path: str | Path = "train_stances.csv",
    bodies_path: str | Path = "train_bodies.csv",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    test_size: float = 0.20,
) -> dict:
    """Train, evaluate, and persist the baseline model."""
    df = load_dataset(stances_path, bodies_path)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["Stance"],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["Stance"],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)

    labels = sorted(df["Stance"].unique().tolist())
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "labels": labels,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classification_report": report,
        "random_state": RANDOM_STATE,
    }

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stances", default="train_stances.csv")
    parser.add_argument("--bodies", default="train_bodies.csv")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH))
    args = parser.parse_args()

    metrics = train(args.stances, args.bodies, args.model, args.metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
