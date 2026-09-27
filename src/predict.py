"""Run inference with the modernized local stance-classification baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

DEFAULT_MODEL_PATH = Path("artifacts/stance_pipeline.joblib")


def format_input(headline: str, article_body: str) -> str:
    return f"HEADLINE: {headline}\nARTICLE: {article_body}"


def predict_stance(
    headline: str,
    article_body: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict:
    pipeline = joblib.load(model_path)
    text = format_input(headline, article_body)
    predicted = pipeline.predict([text])[0]

    result = {"predicted_stance": str(predicted)}
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        result["probabilities"] = {
            str(label): float(prob)
            for label, prob in zip(classes, probabilities)
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline", required=True)
    parser.add_argument("--article", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    result = predict_stance(args.headline, args.article, args.model)
    print(f"Prediction: {result['predicted_stance']}")
    for label, value in sorted(
        result.get("probabilities", {}).items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {label:10s} {value:.1%}")


if __name__ == "__main__":
    main()
