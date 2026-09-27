# Modernization Roadmap

This document separates the **historical 2020 project** from a recommended modern implementation.

## 1. Reproducibility

- Move raw data to `data/raw/` and derived data to `data/processed/`.
- Move exploratory notebooks to `notebooks/`.
- Put reusable preprocessing, training, and inference code in `src/`.
- Pin dependencies with a lock file or reproducible environment.
- Add a single configuration file for random seeds and model parameters.

## 2. Data pipeline

Create a deterministic pipeline that:

1. loads headline/body pairs;
2. validates required columns and labels;
3. normalizes text;
4. splits train/validation/test data before fitting vectorizers;
5. fits TF-IDF on training data only;
6. persists the fitted preprocessing pipeline with the model.

Add tests for nulls, duplicate IDs, unknown labels, and leakage.

## 3. Evaluation

Make the following primary metrics:

- macro-F1;
- weighted-F1;
- per-class precision/recall/F1;
- confusion matrix.

Keep accuracy as a secondary metric.

Add:
- majority-class baseline;
- logistic-regression baseline;
- confidence intervals or repeated cross-validation where appropriate.

## 4. Models

### Classical baselines
- Logistic Regression
- Linear SVM
- Random Forest
- Gradient Boosting

### Neural baseline
Replace the TensorFlow 1.x MLP with current Keras or PyTorch.

### Modern NLP comparison
Add a compact transformer classifier and compare it with the TF-IDF baseline on:
- quality;
- latency;
- training cost;
- inference cost;
- operational complexity.

## 5. Model governance

Add:
- data card;
- model card;
- intended-use statement;
- failure modes;
- class-distribution analysis;
- reproducibility metadata;
- experiment log.

## 6. Serving

Expose inference through a small API:

```text
POST /predict
{
  "headline": "...",
  "article_body": "..."
}
```

Return:
- predicted stance;
- class probabilities;
- model version;
- preprocessing version.

## 7. CI/CD

On every pull request:

- lint Python;
- run unit tests;
- run a small inference smoke test;
- validate the model package can load;
- validate example request/response schemas.

## 8. Portfolio demo

A lightweight demo should allow an interviewer to paste a headline and article body, then show:

- predicted stance;
- confidence by class;
- key TF-IDF terms for the linear baseline;
- side-by-side output from classical and transformer models.

This would transform the repository from a notebook archive into a small, reviewable ML product while preserving the historical project.
