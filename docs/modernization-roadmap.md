# Modernization Roadmap

This document separates the **historical 2020 project** from a recommended modern implementation.

The modernization goal is intentionally **cloud-agnostic and free to run**. The project should remain usable by recruiters, students, and interviewers without requiring paid infrastructure, cloud credits, or proprietary services.

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
Optionally add a compact open-source transformer classifier and compare it with the TF-IDF baseline on:
- quality;
- latency;
- training cost;
- inference cost;
- operational complexity.

The transformer should remain optional so the core demo can still run locally and at no cost.

## 5. Model governance

Add:
- data card;
- model card;
- intended-use statement;
- failure modes;
- class-distribution analysis;
- reproducibility metadata;
- experiment log.

## 6. Local inference

The preferred architecture is local-first:

```text
User Input
   ↓
Browser UI or local Python app
   ↓
Saved TF-IDF pipeline
   ↓
Saved classifier
   ↓
Prediction + confidence
```

No external API key should be required for the core demo.

A simple local endpoint could expose:

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

Use GitHub Actions so the public repository can validate itself at no hosting cost.

On every pull request:

- lint Python;
- run unit tests;
- run a small inference smoke test;
- validate the model package can load;
- validate example request/response schemas.

## 8. Zero-cost portfolio demo

The showcase should avoid requiring AWS, Azure, GCP, or another paid cloud service.

Preferred options:

### Option A — GitHub Pages static showcase
Use GitHub Pages for:
- project explanation;
- architecture;
- model comparison;
- sample predictions;
- evaluation charts;
- interactive examples that do not require a backend.

This is the most durable portfolio option.

### Option B — Local interactive demo
Provide one-command startup:

```bash
python app.py
```

or:

```bash
streamlit run app.py
```

The interviewer can run the model locally using the repository's included artifacts.

### Option C — Browser-only inference
If the final classical model is small enough, export the logic so inference can run entirely in the browser. This would allow GitHub Pages to provide a genuinely interactive demo with no backend and no infrastructure cost.

## 9. Interview experience

An interviewer should be able to:

1. open the repository;
2. understand the problem in under two minutes;
3. inspect model results;
4. run the project locally without credentials;
5. optionally use a browser demo;
6. understand the modernization decisions.

This turns the repository from a notebook archive into a small, reviewable ML product while preserving the original historical project and keeping the operating cost at **$0**.
