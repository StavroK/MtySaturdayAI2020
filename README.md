# NLP Stance Detection — Saturdays AI Monterrey 2020

> **Portfolio case study:** classical NLP experimentation, model comparison, and a modern zero-cost implementation for classifying the relationship between a news headline and an article body.

This repository contains my final NLP project from **Saturdays AI Monterrey, 4th Edition — Cycle 1 (2020)**. The original notebooks are preserved as historical evidence of the work, while the 2026 additions turn the project into a reproducible, cloud-agnostic showcase that can run locally without API keys or paid infrastructure.

## What the model predicts

Given a **headline** and a **news article body**, classify their relationship as:

- **agree**
- **disagree**
- **discuss**
- **unrelated**

This is a **stance-classification** task, not automated fact checking. A predicted stance does not determine whether a claim is true.

## Run the modernized demo for free

```bash
git clone https://github.com/StavroK/news-stance-detection.git
cd news-stance-detection
pip install -r requirements.txt
streamlit run app.py
```

On first use, the app can train the modernized TF-IDF + Logistic Regression baseline directly from the repository data. The trained artifact stays on your machine.

**Operating cost:** $0  
**Cloud account required:** No  
**API key required:** No

## Architecture

```mermaid
flowchart LR
    A[Headline + Article Body] --> B[Text Pipeline]
    B --> C[TF-IDF + n-grams]
    C --> D[Logistic Regression]
    D --> E[Stance + class probabilities]
    E --> F[Local Streamlit UI]
```

The historical notebooks also explore Random Forest, Gradient Boosting, and a one-hidden-layer neural-network prototype.

## Modernized project structure

```text
.
├── app.py
├── src/
│   ├── train_baseline.py
│   └── predict.py
├── tests/
│   └── test_pipeline.py
├── artifacts/
│   └── .gitkeep
├── docs/
│   ├── index.html
│   ├── modernization-roadmap.md
│   └── model-card.md
├── .github/workflows/
│   ├── ci.yml
│   └── pages.yml
├── requirements.txt
├── original notebooks...
└── original data...
```

## What the original project demonstrates

The 2020 workflow covers:

- text preprocessing and normalization;
- class encoding;
- headline/article-body handling;
- TF-IDF vectorization;
- unigram and bigram features;
- train/test splitting;
- chi-square feature inspection;
- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- neural-network experimentation;
- randomized search and grid search;
- classification reports and confusion matrices.

One saved preprocessing experiment produced approximately **42k training rows × 300 TF-IDF features** and **7.5k test rows × 300 features**.

## What was modernized in 2026

Rather than rewriting the historical notebooks, the project now adds a clean modern baseline:

- scikit-learn `Pipeline` to keep preprocessing and prediction together;
- TF-IDF with unigram/bigram features;
- class-balanced Logistic Regression;
- stratified train/test splitting;
- macro-F1, weighted-F1, accuracy, per-class metrics, and confusion matrix;
- persisted local model artifact;
- command-line inference;
- local Streamlit interface;
- automated unit tests;
- GitHub Actions CI;
- static portfolio page prepared for GitHub Pages;
- model card and modernization roadmap.

## Models explored

| Model | Artifact | Purpose |
|---|---|---|
| Logistic Regression | `Ridge_Logistic_Regression.ipynb` + `src/train_baseline.py` | Interpretable linear baseline and reproducible modern implementation |
| Random Forest | `Random_Forrest.ipynb` | Nonlinear ensemble comparison |
| Gradient Boosting | `Gradient_Boosting_Machine.ipynb` | Boosted ensemble comparison |
| One-hidden-layer MLP | `MLP_one_hiden_layer.ipynb` | Historical neural-network experiment |
| Data preparation | `Understand and Standarize Examples.ipynb` | Text cleanup, TF-IDF generation, encoding, and feature inspection |

## Historical technical debt

The original notebooks were created in 2020 and intentionally remain untouched. Important limitations include:

1. The MLP uses TensorFlow 1.x-era APIs such as `tf.contrib` and placeholders.
2. Some hyperparameter-search cells were interrupted or saved without final results.
3. The notebook-era workflow was not packaged as reusable application code.
4. The original environment was not fully pinned.
5. Some evaluations emphasized accuracy despite the four-class imbalance.

The new `src/` implementation addresses the reproducibility and packaging issues while preserving the original work for comparison.

## Evaluation philosophy

For this problem, the modernized version treats **macro-F1 and per-class performance** as first-class metrics because overall accuracy can obscure poor performance on minority classes.

The training script writes evaluation results to:

```text
artifacts/metrics.json
```

and the trained pipeline to:

```text
artifacts/stance_pipeline.joblib
```

Both are generated locally; the trained binary artifact is intentionally not committed.

## Portfolio site

A static showcase is included at **`docs/index.html`** and a GitHub Pages deployment workflow is included in **`.github/workflows/pages.yml`**. After Pages is enabled for the repository using GitHub Actions as the source, pushes to `master` deploy the site without a paid hosting service.

## Interview discussion points

This project supports discussion about:

- converting raw text into ML-ready features;
- establishing and defending a baseline;
- comparing linear, ensemble, and neural approaches;
- class imbalance and metric selection;
- reproducibility and ML technical debt;
- the difference between a notebook experiment and a deployable ML product;
- modernizing a legacy AI asset without obscuring its history;
- delivering a working AI demonstration without cloud cost or vendor lock-in.

## Historical context

The original code and notebooks were produced in **2020** as part of the Saturdays AI Monterrey program. The portfolio and engineering additions were created in **2026** and are clearly separated from the historical implementation.

See:

- [Modernization roadmap](docs/modernization-roadmap.md)
- [Model card](docs/model-card.md)
- [Static portfolio page](docs/index.html)
