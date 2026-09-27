# Model Card — Historical NLP Stance Detection Project

## Status

**Historical educational / portfolio project. Not production-ready.**

## Task

Multiclass stance detection between a news headline and an article body.

Labels:
- agree
- disagree
- discuss
- unrelated

## Model families explored

- Logistic Regression
- Random Forest
- Gradient Boosting
- One-hidden-layer neural network

## Features

The classical models use TF-IDF-based text features, including unigram/bigram experimentation.

## Intended use

Educational experimentation and portfolio discussion about NLP preprocessing, supervised learning, evaluation, and model modernization.

## Not intended for

- automated fact checking;
- moderation enforcement;
- high-stakes decision making;
- production misinformation detection without additional validation.

A stance classifier does **not** determine whether a claim is true. It estimates the relationship between a headline and an article body.

## Known limitations

- historical code targets older Python / ML library APIs;
- class imbalance can make accuracy misleading;
- notebook outputs are incomplete for some tuning runs;
- no current reproducible environment is pinned;
- no fairness or robustness evaluation is included;
- no production monitoring or drift detection is implemented.

## Recommended evaluation before reuse

At minimum:
- macro-F1;
- per-class recall;
- confusion matrix;
- stratified validation;
- robustness to short, noisy, and adversarial text;
- versioned dataset documentation.
