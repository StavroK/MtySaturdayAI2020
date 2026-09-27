"""Zero-cost local Streamlit demo for the NLP stance-detection project."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import predict_stance
from src.train_baseline import DEFAULT_MODEL_PATH, train

st.set_page_config(
    page_title="NLP Stance Detection | Portfolio Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NLP Stance Detection")
st.caption("2020 NLP project · modernized as a local, cloud-agnostic, zero-cost demo")

with st.expander("What does this model do?", expanded=True):
    st.write(
        "Given a news headline and an article body, the model predicts whether "
        "the article agrees, disagrees, discusses, or is unrelated to the headline."
    )
    st.info(
        "This is a stance classifier, not a fact checker. A predicted stance "
        "does not determine whether a claim is true."
    )

model_path = Path(DEFAULT_MODEL_PATH)

if not model_path.exists():
    st.warning("The modernized baseline has not been trained on this machine yet.")
    if st.button("Train baseline locally", type="primary"):
        with st.spinner("Training TF-IDF + Logistic Regression from repository data..."):
            metrics = train()
        st.success(
            f"Model trained. Macro-F1: {metrics['macro_f1']:.3f} · "
            f"Accuracy: {metrics['accuracy']:.3f}"
        )
        st.rerun()

headline = st.text_input(
    "Headline",
    placeholder="Enter the headline to evaluate",
)
article = st.text_area(
    "Article body",
    height=220,
    placeholder="Paste the article text here",
)

if st.button("Classify stance", disabled=not model_path.exists()):
    if not headline.strip() or not article.strip():
        st.error("Enter both a headline and an article body.")
    else:
        result = predict_stance(headline, article)
        st.subheader(f"Prediction: {result['predicted_stance'].upper()}")

        probabilities = result.get("probabilities", {})
        if probabilities:
            chart_df = pd.DataFrame(
                {
                    "stance": list(probabilities.keys()),
                    "probability": list(probabilities.values()),
                }
            ).set_index("stance")
            st.bar_chart(chart_df)

st.divider()
st.markdown(
    """
### Why this demo is intentionally local-first

- No paid cloud account
- No API keys
- No external model endpoint
- Open-source Python stack
- Reproducible baseline trained from repository data
- Suitable for technical interviews, learning, and portfolio review

The original 2020 notebooks remain in the repository as historical artifacts.
The code in src/ is the modernized baseline.
"""
)
