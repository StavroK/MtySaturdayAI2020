from src.predict import format_input
from src.train_baseline import build_pipeline


def test_format_input_contains_both_fields():
    text = format_input("Example headline", "Example article")
    assert "Example headline" in text
    assert "Example article" in text


def test_pipeline_can_fit_and_predict_toy_data():
    x = [
        "HEADLINE: cats are mammals ARTICLE: cats are mammals",
        "HEADLINE: sky is green ARTICLE: the sky is blue",
        "HEADLINE: new phone released ARTICLE: analysts discuss the new phone",
        "HEADLINE: football result ARTICLE: recipe for apple pie",
        "HEADLINE: water freezes ARTICLE: water freezes at low temperature",
        "HEADLINE: earth is flat ARTICLE: evidence shows earth is spherical",
        "HEADLINE: election debate ARTICLE: report discusses the debate",
        "HEADLINE: stock market ARTICLE: local weather forecast",
    ]
    y = [
        "agree",
        "disagree",
        "discuss",
        "unrelated",
        "agree",
        "disagree",
        "discuss",
        "unrelated",
    ]

    model = build_pipeline(max_features=100, min_df=1, max_df=1.0)
    model.fit(x, y)
    pred = model.predict(["HEADLINE: cats are mammals ARTICLE: cats are mammals"])
    assert pred[0] in {"agree", "disagree", "discuss", "unrelated"}
