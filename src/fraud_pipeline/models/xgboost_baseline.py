"""XGBoost supervised classifier -- Phase 3a raw-feature baseline.

Unlike the two unsupervised models, this one does use labels during training -- it's
the ceiling: how much better can a model do when it's allowed to see `Class` directly,
versus the unsupervised approaches that only see it at evaluation time.
"""

from xgboost import XGBClassifier


def fit_xgboost(X_train, y_train, random_state: int = 42) -> XGBClassifier:
    positive_rate = y_train.mean()
    scale_pos_weight = (1 - positive_rate) / positive_rate  # standard imbalance correction
    model = XGBClassifier(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
    )
    model.fit(X_train, y_train)
    return model


def score(model: XGBClassifier, X):
    """Higher = more anomalous: predicted probability of the fraud (1) class."""
    return model.predict_proba(X)[:, 1]
