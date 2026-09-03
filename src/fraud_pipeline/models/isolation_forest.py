"""Isolation Forest anomaly detector -- Phase 3a raw-feature baseline.

Fit on the training split as-is (fraud included, ~99.8% legitimate) -- matches how
it would actually be deployed, where labels aren't available to filter training data.
"""

from sklearn.ensemble import IsolationForest


def fit_isolation_forest(X_train, random_state: int = 42, **kwargs) -> IsolationForest:
    model = IsolationForest(random_state=random_state, **kwargs)
    model.fit(X_train)
    return model


def score(model: IsolationForest, X):
    """Higher = more anomalous. IsolationForest's own decision_function is the opposite
    (higher = more normal), so the sign is flipped here to match the shared convention."""
    return -model.decision_function(X)
