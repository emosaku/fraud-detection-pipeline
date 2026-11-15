import json

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler

from fraud_pipeline.models.tuning import (
    FEATURE_COLUMNS,
    config_key,
    fit_score_mahalanobis,
    matched_test_metrics,
    run_tuned,
    summarize,
    temporal_split,
)


def _synthetic_transactions(n=3000, fraud_rate=0.02, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df["Amount"] = rng.exponential(50, size=n)
    df["Time"] = np.sort(rng.uniform(0, 10_000, size=n))
    df["Class"] = (rng.random(n) < fraud_rate).astype(int)
    # Fraud sits off-center on a few features, so every detector has some signal.
    df.loc[df["Class"] == 1, ["V14", "V4", "V12"]] += 4.0
    return df


def test_temporal_split_is_ordered_and_sized():
    df = _synthetic_transactions(n=1000)
    train, val, test = temporal_split(df, 0.60, 0.15)
    assert len(train) + len(val) + len(test) == len(df)
    assert train["Time"].max() < val["Time"].min()
    assert val["Time"].max() < test["Time"].min()
    assert len(train) == pytest.approx(600, abs=2)
    assert len(val) == pytest.approx(150, abs=2)


def test_temporal_split_rejects_fractions_that_leave_nothing():
    with pytest.raises(ValueError):
        temporal_split(_synthetic_transactions(n=100), 0.7, 0.3)


def test_empirical_variant_matches_sklearn_mahalanobis():
    X = _synthetic_transactions(n=500)[FEATURE_COLUMNS].values
    (scores,), _ = fit_score_mahalanobis(X, [X], "empirical", seed=0)
    Z = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(scores, EmpiricalCovariance().fit(Z).mahalanobis(Z), rtol=1e-6)


def test_shrinkage_raises_the_smallest_eigenvalue():
    X = _synthetic_transactions(n=800)[FEATURE_COLUMNS].values
    cache = {}
    _, plain = fit_score_mahalanobis(X, [X], "mcd", seed=0, mcd_cache=cache)
    _, shrunk = fit_score_mahalanobis(X, [X], "mcd_shrink_0.1", seed=0, mcd_cache=cache)
    assert shrunk["min_eigenvalue"] > plain["min_eigenvalue"]
    assert len(cache) == 1  # the shrinkage variant reused the seed's MinCovDet fit


def test_summarize_ranks_by_mean_and_reports_spread():
    rows = [
        {"config": "a", "seed": 0, "val_pr_auc": 0.2, "val_roc_auc": 0.9},
        {"config": "a", "seed": 1, "val_pr_auc": 0.4, "val_roc_auc": 0.9},
        {"config": "b", "seed": 0, "val_pr_auc": 0.5, "val_roc_auc": 0.8},
    ]
    out = summarize(rows)
    assert list(out.index) == ["b", "a"]
    assert out.loc["a", "mean"] == pytest.approx(0.3)
    assert out.loc["a", "std"] == pytest.approx(np.std([0.2, 0.4], ddof=1))
    assert out.loc["b", "std"] == 0.0


def test_matched_threshold_comes_from_validation_not_test():
    from fraud_pipeline.models.evaluation import evaluate, threshold_for_target_fpr

    rng = np.random.default_rng(0)
    y_val, y_test = rng.random(2000) < 0.05, rng.random(2000) < 0.05
    val_scores = rng.normal(size=2000) + 2 * y_val
    # Test scores shifted up: a threshold re-picked on test would differ from validation's.
    test_scores = rng.normal(size=2000) + 2 * y_test + 1.0

    metrics = matched_test_metrics(y_val, val_scores, y_test, test_scores, target_fpr=0.1)
    val_threshold = threshold_for_target_fpr(y_val, val_scores, 0.1)
    assert metrics == evaluate(y_test, y_score=test_scores, threshold=val_threshold)
    # The shift means the validation threshold overshoots the 10% target on test.
    assert metrics["false_positive_rate"] > 0.2


def test_config_key_is_stable():
    key = config_key({"max_samples": 256, "n_estimators": 100})
    assert key == "max_samples=256, n_estimators=100"
    assert config_key("mcd") == "mcd"


def test_run_tuned_end_to_end_on_a_tiny_grid(tmp_path):
    df = _synthetic_transactions(n=3000)
    payload = run_tuned(
        df,
        seeds=[0, 1],
        out_dir=tmp_path,
        if_grid={"max_samples": [64, 256], "n_estimators": [50], "max_features": [1.0]},
        ae_grid={"bottleneck_dim": [4], "learning_rate": [1e-3]},
        maha_variants=["mcd", "empirical", "mcd_shrink_0.01"],
        ae_jobs=1,
    )
    saved = json.loads((tmp_path / "tuned.json").read_text())
    assert saved["stage"] == "tuned"
    for label in ["Isolation Forest (tuned)", "Mahalanobis (tuned)", "Autoencoder (tuned)"]:
        summary = payload["test"][label]["summary"]
        assert 0.0 <= summary["pr_auc"]["mean"] <= 1.0
        assert len(payload["test"][label]["per_seed"]) == 2
    assert payload["selected"]["mahalanobis"] in ["mcd", "empirical", "mcd_shrink_0.01"]
