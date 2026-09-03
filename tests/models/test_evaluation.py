import numpy as np
import pytest

from fraud_pipeline.models.evaluation import evaluate, threshold_for_target_fpr


def test_evaluate_requires_pred_or_score():
    with pytest.raises(ValueError):
        evaluate(y_true=[0, 1])


def test_evaluate_from_y_pred_perfect():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    metrics = evaluate(y_true, y_pred=y_pred)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert "pr_auc" not in metrics
    assert "roc_auc" not in metrics


def test_evaluate_from_y_pred_known_confusion_matrix():
    # 2 true positives, 1 false positive, 1 false negative, 2 true negatives
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    metrics = evaluate(y_true, y_pred=y_pred)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)


def test_evaluate_false_positive_rate_zero_negatives_does_not_divide_by_zero():
    y_true = [1, 1, 1]
    y_pred = [1, 0, 1]
    metrics = evaluate(y_true, y_pred=y_pred)
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_evaluate_from_y_score_perfect_separation():
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.9, 0.95]  # fraud scores strictly higher
    metrics = evaluate(y_true, y_score=y_score)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert "recall" not in metrics  # no threshold given, no point metrics expected


def test_evaluate_from_y_score_with_threshold_adds_point_metrics():
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.9, 0.95]
    metrics = evaluate(y_true, y_score=y_score, threshold=0.5)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_evaluate_random_scores_roc_auc_near_half():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=2000)
    y_score = rng.uniform(size=2000)  # unrelated to y_true
    metrics = evaluate(y_true, y_score=y_score)
    assert metrics["roc_auc"] == pytest.approx(0.5, abs=0.05)


def test_threshold_for_target_fpr_hits_zero_fpr_threshold():
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.9, 0.95]
    threshold = threshold_for_target_fpr(y_true, y_score, target_fpr=0.0)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    metrics = evaluate(y_true, y_pred=y_pred)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)


def test_threshold_for_target_fpr_matches_known_operating_point():
    # Overlapping distributions (realistic case) rather than a clean separation gap,
    # which can leave no achievable threshold near some target FPR values.
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 1000 + [1] * 200)
    y_score = np.concatenate([rng.normal(0, 1, 1000), rng.normal(2, 1, 200)])
    threshold = threshold_for_target_fpr(y_true, y_score, target_fpr=0.2)
    y_pred = (y_score >= threshold).astype(int)
    metrics = evaluate(y_true, y_pred=y_pred)
    assert metrics["false_positive_rate"] == pytest.approx(0.2, abs=0.02)
