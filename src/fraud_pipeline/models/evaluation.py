"""Shared evaluation harness so every Phase 3 model is scored the same way.

Convention: a higher score always means "more anomalous / more likely fraud" --
each model file is responsible for orienting its own raw output (e.g. flipping
IsolationForest's decision_function) to match this before calling evaluate().
"""

import json
import pickle
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"


def evaluate(y_true, y_pred=None, y_score=None, threshold=None) -> dict:
    """Scores predictions and/or continuous anomaly scores against true labels.

    - y_pred (binary 0/1): recall, precision, false_positive_rate are computed from it.
    - y_score (continuous, higher = more anomalous): pr_auc and roc_auc are computed
      across all thresholds. If y_pred is not given but threshold is, y_score is
      thresholded (score >= threshold => predicted positive) to also get the point metrics.

    At least one of y_pred, y_score must be given. Returns only the keys that could
    actually be computed from what was provided.
    """
    if y_pred is None and y_score is None:
        raise ValueError("evaluate() requires y_pred and/or y_score")

    y_true = np.asarray(y_true)
    metrics = {}

    if y_score is not None:
        y_score = np.asarray(y_score)
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        if y_pred is None and threshold is not None:
            y_pred = (y_score >= threshold).astype(int)

    if y_pred is not None:
        y_pred = np.asarray(y_pred)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return metrics


def threshold_for_target_fpr(y_true, y_score, target_fpr: float) -> float:
    """Finds the score threshold whose false-positive rate is closest to `target_fpr`.

    Used only to build a comparable reporting table across models at a matched
    operating point -- not a substitute for choosing a deployment threshold on a
    held-out validation set.
    """
    fpr, _tpr, thresholds = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thresholds[idx])


def _current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def save_model_artifact(model, name: str, hyperparameters: dict, metrics: dict) -> Path:
    """Saves a trained model + metadata to models/<name>/<ISO-date>/ at the project root."""
    date_str = datetime.now(UTC).date().isoformat()
    out_dir = MODELS_DIR / name / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)

    metadata = {
        "name": name,
        "trained_at": datetime.now(UTC).isoformat(),
        "git_commit": _current_git_commit(),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return out_dir
