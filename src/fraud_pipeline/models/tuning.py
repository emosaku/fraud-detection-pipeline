"""Phase 3a follow-up: tuning the unsupervised detectors on a validation split.

Plan and reasoning: docs/phase3-unsupervised-optimization.md. Results and their
interpretation: docs/phase3a-tuning-results.md.

Three stages, each writing one JSON file under results/phase3a/:

- `baseline`: the original Phase 3a protocol exactly as notebooks/03 runs it (70/30
  temporal split, sklearn defaults, seed 42, matched-FPR thresholds picked on test
  labels). Reproduces the notebook's table, so it's the stored "unoptimized" result.
- `tuned`: a 60/15/25 temporal train/validation/test split. Every configuration is
  fit on train and ranked by mean validation PR-AUC across seeds; thresholds are
  picked on validation. The test split is scored once, for the original default
  configurations and the selected ones, so "defaults on the new split" isolates the
  effect of the split from the effect of tuning.

Run:  python -m fraud_pipeline.models.tuning --stage all
"""

import argparse
import itertools
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

# Same macOS OpenMP guard as autoencoder.py; must precede any torch/xgboost import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import pinvh
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet, shrunk_covariance
from sklearn.preprocessing import RobustScaler, StandardScaler

from fraud_pipeline.models.evaluation import (
    PROJECT_ROOT,
    _current_git_commit,
    evaluate,
    threshold_for_target_fpr,
)

log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "phase3a"
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
IQR_TOP_K = 6

ISOLATION_FOREST_GRID = {
    "max_samples": [256, 1024, 4096, 0.1],
    "n_estimators": [100, 300, 500, 1000],
    "max_features": [0.5, 0.75, 1.0],
}
ISOLATION_FOREST_DEFAULT = {"max_samples": "auto", "n_estimators": 100, "max_features": 1.0}

SHRINKAGE_LEVELS = [1e-4, 1e-3, 1e-2, 0.05, 0.1]
MAHALANOBIS_VARIANTS = ["mcd", "mcd_robust_scaler", "empirical", "ledoit_wolf"] + [
    f"mcd_shrink_{a:g}" for a in SHRINKAGE_LEVELS
]
MAHALANOBIS_DEFAULT = "mcd"

AUTOENCODER_GRID = {"bottleneck_dim": [4, 7, 10], "learning_rate": [1e-3, 3e-4]}
AUTOENCODER_DEFAULT = {"bottleneck_dim": 7, "learning_rate": 1e-3}


# ---------------------------------------------------------------- splits and baseline


def temporal_split(df: pd.DataFrame, *fractions: float) -> list[pd.DataFrame]:
    """Cuts `df` on `Time` quantiles into consecutive pieces, oldest first.

    `fractions` are the sizes of every piece except the last, e.g. (0.6, 0.15) gives
    train 60% / validation 15% / test 25%. Ties at a cut stay in the earlier piece,
    matching notebooks/03's `Time <= cutoff` convention.
    """
    if sum(fractions) >= 1:
        raise ValueError("fractions must leave room for a final piece")
    edges = np.cumsum(fractions)
    cutoffs = [df["Time"].quantile(e) for e in edges]
    pieces, lower = [], -np.inf
    for cutoff in cutoffs:
        pieces.append(df[(df["Time"] > lower) & (df["Time"] <= cutoff)])
        lower = cutoff
    pieces.append(df[df["Time"] > lower])
    return pieces


def iqr_rule(train_df: pd.DataFrame, top_k: int = IQR_TOP_K):
    """notebooks/03's IQR baseline, fit on `train_df` only. Returns a predict function."""
    fraud, normal = train_df[train_df["Class"] == 1], train_df[train_df["Class"] == 0]

    def cohens_d(a, b):
        return (a.mean() - b.mean()) / (((a.std() ** 2 + b.std() ** 2) / 2) ** 0.5)

    separation = pd.Series({c: cohens_d(fraud[c], normal[c]) for c in FEATURE_COLUMNS})
    features = separation.abs().sort_values(ascending=False).index[:top_k].tolist()
    bounds = {c: train_df[c].quantile([0.25, 0.75]).tolist() for c in features}

    def predict(df: pd.DataFrame) -> np.ndarray:
        flags = pd.DataFrame(
            {
                c: (df[c] < q1 - 1.5 * (q3 - q1)) | (df[c] > q3 + 1.5 * (q3 - q1))
                for c, (q1, q3) in bounds.items()
            }
        )
        return flags.any(axis=1).astype(int).values

    return predict, features


# ---------------------------------------------------------------- model wrappers


def fit_score_isolation_forest(X_train, X_eval_list, config: dict, seed: int):
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(random_state=seed, n_jobs=-1, **config).fit(X_train)
    return [-model.decision_function(X) for X in X_eval_list]


def _mahalanobis_squared(X, location, precision):
    diff = X - location
    return np.einsum("ij,jk,ik->i", diff, precision, diff)


def fit_score_mahalanobis(X_train, X_eval_list, variant: str, seed: int, mcd_cache=None):
    """Fits one covariance variant and returns squared distances for each eval set.

    `mcd_cache` (a dict) lets the shrinkage variants reuse one MinCovDet fit per seed;
    shrinking is applied to the fitted covariance, the same way the original test did.
    """
    from fraud_pipeline.models.mahalanobis import MAHALANOBIS_SUPPORT_FRACTION

    scaler = (RobustScaler() if variant == "mcd_robust_scaler" else StandardScaler()).fit(X_train)
    Z = scaler.transform(X_train)

    if variant == "empirical":
        est = EmpiricalCovariance().fit(Z)
        location, covariance = est.location_, est.covariance_
    elif variant == "ledoit_wolf":
        est = LedoitWolf().fit(Z)
        location, covariance = est.location_, est.covariance_
    else:
        key = ("mcd", variant == "mcd_robust_scaler", seed)
        if mcd_cache is not None and key in mcd_cache:
            est = mcd_cache[key]
        else:
            est = MinCovDet(support_fraction=MAHALANOBIS_SUPPORT_FRACTION, random_state=seed).fit(Z)
            if mcd_cache is not None:
                mcd_cache[key] = est
        location, covariance = est.location_, est.covariance_
        if variant.startswith("mcd_shrink_"):
            covariance = shrunk_covariance(covariance, float(variant.removeprefix("mcd_shrink_")))

    precision = pinvh(covariance)
    min_eig = float(np.linalg.eigvalsh(covariance).min())
    scores = [_mahalanobis_squared(scaler.transform(X), location, precision) for X in X_eval_list]
    return scores, {"min_eigenvalue": min_eig}


def fit_score_autoencoder(X_train_legit, X_eval_list, config: dict, seed: int):
    from fraud_pipeline.models.autoencoder import fit_autoencoder
    from fraud_pipeline.models.autoencoder import score as score_ae

    scaler, model, info = fit_autoencoder(X_train_legit, random_state=seed, **config)
    return [score_ae(scaler, model, X) for X in X_eval_list], {
        "epochs_trained": info["epochs_trained"]
    }


def fit_score_xgboost(X_train, y_train, X_eval_list, seed: int):
    from fraud_pipeline.models.xgboost_baseline import fit_xgboost
    from fraud_pipeline.models.xgboost_baseline import score as score_xgb

    model = fit_xgboost(X_train, y_train, random_state=seed)
    return [score_xgb(model, X) for X in X_eval_list]


# ---------------------------------------------------------------- reporting helpers


def config_key(config) -> str:
    if isinstance(config, str):
        return config
    return ", ".join(f"{k}={v}" for k, v in config.items())


def summarize(rows: list[dict], metric: str = "val_pr_auc") -> pd.DataFrame:
    """Mean and std of `metric` (and val ROC-AUC) per config, best first."""
    df = pd.DataFrame(rows)
    out = df.groupby("config").agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
        roc_auc_mean=("val_roc_auc", "mean"),
        seeds=("seed", "nunique"),
    )
    out["std"] = out["std"].fillna(0.0)
    return out.sort_values("mean", ascending=False)


def matched_test_metrics(y_val, val_scores, y_test, test_scores, target_fpr: float) -> dict:
    """Threshold chosen on validation to hit `target_fpr`, then applied to test."""
    threshold = threshold_for_target_fpr(y_val, val_scores, target_fpr)
    return evaluate(y_test, y_score=test_scores, threshold=threshold)


def _mean_std(per_seed: list[dict]) -> dict:
    df = pd.DataFrame(per_seed)
    return {
        m: {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=1)) if len(df) > 1 else 0.0}
        for m in df.columns
    }


def _write(name: str, payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _current_git_commit(),
        **payload,
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("wrote %s", path)
    return path


# ---------------------------------------------------------------- stages


def run_baseline(df: pd.DataFrame, out_dir: Path = RESULTS_DIR) -> dict:
    """The unoptimized Phase 3a run, exactly as notebooks/03 does it."""
    from fraud_pipeline.models.autoencoder import fit_autoencoder
    from fraud_pipeline.models.autoencoder import score as score_ae

    train_df, test_df = temporal_split(df, 0.70)
    X_train, X_test = train_df[FEATURE_COLUMNS].values, test_df[FEATURE_COLUMNS].values
    y_train, y_test = train_df["Class"].values, test_df["Class"].values

    predict_iqr, iqr_features = iqr_rule(train_df)
    iqr_pred = predict_iqr(test_df)
    results = {"IQR baseline": evaluate(y_test, y_pred=iqr_pred, y_score=iqr_pred.astype(float))}
    target_fpr = results["IQR baseline"]["false_positive_rate"]

    (iso,) = fit_score_isolation_forest(X_train, [X_test], {}, seed=42)
    (maha,), _ = fit_score_mahalanobis(X_train, [X_test], "mcd", seed=42)
    (xgb,) = fit_score_xgboost(X_train, y_train, [X_test], seed=42)
    legit = train_df[train_df["Class"] == 0].sort_values("Time")[FEATURE_COLUMNS].values
    ae_scaler, ae_model, _ = fit_autoencoder(legit, random_state=42)
    ae = score_ae(ae_scaler, ae_model, X_test)

    for name, scores in [
        ("Isolation Forest", iso),
        ("Mahalanobis (MinCovDet)", maha),
        ("XGBoost", xgb),
        ("Autoencoder", ae),
    ]:
        # Original protocol: threshold picked on the test set's own labels.
        threshold = threshold_for_target_fpr(y_test, scores, target_fpr)
        results[name] = evaluate(y_test, y_score=scores, threshold=threshold)

    payload = {
        "stage": "baseline",
        "protocol": "notebooks/03: 70/30 temporal split, defaults, seed 42, "
        "matched-FPR thresholds chosen on test labels",
        "split": {"train": len(train_df), "test": len(test_df), "test_fraud": int(y_test.sum())},
        "iqr_features": iqr_features,
        "target_fpr": target_fpr,
        "results": results,
    }
    _write("baseline", payload, out_dir)
    return payload


def run_tuned(
    df: pd.DataFrame,
    seeds=DEFAULT_SEEDS,
    out_dir: Path = RESULTS_DIR,
    if_grid: dict = ISOLATION_FOREST_GRID,
    ae_grid: dict = AUTOENCODER_GRID,
    maha_variants=MAHALANOBIS_VARIANTS,
    ae_jobs: int = 5,
) -> dict:
    seeds = list(seeds)
    train_df, val_df, test_df = temporal_split(df, 0.60, 0.15)
    X_train, X_val, X_test = (d[FEATURE_COLUMNS].values for d in (train_df, val_df, test_df))
    y_train, y_val, y_test = (d["Class"].values for d in (train_df, val_df, test_df))
    legit = train_df[train_df["Class"] == 0].sort_values("Time")[FEATURE_COLUMNS].values
    log.info(
        "split: train %d (%d fraud), val %d (%d fraud), test %d (%d fraud)",
        len(train_df),
        y_train.sum(),
        len(val_df),
        y_val.sum(),
        len(test_df),
        y_test.sum(),
    )

    predict_iqr, iqr_features = iqr_rule(train_df)
    iqr_val = evaluate(y_val, y_pred=predict_iqr(val_df))
    target_fpr = iqr_val["false_positive_rate"]  # chosen on validation, not test
    iqr_test_pred = predict_iqr(test_df)
    iqr_test = evaluate(y_test, y_pred=iqr_test_pred, y_score=iqr_test_pred.astype(float))

    def val_row(config, seed, val_scores, extra=None):
        m = evaluate(y_val, y_score=val_scores)
        return {
            "config": config_key(config),
            "seed": seed,
            "val_pr_auc": m["pr_auc"],
            "val_roc_auc": m["roc_auc"],
            **(extra or {}),
        }

    # ---- Isolation Forest grid (validation only)
    t0 = time.time()
    if_candidates = [ISOLATION_FOREST_DEFAULT] + [
        dict(zip(if_grid, v, strict=True)) for v in itertools.product(*if_grid.values())
    ]
    if_rows = []
    for config in if_candidates:
        for seed in seeds:
            (val_scores,) = fit_score_isolation_forest(X_train, [X_val], config, seed)
            if_rows.append(val_row(config, seed, val_scores))
    if_summary = summarize(if_rows)
    if_best = next(c for c in if_candidates if config_key(c) == if_summary.index[0])
    log.info("isolation forest grid: %d fits in %.0fs", len(if_rows), time.time() - t0)

    # ---- Mahalanobis variants (validation only)
    t0 = time.time()
    maha_rows, cache = [], {}
    for variant in maha_variants:
        deterministic = variant in ("empirical", "ledoit_wolf")
        for seed in seeds[:1] if deterministic else seeds:
            (val_scores,), extra = fit_score_mahalanobis(X_train, [X_val], variant, seed, cache)
            maha_rows.append(val_row(variant, seed, val_scores, extra))
    maha_summary = summarize(maha_rows)
    maha_best = maha_summary.index[0]
    log.info("mahalanobis variants: %d fits in %.0fs", len(maha_rows), time.time() - t0)

    # ---- Autoencoder grid (validation only), parallel across processes
    t0 = time.time()
    ae_configs = [dict(zip(ae_grid, v, strict=True)) for v in itertools.product(*ae_grid.values())]
    jobs = [(c, s) for c in ae_configs for s in seeds]
    outputs = Parallel(n_jobs=ae_jobs)(
        delayed(fit_score_autoencoder)(legit, [X_val], c, s) for c, s in jobs
    )
    ae_rows = [
        val_row(c, s, scores[0], extra)
        for (c, s), (scores, extra) in zip(jobs, outputs, strict=True)
    ]
    ae_summary = summarize(ae_rows)
    ae_best = next(c for c in ae_configs if config_key(c) == ae_summary.index[0])
    log.info("autoencoder grid: %d fits in %.0fs", len(ae_rows), time.time() - t0)

    # ---- Final: score test once for default and selected configs
    t0 = time.time()
    finals = {}

    def finalize(label, model, config, fit_fn):
        per_seed = []
        for seed in seeds:
            val_scores, test_scores = fit_fn(config, seed)
            per_seed.append(
                {
                    **matched_test_metrics(y_val, val_scores, y_test, test_scores, target_fpr),
                    "val_pr_auc": evaluate(y_val, y_score=val_scores)["pr_auc"],
                }
            )
        finals[label] = {
            "model": model,
            "config": config_key(config),
            "per_seed": per_seed,
            "summary": _mean_std(per_seed),
        }

    def if_fit(config, seed):
        return fit_score_isolation_forest(X_train, [X_val, X_test], config, seed)

    def maha_fit(variant, seed):
        return fit_score_mahalanobis(X_train, [X_val, X_test], variant, seed, cache)[0]

    finalize("Isolation Forest (default)", "isolation_forest", ISOLATION_FOREST_DEFAULT, if_fit)
    finalize("Isolation Forest (tuned)", "isolation_forest", if_best, if_fit)
    finalize("Mahalanobis (default)", "mahalanobis", MAHALANOBIS_DEFAULT, maha_fit)
    finalize("Mahalanobis (tuned)", "mahalanobis", maha_best, maha_fit)

    ae_final_jobs = [(c, s) for c in (AUTOENCODER_DEFAULT, ae_best) for s in seeds]
    ae_final_out = dict(
        zip(
            [(config_key(c), s) for c, s in ae_final_jobs],
            Parallel(n_jobs=ae_jobs)(
                delayed(fit_score_autoencoder)(legit, [X_val, X_test], c, s)
                for c, s in ae_final_jobs
            ),
            strict=True,
        )
    )
    finalize(
        "Autoencoder (default)",
        "autoencoder",
        AUTOENCODER_DEFAULT,
        lambda c, s: ae_final_out[(config_key(c), s)][0],
    )
    finalize(
        "Autoencoder (tuned)",
        "autoencoder",
        ae_best,
        lambda c, s: ae_final_out[(config_key(c), s)][0],
    )
    finalize(
        "XGBoost (default, reference)",
        "xgboost",
        "default",
        lambda c, s: fit_score_xgboost(X_train, y_train, [X_val, X_test], s),
    )
    log.info("final test scoring in %.0fs", time.time() - t0)

    payload = {
        "stage": "tuned",
        "protocol": "60/15/25 temporal split; configs ranked by mean validation PR-AUC "
        "across seeds; matched-FPR threshold chosen on validation; test scored once",
        "seeds": seeds,
        "split": {
            "train": len(train_df),
            "train_fraud": int(y_train.sum()),
            "val": len(val_df),
            "val_fraud": int(y_val.sum()),
            "test": len(test_df),
            "test_fraud": int(y_test.sum()),
        },
        "iqr_features": iqr_features,
        "target_fpr_from_validation": target_fpr,
        "iqr_test": iqr_test,
        "selected": {"isolation_forest": if_best, "mahalanobis": maha_best, "autoencoder": ae_best},
        "validation": {
            "isolation_forest": if_summary.reset_index().to_dict("records"),
            "mahalanobis": maha_summary.reset_index().to_dict("records"),
            "mahalanobis_min_eigenvalues": pd.DataFrame(maha_rows)
            .groupby("config")["min_eigenvalue"]
            .mean()
            .to_dict(),
            "autoencoder": ae_summary.reset_index().to_dict("records"),
            "autoencoder_epochs": pd.DataFrame(ae_rows)
            .groupby("config")["epochs_trained"]
            .mean()
            .to_dict(),
        },
        "test": finals,
    }
    _write("tuned", payload, out_dir)
    return payload


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["baseline", "tuned", "all"], default="all")
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS), help="number of seeds")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--ae-jobs", type=int, default=5)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    from fraud_pipeline.ingestion.loader import load_transactions

    df = load_transactions()
    out_dir = Path(args.out_dir)
    if args.stage in ("baseline", "all"):
        run_baseline(df, out_dir)
    if args.stage in ("tuned", "all"):
        run_tuned(df, seeds=range(args.seeds), out_dir=out_dir, ae_jobs=args.ae_jobs)


if __name__ == "__main__":
    main()
