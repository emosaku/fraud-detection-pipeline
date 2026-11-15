"""Autoencoder anomaly detector -- Phase 3a raw-feature comparison, 4th of 4 methods.

Unlike Isolation Forest and Mahalanobis, this one trains ONLY on legitimate
transactions. Isolation Forest assumes anomalies are rare and doesn't need them
excluded; MinCovDet is specifically a robust estimator designed to resist
contamination. A neural network has no such built-in safeguard -- given enough
capacity and training time, it can learn to reconstruct fraud just as well as
legitimate transactions, which would defeat the entire premise of using
reconstruction error as an anomaly score.
"""

import json
import os
import pickle
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# XGBoost and PyTorch each load their own OpenMP runtime. On macOS, having both
# loaded in the same process deadlocks the moment torch runs its own parallel region
# after XGBoost has already claimed the thread pool (reproduced: fit_xgboost()
# completes fine, then fit_autoencoder() hangs indefinitely on the very first
# training step). Must be set before `import torch` below -- setting it later, even
# earlier in the same function, is too late once the library has already loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from fraud_pipeline.models.evaluation import MODELS_DIR, PROJECT_ROOT

INPUT_DIM = 29
BOTTLENECK_DIM = 7
HIDDEN_DIMS = (20, 14)


class Autoencoder(nn.Module):
    """Symmetric encoder-decoder: 29 -> 20 -> 14 -> 7 -> 14 -> 20 -> 29.

    ReLU on every hidden layer (including the bottleneck); the final output layer is
    linear -- we're reconstructing scaled continuous features, not probabilities.
    """

    def __init__(self, input_dim: int = INPUT_DIM, bottleneck_dim: int = BOTTLENECK_DIM):
        super().__init__()
        h1, h2 = HIDDEN_DIMS
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),  # linear output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def architecture_description(self) -> str:
        bottleneck = self.encoder[-2].out_features
        input_dim = self.encoder[0].in_features
        dims = [input_dim, *HIDDEN_DIMS, bottleneck, *reversed(HIDDEN_DIMS), input_dim]
        return " -> ".join(str(d) for d in dims)


def fit_autoencoder(
    X_train_legit: np.ndarray,
    random_state: int = 42,
    bottleneck_dim: int = BOTTLENECK_DIM,
    validation_fraction: float = 0.15,
    max_epochs: int = 200,
    patience: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> tuple[StandardScaler, Autoencoder, dict]:
    """Fits the autoencoder on legitimate-only training data.

    `X_train_legit` must already be Class == 0 only and sorted in temporal order --
    the most recent `validation_fraction` of it is carved out for early stopping,
    not the real test split.
    """
    torch.manual_seed(random_state)

    n_val = int(len(X_train_legit) * validation_fraction)
    X_fit_raw = X_train_legit[:-n_val]
    X_val_raw = X_train_legit[-n_val:]

    scaler = StandardScaler().fit(X_fit_raw)
    X_fit = torch.tensor(scaler.transform(X_fit_raw), dtype=torch.float32)
    X_val = torch.tensor(scaler.transform(X_val_raw), dtype=torch.float32)

    model = Autoencoder(input_dim=X_fit.shape[1], bottleneck_dim=bottleneck_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_fit), batch_size=batch_size, shuffle=True
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for _epoch in range(max_epochs):
        model.train()
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), X_val).item()
        epochs_trained += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    training_info = {
        "epochs_trained": epochs_trained,
        "final_val_loss": best_val_loss,
        # Everything needed to reproduce this fit from metadata.json alone.
        "hyperparameters": {
            "architecture": model.architecture_description(),
            "input_dim": int(X_fit.shape[1]),
            "hidden_dims": list(HIDDEN_DIMS),
            "bottleneck_dim": bottleneck_dim,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "validation_fraction": validation_fraction,
            "random_state": random_state,
            "optimizer": "Adam",
            "loss": "MSE",
        },
    }
    return scaler, model, training_info


def score(scaler: StandardScaler, model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous: per-sample MSE between input and reconstruction."""
    model.eval()
    X_t = torch.tensor(scaler.transform(X), dtype=torch.float32)
    with torch.no_grad():
        reconstruction = model(X_t)
    return ((X_t - reconstruction) ** 2).mean(dim=1).numpy()


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


def save_autoencoder_artifact(
    model: Autoencoder, scaler: StandardScaler, hyperparameters: dict, metrics: dict
) -> Path:
    """Saves model weights (model.pt), the fitted scaler (scaler.pkl), and metadata.json
    to models/autoencoder/<ISO-date>/ -- a different shape from save_model_artifact()
    since PyTorch weights and a separately-fitted scaler don't fit that helper's
    single-pickled-object convention.
    """
    date_str = datetime.now(UTC).date().isoformat()
    out_dir = MODELS_DIR / "autoencoder" / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")

    with (out_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "name": "autoencoder",
        "architecture": model.architecture_description(),
        "trained_at": datetime.now(UTC).isoformat(),
        "git_commit": _current_git_commit(),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return out_dir


def load_autoencoder_artifact(model_dir: Path) -> tuple[StandardScaler, Autoencoder]:
    """Loads a model saved by save_autoencoder_artifact() -- the necessary complement,
    since a saved-but-unloadable artifact isn't usable by anything downstream."""
    with (model_dir / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)

    # Artifacts saved before hyperparameters were recorded used the default bottleneck.
    hyperparameters = {}
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        hyperparameters = json.loads(metadata_path.read_text()).get("hyperparameters", {})
    model = Autoencoder(
        input_dim=hyperparameters.get("input_dim", INPUT_DIM),
        bottleneck_dim=hyperparameters.get("bottleneck_dim", BOTTLENECK_DIM),
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    model.eval()

    return scaler, model
