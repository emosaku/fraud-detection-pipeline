import numpy as np
import torch

from fraud_pipeline.models.autoencoder import (
    Autoencoder,
    fit_autoencoder,
    load_autoencoder_artifact,
    save_autoencoder_artifact,
    score,
)


def _synthetic_data(n=200, p=29, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p))


def test_autoencoder_forward_pass_preserves_shape():
    model = Autoencoder(input_dim=29, bottleneck_dim=7)
    x = torch.rand(5, 29)
    assert model(x).shape == x.shape


def test_fit_autoencoder_runs_end_to_end():
    X = _synthetic_data(n=200)
    _scaler, _model, info = fit_autoencoder(X, max_epochs=3, patience=2, batch_size=32)
    assert info["epochs_trained"] >= 1
    assert "final_val_loss" in info


def test_score_output_shape_and_nonnegative():
    X_train = _synthetic_data(n=200, seed=1)
    X_test = _synthetic_data(n=20, seed=2)
    scaler, model, _info = fit_autoencoder(X_train, max_epochs=3, patience=2, batch_size=32)
    scores = score(scaler, model, X_test)
    assert scores.shape == (20,)
    assert (scores >= 0).all()


def test_score_higher_for_out_of_distribution_points():
    rng = np.random.default_rng(3)
    X_train = rng.normal(0, 1, size=(300, 29))
    scaler, model, _info = fit_autoencoder(
        X_train, max_epochs=20, patience=5, batch_size=32, random_state=42
    )

    X_normal = rng.normal(0, 1, size=(20, 29))
    X_extreme = rng.normal(0, 1, size=(20, 29)) + 20  # far outside the training distribution

    normal_scores = score(scaler, model, X_normal)
    extreme_scores = score(scaler, model, X_extreme)
    assert extreme_scores.mean() > normal_scores.mean()


def test_save_and_load_artifact_round_trip(tmp_path, monkeypatch):
    import fraud_pipeline.models.autoencoder as ae_module

    monkeypatch.setattr(ae_module, "MODELS_DIR", tmp_path)

    X_train = _synthetic_data(n=200, seed=4)
    scaler, model, info = fit_autoencoder(X_train, max_epochs=2, patience=1, batch_size=32)

    out_dir = save_autoencoder_artifact(
        model, scaler, hyperparameters={"bottleneck_dim": 7}, metrics=info
    )
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "scaler.pkl").exists()
    assert (out_dir / "metadata.json").exists()

    loaded_scaler, loaded_model = load_autoencoder_artifact(out_dir)
    X_test = _synthetic_data(n=10, seed=5)
    np.testing.assert_allclose(
        score(scaler, model, X_test), score(loaded_scaler, loaded_model, X_test), rtol=1e-5
    )
