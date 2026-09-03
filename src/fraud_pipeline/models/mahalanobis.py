"""Mahalanobis-distance anomaly detector using a robust covariance estimate (MinCovDet).

Features are standardized before fitting. Mahalanobis distance is mathematically
scale-invariant in exact arithmetic, but MinCovDet's *estimation* of the covariance
matrix it depends on involves iterative subsampling and matrix inversion -- which,
like PCA, can be numerically dominated by whichever feature has the largest raw
variance (here, Amount, at roughly 250x the variance of most V-features). This
mirrors the standardization lesson from the earlier PCA project: without it, the
"robust" covariance estimate isn't reliably estimating what it claims to.

`support_fraction` is set explicitly rather than left at MinCovDet's default (~0.5,
implying tolerance for up to ~50% contamination). That mismatch against this data's
real ~0.19% fraud rate produced hundreds of "Determinant has increased; this should
not happen" RuntimeWarnings during fitting -- a real numerical instability in the
C-step iterations, not noise. Testing support_fraction in {0.501, 0.75, 0.9, 0.95}
against the training split showed 0.95 was the first value to fully eliminate the
warnings (181, 869, 181, and 0 occurrences respectively -- not a smooth function of
support_fraction, so this was verified empirically rather than assumed). 0.95 is
also comfortably justified statistically: it tolerates far more contamination than
the ~0.19% actually present, while giving the C-step algorithm a large enough,
numerically stable subset to work with.
"""

from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler

MAHALANOBIS_SUPPORT_FRACTION = 0.95


def fit_mahalanobis(X_train, random_state: int = 42) -> tuple[StandardScaler, MinCovDet]:
    scaler = StandardScaler().fit(X_train)
    model = MinCovDet(support_fraction=MAHALANOBIS_SUPPORT_FRACTION, random_state=random_state).fit(
        scaler.transform(X_train)
    )
    return scaler, model


def score(scaler: StandardScaler, model: MinCovDet, X):
    """Higher = more anomalous. MinCovDet.mahalanobis already returns squared distance
    from the robust center, which increases with anomalousness -- no sign flip needed."""
    return model.mahalanobis(scaler.transform(X))
