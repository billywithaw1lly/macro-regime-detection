import numpy as np, pandas as pd
from src import analysis, config as C, features, models, predict


def synthetic(n=300, seed=0):
    r = np.random.default_rng(seed); idx = pd.date_range("1995-01-31", periods=n, freq="ME")
    reg = np.repeat(r.integers(0, 3, n // 30 + 1), 30)[:n]
    d = {c: r.normal(0, 1, n) + reg * (1 if c not in C.GROWTH_NEG else -1) for c in C.FRED}
    return pd.DataFrame(d, index=idx), reg


def test_end_to_end():
    macro, _ = synthetic(); F = features.build_features(macro)
    assert F.shape[1] == 45
    for m in ("kmeans", "gmm"):
        labels, tags = models.detect_regimes(F, 3, m)
        assert len(tags) == 3 and set(labels) == {0, 1, 2}
    assert abs(models.transition_matrix(labels).sum(axis=1) - 1).max() < 1e-9
    clf, rep = predict.fit_forecaster(F, labels, 3); assert 0 <= rep["accuracy"] <= 1
    rets = pd.DataFrame(np.random.default_rng(1).normal(0.005, 0.04, (len(macro), 4)), index=macro.index, columns=list(C.ASSETS))
    assert not analysis.regime_returns(labels, rets).empty
