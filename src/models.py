"""Unsupervised regime detection: K-Means and Gaussian Mixture, plus model selection and diagnostics."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from . import config as C


def scale(F: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(StandardScaler().fit_transform(F), index=F.index, columns=F.columns)


def _fit(Z, k, method, seed):
    if method == "kmeans":
        return KMeans(k, n_init=20, random_state=seed).fit_predict(Z)
    gm = GaussianMixture(k, covariance_type="diag", n_init=5, reg_covar=1e-3, random_state=seed).fit(Z)
    return gm.predict(Z)


def growth_score(Z: pd.DataFrame) -> pd.Series:
    pos = Z[[f"{c}__lvl" for c in C.GROWTH_POS]].sum(axis=1)
    neg = Z[[f"{c}__lvl" for c in C.GROWTH_NEG]].sum(axis=1)
    return (pos - neg) / (len(C.GROWTH_POS) + len(C.GROWTH_NEG))


def detect_regimes(F: pd.DataFrame, k: int = 4, method: str = "kmeans", seed: int = 42):
    """Return (labels, tags). Label 0 = strongest-growth regime; tags like 'R1 · Expansion'."""
    Z = scale(F)
    raw = _fit(Z.values, k, method, seed)
    g = pd.Series(growth_score(Z).values).groupby(raw).mean().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(g.index)}
    labels = pd.Series(raw, index=F.index).map(mapping).rename("regime")
    tags = []
    for new, old in enumerate(g.index):
        v = g[old]
        kind = "Expansion" if v > 0.35 else "Contraction" if v < -0.35 else "Transition"
        tags.append(f"R{new + 1} · {kind}")
    return labels, tags


def agreement(F, k=4, seed=42) -> float:
    """Adjusted Rand Index between K-Means and GMM partitions."""
    a, _ = detect_regimes(F, k, "kmeans", seed)
    b, _ = detect_regimes(F, k, "gmm", seed)
    return adjusted_rand_score(a, b)


def select_k(F, ks=range(2, 8), seed=42) -> pd.DataFrame:
    Z = scale(F).values
    rows = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=seed).fit(Z)
        gm = GaussianMixture(k, covariance_type="diag", n_init=3, reg_covar=1e-3, random_state=seed).fit(Z)
        rows.append({"k": k, "silhouette": silhouette_score(Z, km.labels_), "gmm_bic": gm.bic(Z)})
    return pd.DataFrame(rows).set_index("k")


def transition_matrix(labels: pd.Series) -> pd.DataFrame:
    k = int(labels.max()) + 1
    M = pd.crosstab(labels.iloc[:-1].values, labels.iloc[1:].values).reindex(index=range(k), columns=range(k), fill_value=0)
    return M.div(M.sum(axis=1).replace(0, 1), axis=0)


def fingerprints(F: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Mean standardized level of each indicator within each regime."""
    Z = scale(F)[[c for c in F.columns if c.endswith("__lvl")]]
    Z.columns = [c.replace("__lvl", "") for c in Z.columns]
    return Z.groupby(labels).mean().T
