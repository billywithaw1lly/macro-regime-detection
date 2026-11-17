"""Regime-transition forecasting: predict the regime h months ahead, evaluated on a chronological hold-out."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_xy(F, labels, h):
    d = F.join(labels.shift(-h).rename("y")).dropna()
    return d.drop(columns="y"), d["y"].astype(int)


def fit_forecaster(F, labels, h=3, train_frac=0.7, C_=0.1):
    X, y = make_xy(F, labels, h)
    cut = int(len(X) * train_frac)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C_, max_iter=3000)).fit(X.iloc[:cut], y.iloc[:cut])
    Xte, yte = X.iloc[cut:], y.iloc[cut:]
    pred = clf.predict(Xte)
    cur = labels.loc[Xte.index].values
    chg = cur != yte.values
    proba = clf.predict_proba(Xte)
    cols = {c: i for i, c in enumerate(clf.classes_)}
    p_stay = np.array([proba[i, cols[c]] if c in cols else 0.0 for i, c in enumerate(cur)])
    report = {
        "accuracy": accuracy_score(yte, pred),
        "persistence_accuracy": float((cur == yte.values).mean()),
        "n_test": int(len(yte)),
        "n_transitions": int(chg.sum()),
        "transition_recall": float((pred[chg] == yte.values[chg]).mean()) if chg.any() else float("nan"),
        "transition_auc": float(roc_auc_score(chg, 1 - p_stay)) if 0 < chg.sum() < len(chg) else float("nan"),
        "test_start": str(Xte.index[0].date()),
    }
    return clf, report


def outlook(clf, F) -> pd.DataFrame:
    """Probability of each regime h months after each date (in-sample before test_start)."""
    return pd.DataFrame(clf.predict_proba(F), index=F.index, columns=clf.classes_)
