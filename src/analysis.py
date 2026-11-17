"""Regime vs asset-class return analysis."""
import numpy as np
import pandas as pd


def regime_returns(labels: pd.Series, rets: pd.DataFrame, forward: bool = True) -> pd.DataFrame:
    """Per regime x asset stats. forward=True uses next-month return (what you could act on)."""
    r = rets.shift(-1) if forward else rets
    d = r.join(labels, how="inner").dropna(subset=["regime"])
    rows = []
    for reg, grp in d.groupby("regime"):
        for a in rets.columns:
            x = grp[a].dropna()
            if len(x) < 6:
                continue
            vol = x.std() * np.sqrt(12)
            rows.append({"regime": int(reg), "asset": a, "months": len(x), "ann_return": x.mean() * 12,
                         "ann_vol": vol, "sharpe": x.mean() * 12 / vol if vol else np.nan, "hit_rate": (x > 0).mean()})
    return pd.DataFrame(rows)
