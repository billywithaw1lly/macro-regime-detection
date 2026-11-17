"""Feature engineering: 3 features per indicator -> 45 features for 15 indicators."""
import pandas as pd


def build_features(macro: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in macro.columns:
        s = macro[c]
        out[f"{c}__lvl"] = s                      # current level
        out[f"{c}__chg6"] = s.diff(6)             # 6-month momentum
        out[f"{c}__vol12"] = s.rolling(12).std()  # 12-month rolling volatility
    return pd.DataFrame(out).dropna()
