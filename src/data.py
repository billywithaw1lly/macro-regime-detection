"""Data loading. FRED via public CSV endpoint (no API key); assets via yfinance. Cached in data/raw/."""
from pathlib import Path
import pandas as pd
from . import config as C

RAW = Path("data/raw")


def _fred(sid: str) -> pd.Series:
    p = RAW / f"{sid}.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True, na_values=".").iloc[:, 0].astype(float)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    s = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".").iloc[:, 0].astype(float)
    RAW.mkdir(parents=True, exist_ok=True)
    s.to_frame(sid).to_csv(p)
    return s


def load_macro() -> pd.DataFrame:
    """Month-end panel of the 15 indicators. Delete data/raw/ to force a refresh."""
    cols = {}
    for sid, tf in C.FRED.items():
        s = _fred(sid).resample("ME").last().ffill(limit=3)
        cols[sid] = s.pct_change(12) * 100 if tf == "yoy" else s
    return pd.DataFrame(cols).loc[C.START:].dropna()


def load_assets() -> pd.DataFrame:
    """Monthly total returns for the 4 asset classes."""
    import yfinance as yf
    p = RAW / "assets.csv"
    if p.exists():
        px = pd.read_csv(p, index_col=0, parse_dates=True)
    else:
        px = yf.download(list(C.ASSETS.values()), start=C.START, auto_adjust=True, progress=False)["Close"]
        RAW.mkdir(parents=True, exist_ok=True)
        px.to_csv(p)
    px = px.rename(columns={v: k for k, v in C.ASSETS.items()})[list(C.ASSETS)]
    return px.resample("ME").last().pct_change(fill_method=None)
