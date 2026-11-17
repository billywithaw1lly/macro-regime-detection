"""CLI: python run_pipeline.py --k 4 --method gmm --horizon 3  -> writes CSVs to outputs/"""
import argparse
from pathlib import Path
from src import analysis, data, features, models, predict

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--method", choices=["kmeans", "gmm"], default="kmeans")
ap.add_argument("--horizon", type=int, default=3)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

macro = data.load_macro(); F = features.build_features(macro); rets = data.load_assets()
print(f"{macro.shape[1]} indicators, {F.shape[1]} features, {len(F)} months ({F.index[0].date()} -> {F.index[-1].date()})")
labels, tags = models.detect_regimes(F, a.k, a.method, a.seed)
print("K-Means vs GMM agreement (ARI):", round(models.agreement(F, a.k, a.seed), 3))
clf, rep = predict.fit_forecaster(F, labels, a.horizon)
print("Forecast report:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()})
res = analysis.regime_returns(labels, rets); res["regime_name"] = res["regime"].map(dict(enumerate(tags)))
out = Path("outputs"); out.mkdir(exist_ok=True)
labels.to_csv(out / "regimes.csv"); res.to_csv(out / "regime_returns.csv", index=False)
models.transition_matrix(labels).to_csv(out / "transition_matrix.csv")
models.select_k(F).to_csv(out / "k_selection.csv")
print(res.round(3).to_string(index=False))
