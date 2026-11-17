# Global Macro Regime Detection & Predictive Analytics

An ML pipeline that identifies macroeconomic regimes from 15 indicators, forecasts regime transitions, and measures how equities, bonds, currencies and commodities behave in each regime.

| Component | Where |
|---|---|
| 15 macro indicators (FRED, no API key needed) | `src/config.py`, `src/data.py` |
| 45 engineered time-series features (level, 6m momentum, 12m rolling vol per indicator) | `src/features.py` |
| K-Means and Gaussian Mixture regime detection, k-selection (silhouette / BIC), K-Means vs GMM agreement (ARI) | `src/models.py` |
| Regime-transition forecasting (multinomial logistic regression, chronological hold-out, vs "no change" baseline) | `src/predict.py` |
| Regime vs returns across 4 asset classes | `src/analysis.py` |
| Interactive front end | `app.py` (Streamlit) · `demo/index.html` (static, simulated data) |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                 # interactive app
python run_pipeline.py --k 4 --method gmm --horizon 3   # CLI -> CSVs in outputs/
python -m pytest -q                                  # tests (synthetic data, no network)
```
First run downloads data into `data/raw/` (git-ignored). Delete that folder to refresh.

## Deploy
- **Streamlit Community Cloud (free):** push this repo to GitHub → share.streamlit.io → New app → pick the repo, branch `main`, main file `app.py`.
- **Static demo only:** `demo/index.html` is a self-contained page (simulated data) for GitHub Pages / Netlify Drop.

## Known limitations (read before quoting results)
- **Look-ahead in regime labels:** scaling and clustering use the full sample, so regimes are a *descriptive* history, not a real-time signal. Only the forecaster is evaluated out-of-sample.
- **Data revisions & release lags:** FRED series are latest-vintage and published with delay, so a regime "known" at month-end was not knowable then. Use ALFRED vintages for a real-time study.
- **Commodities history:** `GSG` starts in 2006; swap in a longer proxy in `src/config.py`.
- Regime tags (Expansion / Transition / Contraction) are assigned from a growth score after fitting; they are interpretations, not ground truth.
- Not investment advice.

## Ideas to extend
Hidden Markov / Markov-switching models · walk-forward forecasting · gradient boosting with SHAP · regime-conditioned portfolio backtest with transaction costs.
