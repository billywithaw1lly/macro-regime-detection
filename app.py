import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from src import analysis, config as C, data, features, models, predict

st.set_page_config(page_title="Macro Regime Lab", layout="wide")
PAL = ["#2a9d8f", "#8ab17d", "#e9c46a", "#e76f51", "#6a8fd8", "#a66cc2"]


@st.cache_data(show_spinner="Loading FRED and market data (first run downloads)…")
def load():
    m = data.load_macro()
    return m, features.build_features(m), data.load_assets()


macro, F, rets = load()
st.title("Macro Regime Lab")
st.caption(f"{macro.shape[1]} indicators → {F.shape[1]} engineered features · {F.index[0]:%Y-%m} to {F.index[-1]:%Y-%m}")
with st.sidebar:
    method = st.radio("Algorithm", ["kmeans", "gmm"], format_func=lambda x: "K-Means" if x == "kmeans" else "Gaussian Mixture")
    k = st.slider("Regimes (k)", 2, 6, 4)
    h = st.slider("Forecast horizon (months)", 1, 12, 3)
    seed = st.number_input("Seed", value=42, step=1)

labels, tags = models.detect_regimes(F, k, method, int(seed))
Z = models.scale(F); g = models.growth_score(Z)
clf, rep = predict.fit_forecaster(F, labels, h)
c = st.columns(4)
c[0].metric("K-Means ↔ GMM agreement (ARI)", f"{models.agreement(F, k, int(seed)):.2f}")
c[1].metric(f"Out-of-sample accuracy (t+{h})", f"{rep['accuracy']:.0%}")
c[2].metric("'No change' baseline", f"{rep['persistence_accuracy']:.0%}")
c[3].metric("Transition AUC", "n/a" if rep["transition_auc"] != rep["transition_auc"] else f"{rep['transition_auc']:.2f}")

t1, t2, t3, t4 = st.tabs(["Regimes", "Asset returns", "Transitions & outlook", "Data & model selection"])
with t1:
    fig = go.Figure(go.Scatter(x=g.index, y=g, line=dict(width=1.5), name="Growth score"))
    run = (labels != labels.shift()).cumsum()
    for _, seg in labels.groupby(run):
        fig.add_vrect(x0=seg.index[0], x1=seg.index[-1], fillcolor=PAL[int(seg.iloc[0])], opacity=0.25, line_width=0)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Composite growth (z)")
    st.plotly_chart(fig, use_container_width=True)
    fp = models.fingerprints(F, labels); fp.columns = tags
    st.subheader("Regime fingerprints (mean standardized level)")
    st.dataframe(fp.rename(index=C.NAMES).style.background_gradient(cmap="RdYlGn", axis=None).format("{:.2f}"), use_container_width=True)
with t2:
    fwd = st.toggle("Use next-month returns (actionable)", value=True)
    res = analysis.regime_returns(labels, rets, forward=fwd); res["Regime"] = res["regime"].map(dict(enumerate(tags)))
    st.plotly_chart(px.bar(res, x="asset", y="ann_return", color="Regime", barmode="group",
                           color_discrete_sequence=PAL, labels={"ann_return": "Annualized return"}).update_yaxes(tickformat=".0%"), use_container_width=True)
    st.dataframe(res.drop(columns="regime").round(3), use_container_width=True, hide_index=True)
with t3:
    T = models.transition_matrix(labels); T.index = T.columns = tags
    st.plotly_chart(px.imshow(T, text_auto=".0%", color_continuous_scale="Blues", aspect="auto"), use_container_width=True)
    P = predict.outlook(clf, F); d = st.select_slider("As-of month", options=list(F.index), value=F.index[-1], format_func=lambda x: f"{x:%Y-%m}")
    p = P.loc[d]; cur = int(labels.loc[d])
    st.write(f"**{d:%Y-%m}** is in **{tags[cur]}** ({'out-of-sample' if str(d.date()) >= rep['test_start'] else 'in-sample'}). "
             f"Probability of a different regime in {h} months: **{1 - p.get(cur, 0):.0%}**")
    st.bar_chart(pd.Series({tags[i]: v for i, v in p.items()}))
with t4:
    st.dataframe(macro.rename(columns=C.NAMES).tail(24).round(2), use_container_width=True)
    ks = models.select_k(F); st.subheader("Choosing k"); st.line_chart(ks["silhouette"]); st.line_chart(ks["gmm_bic"])
