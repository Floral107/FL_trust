"""RobShap — per-client trustworthiness contribution dashboard.

Run locally:   streamlit run demo/app.py
Run in Docker: docker build -t robshap-demo demo/ && docker run -p 8501:8501 robshap-demo
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from robshap_demo.loader import DIMENSIONS, METHODS, TRADEOFF_DIMS, load_results, tradeoff_matrix
from robshap_demo.scoring import score_clients

DATA_DIR = os.environ.get("ROBSHAP_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "sample_data", "clients_demo.csv")

st.set_page_config(page_title="RobShap — FL contribution scoring", page_icon="⚖️", layout="wide")

st.title("⚖️ RobShap — who helps, who hurts, and at what cost?")
st.caption(
    "Per-client contribution scoring for federated learning across **robustness, "
    "adversarial resilience, fairness, and privacy** — Leave-One-Out and GTG-Shapley. "
    "Built on the FLRobustness thesis pipeline (BME)."
)

tab_explore, tab_live, tab_about = st.tabs(
    ["📊 Explore paper results", "🚀 Score your own clients", "ℹ️ About"]
)


# --------------------------------------------------------------------------- Explore
@st.cache_data(show_spinner="Loading precomputed results…")
def _cached_results(data_dir: str) -> pd.DataFrame:
    return load_results(data_dir)


with tab_explore:
    df = _cached_results(DATA_DIR)
    if df.empty:
        st.error(f"No precomputed results found in `{DATA_DIR}`.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        dataset = c1.selectbox("Dataset", sorted(df["dataset"].unique()), index=0)
        sub0 = df[df["dataset"] == dataset]
        partition = c2.selectbox("Partition", sorted(sub0["partition"].unique()))
        sub0 = sub0[sub0["partition"] == partition]
        method = c3.selectbox("Scoring method", sorted(sub0["method"].unique()),
                              format_func=lambda m: METHODS.get(m, m))
        sub0 = sub0[sub0["method"] == method]
        num_clients = c4.selectbox("Clients", sorted(sub0["num_clients"].unique()))
        sub0 = sub0[sub0["num_clients"] == num_clients]
        rounds = sorted(sub0["round"].unique())
        round_sel = c5.select_slider("Round", options=rounds, value=rounds[-1])
        sub = sub0[sub0["round"] == round_sel]

        st.markdown(
            f"Contributions averaged over **{sub['seed'].nunique()} seeds** · "
            f"{METHODS.get(method, method)} · round {round_sel}"
        )

        avail_dims = [d for d in TRADEOFF_DIMS if d in sub["dimension"].unique()]
        wide = tradeoff_matrix(sub, avail_dims)

        left, right = st.columns([3, 2])

        with left:
            st.subheader("Trade-off map")
            if len(avail_dims) >= 2:
                xdim = st.selectbox("X axis", avail_dims, index=0,
                                    format_func=lambda d: DIMENSIONS[d], key="xdim")
                ydim = st.selectbox("Y axis", [d for d in avail_dims if d != xdim], index=0,
                                    format_func=lambda d: DIMENSIONS[d], key="ydim")
                color_dim = next((d for d in avail_dims if d not in (xdim, ydim)), None)
                fig = px.scatter(
                    wide, x=xdim, y=ydim,
                    color=color_dim, color_continuous_scale="RdYlGn",
                    text=wide["client_id"].astype(str),
                    labels={d: DIMENSIONS[d] + " contribution" for d in avail_dims},
                )
                fig.update_traces(textposition="top center", marker_size=12)
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.add_vline(x=0, line_dash="dot", line_color="gray")
                if color_dim:
                    fig.update_coloraxes(colorbar_title=DIMENSIONS[color_dim])
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Each point is a client. Top-right quadrant helps on both axes; "
                    "off-diagonal clients are the trade-off: they buy one property "
                    "at the cost of another."
                )
            else:
                st.info("Fewer than two trade-off dimensions available for this slice.")

        with right:
            st.subheader("Client detail")
            cid = st.selectbox("Client", wide["client_id"].tolist())
            row = wide[wide["client_id"] == cid].iloc[0]
            dims_present = [d for d in avail_dims if pd.notna(row.get(d))]
            if dims_present:
                # Normalize each axis to the client population so the radar shows
                # relative standing (min-max per dimension), not raw magnitudes.
                norm = {}
                for d in dims_present:
                    col = wide[d]
                    span = (col.max() - col.min()) or 1.0
                    norm[d] = float((row[d] - col.min()) / span)
                fig = go.Figure(go.Scatterpolar(
                    r=[norm[d] for d in dims_present] + [norm[dims_present[0]]],
                    theta=[DIMENSIONS[d] for d in dims_present] + [DIMENSIONS[dims_present[0]]],
                    fill="toself",
                ))
                fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
                                  height=330, margin=dict(t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Relative standing vs. the other clients (per-dimension min–max).")

        st.subheader("All clients")
        table = wide.set_index("client_id").rename(columns=DIMENSIONS)
        st.dataframe(
            table.style.background_gradient(cmap="RdYlGn", axis=0).format("{:+.4f}"),
            use_container_width=True,
        )

        st.subheader("Contribution across rounds")
        dim_t = st.selectbox("Dimension", avail_dims, format_func=lambda d: DIMENSIONS[d], key="tdim")
        tl = (
            sub0[sub0["dimension"] == dim_t]
            .groupby(["round", "client_id"])["contribution"].mean().reset_index()
        )
        fig = px.line(tl, x="round", y="contribution", color=tl["client_id"].astype(str),
                      labels={"color": "client", "contribution": DIMENSIONS[dim_t] + " contribution"})
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- Live
with tab_live:
    st.markdown(
        "Upload a **CSV** with one column identifying the client each row belongs to, plus a "
        "target column. The app trains a fast surrogate model per leave-one-out coalition and "
        "computes **real L1O contribution scores** for accuracy, loss, noise robustness, and "
        "(optionally) demographic-parity / equalized-odds fairness — in seconds, on CPU."
    )

    up = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("…or use the bundled sample (5 clients, one label-flipped, one biased)",
                             value=up is None)

    raw = None
    if up is not None:
        raw = pd.read_csv(up)
        use_sample = False
    elif use_sample and os.path.exists(SAMPLE_CSV):
        raw = pd.read_csv(SAMPLE_CSV)

    if raw is not None:
        st.dataframe(raw.head(5), use_container_width=True)
        cols = list(raw.columns)
        c1, c2, c3 = st.columns(3)
        guess_client = next((c for c in cols if "client" in c.lower()), cols[0])
        guess_target = next((c for c in cols if c.lower() in ("target", "label", "y", "income")), cols[-1])
        client_col = c1.selectbox("Client column", cols, index=cols.index(guess_client))
        target_col = c2.selectbox("Target column", cols, index=cols.index(guess_target))
        sens_opts = ["(none)"] + [c for c in cols if c not in (client_col, target_col)]
        guess_sens = next((c for c in sens_opts if c.lower() in ("sex", "gender", "race", "group")), "(none)")
        sensitive_col = c3.selectbox("Sensitive attribute (for fairness)", sens_opts,
                                     index=sens_opts.index(guess_sens))
        sensitive_col = None if sensitive_col == "(none)" else sensitive_col

        if st.button("Score clients", type="primary"):
            try:
                with st.spinner("Training leave-one-out coalitions…"):
                    res = score_clients(raw, client_col, target_col, sensitive_col)
            except ValueError as e:
                st.error(str(e))
            else:
                for w in res.warnings:
                    st.warning(w)

                gcols = st.columns(len(res.global_metrics))
                for (name, val), col in zip(res.global_metrics.items(), gcols):
                    col.metric(f"Global {name.replace('_', ' ')}", f"{val:.4f}")
                st.caption(f"All-clients model · {res.n_test} held-out test rows")

                contrib = res.contributions
                st.subheader("Per-client L1O contributions")
                st.caption(
                    "contribution = metric(all clients) − metric(all except this client). "
                    "Positive = the client improves that property; negative = it hurts."
                )
                st.dataframe(
                    contrib.style.background_gradient(cmap="RdYlGn", axis=0).format("{:+.4f}"),
                    use_container_width=True,
                )

                melted = contrib.reset_index().melt(id_vars="client_id",
                                                    var_name="metric", value_name="contribution")
                fig = px.bar(melted, x="client_id", y="contribution", color="metric",
                             barmode="group",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.add_hline(y=0, line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

                if {"accuracy", "fairness_dp"}.issubset(contrib.columns):
                    st.subheader("Accuracy vs. fairness trade-off")
                    fig = px.scatter(contrib.reset_index(), x="accuracy", y="fairness_dp",
                                     text="client_id",
                                     labels={"accuracy": "Accuracy contribution",
                                             "fairness_dp": "Fairness (DP) contribution"})
                    fig.update_traces(textposition="top center", marker_size=12)
                    fig.add_hline(y=0, line_dash="dot", line_color="gray")
                    fig.add_vline(x=0, line_dash="dot", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "Download scores (CSV)",
                    contrib.to_csv().encode(),
                    file_name="robshap_client_scores.csv",
                    mime="text/csv",
                )

    st.divider()
    st.caption(
        "**Honest scope note:** live scoring uses a fast logistic-regression surrogate and "
        "covers the cheap axes (accuracy, loss, noise robustness, fairness). The paper's "
        "heavy axes — certified robustness, PGD adversarial, MIA privacy — need GPU-hours "
        "per coalition and are shown from precomputed research runs in the Explore tab."
    )


# --------------------------------------------------------------------------- About
with tab_about:
    st.markdown(
        """
### What is this?

A clickable front-end for the **FLRobustness** master's-thesis pipeline (BME): measuring how
much each federated-learning client contributes to — or detracts from — the *trustworthiness*
of the global model, not just its accuracy.

**Scoring methods**
- **Leave-One-Out (L1O):** `contribution_i = Metric(all) − Metric(all \\ i)`
- **GTG-Shapley:** permutation-sampled Shapley values with truncation, amortized across rounds

**Trustworthiness dimensions** (from the paper, computed on saved FL checkpoints)
| Dimension | Metric |
|---|---|
| Certified robustness | consistency under Gaussian noise (randomized smoothing, σ=0.1) |
| Adversarial | accuracy retained under PGD attack |
| Fairness | demographic parity / equalized odds gap (sensitive attr: sex) |
| Privacy | membership-inference resistance, `1 − 2·max(0, AUC − ½)` |

**Datasets:** CIFAR-10, IMDB, Adult, CelebA · IID and non-IID partitions · 4 & 20 clients · 5 seeds.

### Architecture

- `robshap_demo/loader.py` — normalizes the research pipeline's CSV outputs
- `robshap_demo/scoring.py` — live L1O scoring with sklearn surrogates (CPU, seconds)
- Research code (Flower + TensorFlow, GPU) stays untouched; this app only consumes its outputs.

Container: `docker build -t robshap-demo .` → `docker run -p 8501:8501 robshap-demo`.
CI runs lint + tests + docker build on every push.
        """
    )
