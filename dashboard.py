"""
FL Robustness Dashboard — metric comparison & trade-off analysis.
Run with:  streamlit run dashboard.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="FL Robustness Dashboard", layout="wide", page_icon="🧠")

DATA_DIR = Path(__file__).parent / "data"

# ─── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_combo_data() -> pd.DataFrame:
    dfs = []
    for method in ("gtg", "l1o"):
        for f in glob.glob(str(DATA_DIR / "combo" / f"results_*_{method}.csv")):
            df = pd.read_csv(f)
            df["method"] = method
            # parse dataset/partition from filename
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

df_all = load_combo_data()

# base_dataset is already clean (the CSV "dataset" col never contains "noniid")
df_all["base_dataset"] = df_all["dataset"]
# Normalise partition label: "iid" → "IID", "noniid" → "non-IID"
df_all["partition"] = df_all["partition"].map({"iid": "IID", "noniid": "non-IID"}).fillna(df_all["partition"])

# ─── Metric catalogue ─────────────────────────────────────────────────────────
# global_col_fn(method)    → column name for the model-level metric
# contrib_col_fn(method)   → column name for the per-client contribution
METRICS = {
    "Certified Robustness":    "rob",
    "Adversarial Robustness":  "adv",
    "Fairness (Dem. Parity)":  "fair",
    "Fairness (Eq. Odds)":     "fair_eo",
    "Accuracy":                "acc",
    "Loss":                    "loss",
}

# Special: accuracy and loss live in plain columns, not global_method_* pattern
SPECIAL_GLOBAL = {"acc": "accuracy", "loss": "loss"}

def global_col(method: str, key: str) -> str:
    if key in SPECIAL_GLOBAL:
        return SPECIAL_GLOBAL[key]
    return f"global_{method}_{key}"

def contrib_col(method: str, key: str) -> str:
    if key == "acc":
        return "acc_contribution"
    if key == "loss":
        return "loss_contribution"
    return f"{method}_{key}_contribution"

METRIC_LABELS = list(METRICS.keys())
METRIC_KEYS   = list(METRICS.values())

# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🔧 Filters")

BASE_DATASETS = sorted(df_all["base_dataset"].unique().tolist())
dataset_base  = st.sidebar.selectbox("Dataset", BASE_DATASETS)
method_choice = st.sidebar.selectbox("Method", ["GTG", "L1O"])
partitions    = st.sidebar.multiselect("Partition", ["IID", "non-IID"], default=["IID", "non-IID"])
all_nc        = sorted(df_all["num_clients"].unique().tolist())
nc_filter     = st.sidebar.multiselect("Num clients", all_nc, default=all_nc)
all_seeds     = sorted(df_all["seed"].unique().tolist())
seed_filter   = st.sidebar.multiselect("Seeds", all_seeds, default=all_seeds)

method = method_choice.lower()

def base_filter(extra_partition: str | None = None) -> pd.DataFrame:
    parts = [extra_partition] if extra_partition else (partitions or ["IID", "non-IID"])
    return df_all[
        (df_all["base_dataset"] == dataset_base)
        & (df_all["method"] == method)
        & (df_all["partition"].isin(parts))
        & (df_all["num_clients"].isin(nc_filter or all_nc))
        & (df_all["seed"].isin(seed_filter or all_seeds))
    ].copy()

# ─── Page header ──────────────────────────────────────────────────────────────

st.title("🧠 FL Robustness — Metric Comparison Dashboard")
st.caption(
    f"Dataset: **{dataset_base}** · Method: **{method_choice}** · "
    f"Partitions: **{partitions}** · Clients: **{nc_filter}** · Seeds: **{seed_filter}**"
)

tab_over, tab_tradeoff, tab_radar, tab_corr, tab_client = st.tabs([
    "📊 All Metrics",
    "🔀 Trade-off Scatter",
    "🕸️  Radar Profiles",
    "📈 Correlation Matrix",
    "🏆 Client Profiles",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 · All global metrics over rounds — normalised on same axis
# ═══════════════════════════════════════════════════════════════════════════════

with tab_over:
    st.subheader("All global metrics over training rounds")

    df = base_filter()
    if df.empty:
        st.warning("No data for current selection.")
    else:
        nc_opts = sorted(df["num_clients"].unique().tolist())
        nc_sel  = st.selectbox("Num clients", nc_opts, key="ov_nc")
        norm    = st.toggle("Normalise 0–1 per metric (for visual comparison)", value=True)

        df_g = (
            df[df["num_clients"] == nc_sel]
            .drop_duplicates(subset=["partition", "seed", "round"])
        )

        # Build one row per (partition, seed, round) with all metric values
        records = []
        for _, row in df_g.iterrows():
            base_row = {"partition": row["partition"], "seed": row["seed"], "round": row["round"]}
            for lbl, key in METRICS.items():
                col = global_col(method, key)
                if col in df_g.columns:
                    base_row[lbl] = row[col]
            records.append(base_row)

        df_long = pd.DataFrame(records).melt(
            id_vars=["partition", "seed", "round"],
            var_name="Metric", value_name="value"
        )

        # Average over seeds
        df_mean = df_long.groupby(["partition", "round", "Metric"])["value"].agg(["mean", "std"]).reset_index()

        if norm:
            for metric in df_mean["Metric"].unique():
                mask = df_mean["Metric"] == metric
                v = df_mean.loc[mask, "mean"]
                vmin, vmax = v.min(), v.max()
                if vmax > vmin:
                    df_mean.loc[mask, "mean"] = (v - vmin) / (vmax - vmin)
                    df_mean.loc[mask, "std"]  = df_mean.loc[mask, "std"] / (vmax - vmin)

        fig = px.line(
            df_mean, x="round", y="mean", color="Metric",
            line_dash="partition",
            error_y="std",
            labels={"round": "Round", "mean": "Value (normalised)" if norm else "Value"},
            markers=True, height=480,
        )
        fig.update_layout(hovermode="x unified", legend_title="Metric / Partition")
        st.plotly_chart(fig, use_container_width=True)

        # Small multiples: one chart per metric, un-normalised
        st.subheader("Individual metric trends")
        cols_per_row = 3
        metric_items = [(lbl, key) for lbl, key in METRICS.items()
                        if global_col(method, key) in df.columns]
        rows = [metric_items[i:i+cols_per_row] for i in range(0, len(metric_items), cols_per_row)]
        for row_items in rows:
            cols = st.columns(len(row_items))
            for (lbl, key), col in zip(row_items, cols):
                with col:
                    gcol = global_col(method, key)
                    sub = df_mean[df_mean["Metric"] == lbl].copy()
                    # re-use raw mean (need un-normalised → re-compute)
                    sub2 = df_long[df_long["Metric"] == lbl].groupby(
                        ["partition", "round"])["value"].agg(["mean","std"]).reset_index()
                    fig_s = px.line(
                        sub2, x="round", y="mean", color="partition",
                        error_y="std", markers=True,
                        title=lbl,
                        labels={"round": "Round", "mean": lbl},
                        height=260,
                    )
                    fig_s.update_layout(showlegend=(lbl == metric_items[0][0]),
                                        margin=dict(t=40, b=20))
                    st.plotly_chart(fig_s, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 · Trade-off scatter — global metric A vs global metric B
# ═══════════════════════════════════════════════════════════════════════════════

with tab_tradeoff:
    st.subheader("Trade-off between two global metrics")
    st.markdown("Each point is one *(seed × round × partition)* tuple, averaged over clients.")

    c1, c2, c3 = st.columns(3)
    x_label = c1.selectbox("X axis", METRIC_LABELS, index=0, key="tr_x")
    y_label = c2.selectbox("Y axis", METRIC_LABELS, index=2, key="tr_y")
    color_by = c3.selectbox("Color by", ["round", "partition", "seed", "num_clients"], key="tr_color")

    df = base_filter()
    x_key, y_key = METRICS[x_label], METRICS[y_label]
    x_col = global_col(method, x_key)
    y_col = global_col(method, y_key)

    if df.empty or x_col not in df.columns or y_col not in df.columns:
        st.warning("Columns not found for current selection.")
    else:
        df_g = df.drop_duplicates(subset=["partition", "seed", "num_clients", "round"])

        fig = px.scatter(
            df_g, x=x_col, y=y_col,
            color=color_by,
            symbol="partition",
            hover_data=["seed", "round", "partition", "num_clients"],
            labels={x_col: x_label, y_col: y_label},
            trendline="ols",
            trendline_scope="overall",
            color_continuous_scale="Viridis",
            height=480,
        )
        fig.update_traces(marker=dict(size=8, opacity=0.75))
        st.plotly_chart(fig, use_container_width=True)

        # Pearson correlation summary
        r = df_g[[x_col, y_col]].corr().iloc[0, 1]
        direction = "positive" if r > 0 else "negative"
        strength = "strong" if abs(r) > 0.7 else ("moderate" if abs(r) > 0.4 else "weak")
        st.info(f"**Pearson r = {r:.3f}** — {strength} {direction} correlation between "
                f"*{x_label}* and *{y_label}* across all rounds/seeds.")

        # ── Contribution-level scatter ───────────────────────────────────────
        st.divider()
        st.subheader("Client-level contribution trade-off")
        st.markdown("Each point is one *(client × seed × round)* — shows if clients who help "
                    "one metric hurt another.")

        nc_tr  = st.selectbox("Num clients", sorted(df["num_clients"].unique()), key="tr_nc")
        df_c   = df[df["num_clients"] == nc_tr]
        cx_col = contrib_col(method, x_key)
        cy_col = contrib_col(method, y_key)

        if cx_col not in df_c.columns or cy_col not in df_c.columns:
            st.warning("Contribution columns not available.")
        else:
            fig2 = px.scatter(
                df_c, x=cx_col, y=cy_col,
                color="client_id",
                symbol="partition",
                hover_data=["client_id", "seed", "round"],
                labels={cx_col: f"{x_label} contribution", cy_col: f"{y_label} contribution"},
                opacity=0.6,
                height=430,
                trendline="ols",
                trendline_scope="overall",
            )
            # Reference lines at 0
            fig2.add_hline(y=0, line_dash="dot", line_color="gray")
            fig2.add_vline(x=0, line_dash="dot", line_color="gray")
            fig2.update_layout(coloraxis_colorbar_title="Client")
            st.plotly_chart(fig2, use_container_width=True)

            r2 = df_c[[cx_col, cy_col]].corr().iloc[0, 1]
            st.info(f"**Client-level Pearson r = {r2:.3f}**")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 · Radar profiles — metric fingerprint per dataset / round
# ═══════════════════════════════════════════════════════════════════════════════

with tab_radar:
    st.subheader("Radar profiles — metric fingerprints")

    df = base_filter()
    if df.empty:
        st.warning("No data for current selection.")
    else:
        radar_mode = st.radio(
            "Compare across", ["Rounds (first vs last)", "Partitions (IID vs non-IID)", "Num clients"],
            horizontal=True,
        )
        nc_r = st.selectbox("Num clients", sorted(df["num_clients"].unique()), key="rad_nc")
        df_r = df[df["num_clients"] == nc_r]

        # Collect available global metric columns
        avail_metrics = [(lbl, key) for lbl, key in METRICS.items()
                         if global_col(method, key) in df_r.columns]

        def radar_trace(sub: pd.DataFrame, name: str, color: str) -> go.Scatterpolar:
            """Build one normalised radar trace from mean global metrics."""
            values = []
            labels = []
            for lbl, key in avail_metrics:
                col = global_col(method, key)
                values.append(sub[col].mean())
                labels.append(lbl)
            # normalise 0–1 globally across dataset for fair visual
            values_arr = np.array(values, dtype=float)
            # close the polygon
            values_arr = np.append(values_arr, values_arr[0])
            labels_closed = labels + [labels[0]]
            return go.Scatterpolar(
                r=values_arr, theta=labels_closed, name=name,
                fill="toself", opacity=0.6,
                line=dict(color=color),
            )

        COLORS = px.colors.qualitative.Plotly

        if radar_mode == "Rounds (first vs last)":
            rounds = sorted(df_r["round"].unique())
            groups = [
                (df_r[df_r["round"] == rounds[0]],  f"Round {rounds[0]}"),
                (df_r[df_r["round"] == rounds[-1]], f"Round {rounds[-1]}"),
            ]
        elif radar_mode == "Partitions (IID vs non-IID)":
            groups = [
                (df_r[df_r["partition"] == "IID"],     "IID"),
                (df_r[df_r["partition"] == "non-IID"], "non-IID"),
            ]
        else:
            groups = [
                (df_r[df_r["num_clients"] == nc], f"{nc} clients")
                for nc in sorted(df["num_clients"].unique())
            ]

        fig_rad = go.Figure()
        for i, (sub, name) in enumerate(groups):
            if not sub.empty:
                fig_rad.add_trace(radar_trace(sub, name, COLORS[i % len(COLORS)]))

        fig_rad.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            legend_title="Group",
            height=480,
        )
        st.plotly_chart(fig_rad, use_container_width=True)

        # Multi-dataset radar comparison
        st.divider()
        st.subheader("Cross-dataset comparison at last round")
        last_r = df_all["round"].max()
        df_cross = df_all[
            (df_all["method"] == method)
            & (df_all["num_clients"].isin(nc_filter or all_nc))
            & (df_all["seed"].isin(seed_filter or all_seeds))
            & (df_all["round"] == last_r)
            & (df_all["partition"].isin(partitions or ["IID","non-IID"]))
        ]

        fig_cross = go.Figure()
        for i, (ds, grp) in enumerate(df_cross.groupby("dataset")):
            values, labels = [], []
            for lbl, key in avail_metrics:
                col = global_col(method, key)
                if col in grp.columns:
                    values.append(grp[col].mean())
                    labels.append(lbl)
            if values:
                v = np.append(values, values[0])
                l = labels + [labels[0]]
                fig_cross.add_trace(go.Scatterpolar(
                    r=v, theta=l, name=ds, fill="toself",
                    opacity=0.55, line=dict(color=COLORS[i % len(COLORS)]),
                ))

        fig_cross.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            height=480, legend_title="Dataset",
        )
        st.plotly_chart(fig_cross, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 · Correlation matrix between metric contributions
# ═══════════════════════════════════════════════════════════════════════════════

with tab_corr:
    st.subheader("Correlation between metric contributions")
    st.markdown(
        "Reveals whether clients that contribute positively to one metric also "
        "help (or hurt) the others — key for understanding multi-objective trade-offs."
    )

    df = base_filter()
    nc_co = st.selectbox("Num clients", sorted(df["num_clients"].unique()), key="co_nc") if not df.empty else None

    if df.empty or nc_co is None:
        st.warning("No data.")
    else:
        df_co = df[df["num_clients"] == nc_co]
        # Build a DataFrame of contribution columns only
        contrib_cols = []
        col_labels   = []
        for lbl, key in METRICS.items():
            cc = contrib_col(method, key)
            if cc in df_co.columns:
                contrib_cols.append(cc)
                col_labels.append(lbl)

        df_contribs = df_co[contrib_cols].copy()
        df_contribs.columns = col_labels

        corr_mat = df_contribs.corr()

        fig_corr = px.imshow(
            corr_mat,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            zmin=-1, zmax=1,
            text_auto=".2f",
            labels=dict(color="Pearson r"),
            height=480,
        )
        fig_corr.update_layout(
            xaxis_title="", yaxis_title="",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Highlight the strongest trade-offs
        st.subheader("Top metric pairs by |correlation|")
        pairs = []
        for i, a in enumerate(col_labels):
            for j, b in enumerate(col_labels):
                if j > i:
                    r_val = corr_mat.iloc[i, j]
                    pairs.append({"Metric A": a, "Metric B": b, "Pearson r": round(r_val, 4)})
        df_pairs = pd.DataFrame(pairs).sort_values("Pearson r", ascending=True)
        # colour negative (trade-offs) red, positive green
        def colour_r(val):
            if val < -0.3:
                return "background-color: #ffcccc"
            if val > 0.3:
                return "background-color: #ccffcc"
            return ""
        st.dataframe(
            df_pairs.style.applymap(colour_r, subset=["Pearson r"]),
            use_container_width=True, hide_index=True,
        )

        # Scatter matrix for the contribution columns
        st.divider()
        st.subheader("Scatter matrix")
        fig_sm = px.scatter_matrix(
            df_contribs,
            dimensions=col_labels,
            opacity=0.4,
            height=700,
        )
        fig_sm.update_traces(diagonal_visible=False, marker=dict(size=3))
        st.plotly_chart(fig_sm, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 · Client profiles — per-client multi-metric contribution
# ═══════════════════════════════════════════════════════════════════════════════

with tab_client:
    st.subheader("Client contribution profiles across all metrics")

    df = base_filter()
    if df.empty:
        st.warning("No data.")
    else:
        nc_cl   = st.selectbox("Num clients", sorted(df["num_clients"].unique()), key="cl_nc")
        round_cl = st.slider("Round", 1, int(df["round"].max()), int(df["round"].max()), key="cl_round")
        part_cl  = st.radio("Partition", [p for p in ["IID", "non-IID"] if p in df["partition"].unique()],
                            horizontal=True, key="cl_part")

        df_cl = df[
            (df["num_clients"] == nc_cl)
            & (df["round"] == round_cl)
            & (df["partition"] == part_cl)
        ]

        contrib_cols = []
        col_labels   = []
        for lbl, key in METRICS.items():
            cc = contrib_col(method, key)
            if cc in df_cl.columns:
                contrib_cols.append(cc)
                col_labels.append(lbl)

        if df_cl.empty or not contrib_cols:
            st.warning("No data for selection.")
        else:
            # Average over seeds
            df_avg = df_cl.groupby("client_id")[contrib_cols].mean().reset_index()
            df_avg.columns = ["client_id"] + col_labels

            # ── Grouped bar: all metrics side-by-side per client ─────────────
            st.markdown("**Grouped bar — all metric contributions per client** *(avg over seeds)*")
            df_melt = df_avg.melt(id_vars="client_id", var_name="Metric", value_name="contribution")
            fig_bar = px.bar(
                df_melt, x="client_id", y="contribution",
                color="Metric", barmode="group",
                labels={"client_id": "Client ID", "contribution": "Contribution"},
                height=420,
            )
            fig_bar.add_hline(y=0, line_color="black", line_width=1)
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Radar per client ─────────────────────────────────────────────
            st.markdown("**Radar — metric profile per client**")
            COLORS = px.colors.qualitative.Plotly
            n_clients = len(df_avg["client_id"].unique())
            cols_r = min(4, n_clients)
            rows_r = (n_clients + cols_r - 1) // cols_r

            fig_rad = make_subplots(
                rows=rows_r, cols=cols_r,
                specs=[[{"type": "polar"}] * cols_r for _ in range(rows_r)],
                subplot_titles=[f"Client {i}" for i in sorted(df_avg["client_id"].unique())],
            )
            for idx, (_, row) in enumerate(df_avg.sort_values("client_id").iterrows()):
                r_idx = idx // cols_r + 1
                c_idx = idx % cols_r + 1
                vals = [row[lbl] for lbl in col_labels]
                vals_closed = vals + [vals[0]]
                labels_closed = col_labels + [col_labels[0]]
                fig_rad.add_trace(
                    go.Scatterpolar(
                        r=vals_closed, theta=labels_closed,
                        fill="toself", opacity=0.55,
                        line=dict(color=COLORS[idx % len(COLORS)]),
                        name=f"Client {int(row['client_id'])}",
                        showlegend=False,
                    ),
                    row=r_idx, col=c_idx,
                )
            fig_rad.update_layout(height=260 * rows_r)
            st.plotly_chart(fig_rad, use_container_width=True)

            # ── Heatmap: clients × metrics ───────────────────────────────────
            st.markdown("**Heatmap — contribution magnitude per client × metric**")
            heat_data = df_avg.set_index("client_id")[col_labels]
            fig_heat = px.imshow(
                heat_data.T,
                labels=dict(x="Client ID", y="Metric", color="Contribution"),
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                aspect="auto",
                height=320,
            )
            st.plotly_chart(fig_heat, use_container_width=True)
