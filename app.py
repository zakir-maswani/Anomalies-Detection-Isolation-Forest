"""
Anomaly Detection using Isolation Forest — Streamlit App
==========================================================
A simple, interactive web app that wraps the modeling pipeline from
`data_preprocessing_and_model_training.ipynb`:

    load data -> standardize features -> Isolation Forest -> PCA visualization

Run with:
    streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Anomaly Detection | Isolation Forest",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Light styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem;}
        h1, h2, h3 {font-weight: 700;}
        div[data-testid="stMetric"] {
            background-color: rgba(120, 120, 120, 0.08);
            border-radius: 12px;
            padding: 14px 16px;
            border: 1px solid rgba(120, 120, 120, 0.15);
        }
        .stTabs [data-baseweb="tab-list"] {gap: 6px;}
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🔍 Anomaly Detection using Isolation Forest")
st.caption(
    "Upload any tabular CSV dataset and detect outliers with an unsupervised "
    "Isolation Forest model — standardized features, tunable hyperparameters, "
    "and an interactive PCA visualization."
)

# --------------------------------------------------------------------------
# Sidebar — configuration
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

    use_sample = False
    if uploaded_file is None:
        use_sample = st.checkbox("Use a generated sample dataset instead", value=True)

    st.subheader("Model Hyperparameters")
    n_estimators = st.slider("n_estimators", min_value=50, max_value=500, value=200, step=10)
    contamination = st.slider(
        "contamination (expected outlier fraction)",
        min_value=0.01,
        max_value=0.50,
        value=0.036,
        step=0.001,
        format="%.3f",
    )
    random_state = st.number_input("random_state", value=42, step=1)

    st.markdown("---")
    st.caption(
        "💡 **contamination** is your best estimate of the proportion of "
        "anomalies in the data. Lower it if you expect very few outliers."
    )

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def make_sample_dataset(n_samples: int = 800, n_features: int = 6, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic numeric dataset with a small injected anomaly cluster."""
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0.0, scale=1.0, size=(int(n_samples * 0.95), n_features))
    anomalies = rng.normal(loc=6.0, scale=1.5, size=(n_samples - normal.shape[0], n_features))
    data = np.vstack([normal, anomalies])
    rng.shuffle(data)
    cols = [f"feature_{i + 1}" for i in range(n_features)]
    return pd.DataFrame(data, columns=cols)


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def run_isolation_forest(
    df: pd.DataFrame,
    feature_cols: list,
    n_estimators: int,
    contamination: float,
    random_state: int,
):
    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    labels = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)

    return labels, scores, coords, pca


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
if uploaded_file is not None:
    df = load_csv(uploaded_file)
    data_source = uploaded_file.name
elif use_sample:
    df = make_sample_dataset()
    data_source = "synthetic sample dataset"
else:
    df = None
    data_source = None

if df is None:
    st.info("👈 Upload a CSV file from the sidebar (or enable the sample dataset) to get started.")
    st.stop()

# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

with st.sidebar:
    st.markdown("---")
    st.subheader("Feature Selection")
    exclude_cols = st.multiselect(
        "Exclude columns from training (e.g. an existing label column)",
        options=df.columns.tolist(),
        default=[c for c in df.columns if c.lower() in ("outlier_label", "label", "target", "class")],
    )

feature_cols = [c for c in numeric_cols if c not in exclude_cols]

if not feature_cols:
    st.error("No numeric feature columns available for training after exclusions. Please adjust your selection.")
    st.stop()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_results, tab_viz, tab_export = st.tabs(
    ["📁 Data Overview", "🚨 Detection Results", "📊 Visualization", "⬇️ Export"]
)

# ---- Tab 1: Data Overview -------------------------------------------------
with tab_overview:
    st.subheader(f"Dataset preview — `{data_source}`")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Features used", f"{len(feature_cols):,}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Missing values**")
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if nulls.empty:
            st.success("No missing values detected ✅")
        else:
            st.dataframe(nulls.rename("missing_count"), use_container_width=True)
    with col_b:
        st.markdown("**Statistical summary**")
        st.dataframe(df[feature_cols].describe().T, use_container_width=True)

    if non_numeric_cols:
        st.caption(f"Non-numeric columns ignored from modeling: {', '.join(non_numeric_cols)}")

# ---- Run model --------------------------------------------------------------
run = st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = None

if run:
    with st.spinner("Standardizing features and training Isolation Forest..."):
        labels, scores, coords, pca = run_isolation_forest(
            df, feature_cols, n_estimators, contamination, random_state
        )
        st.session_state.results = {
            "labels": labels,
            "scores": scores,
            "coords": coords,
            "explained_var": pca.explained_variance_ratio_,
        }
    st.toast("Detection complete!", icon="✅")

results = st.session_state.results

# ---- Tab 2: Detection Results -----------------------------------------------
with tab_results:
    if results is None:
        st.info("Click **🚀 Run Anomaly Detection** above to see results here.")
    else:
        labels = results["labels"]
        scores = results["scores"]
        n_outliers = int(np.sum(labels == -1))
        n_normal = int(np.sum(labels == 1))

        c1, c2, c3 = st.columns(3)
        c1.metric("Normal points", f"{n_normal:,}")
        c2.metric("Detected anomalies", f"{n_outliers:,}")
        c3.metric("Anomaly rate", f"{n_outliers / len(labels):.2%}")

        st.markdown("**Anomaly score distribution**")
        st.caption("More negative scores indicate stronger anomalies.")
        score_fig = px.histogram(
            x=scores,
            nbins=40,
            labels={"x": "Anomaly score"},
            color=(labels == -1),
            color_discrete_map={True: "#EF553B", False: "#636EFA"},
        )
        score_fig.update_layout(showlegend=False, height=350, margin=dict(t=10, b=10))
        st.plotly_chart(score_fig, use_container_width=True)

# ---- Tab 3: Visualization ---------------------------------------------------
with tab_viz:
    if results is None:
        st.info("Run the model first to view the PCA visualization.")
    else:
        coords = results["coords"]
        labels = results["labels"]
        var = results["explained_var"]

        plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        plot_df["Prediction"] = np.where(labels == -1, "Anomaly", "Normal")

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Prediction",
            color_discrete_map={"Normal": "#636EFA", "Anomaly": "#EF553B"},
            opacity=0.75,
            title="Isolation Forest results projected via PCA",
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
        fig.update_layout(height=520, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"PC1 explains {var[0]:.1%} of variance · PC2 explains {var[1]:.1%} of variance "
            f"(total: {sum(var):.1%}). This projection is for interpretation only and does not affect training."
        )

# ---- Tab 4: Export -----------------------------------------------------------
with tab_export:
    if results is None:
        st.info("Run the model first to export results.")
    else:
        export_df = df.copy()
        export_df["anomaly_score"] = results["scores"]
        export_df["prediction"] = np.where(results["labels"] == -1, "Anomaly", "Normal")

        st.dataframe(export_df.head(20), use_container_width=True)

        buffer = io.StringIO()
        export_df.to_csv(buffer, index=False)
        st.download_button(
            label="⬇️ Download results as CSV",
            data=buffer.getvalue(),
            file_name="anomaly_detection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit, scikit-learn & Plotly · Isolation Forest anomaly detection demo")
