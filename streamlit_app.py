"""
Streamlit Dashboard – Credit‑Card Fraud Detection
Author: Naushin Uddin

**What the app does**
1. Loads the Kaggle credit‑card fraud dataset (or a user‑uploaded CSV).
2. Shows an at‑a‑glance EDA section.
3. Lets the user pick a ML model (Logistic Regression or Random Forest).
4. Trains & evaluates the model, then visualises performance.
"""

# imports
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# configuration

st.set_page_config(
    page_title="Credit Fraud Detection",
    layout="wide",
)

PASTEL_CSS = """
<style>
body {
    background-color:#fdfdfd;
}
[data-testid="stHeader"] {
    background: linear-gradient(90deg,#e9f3ff,#fff0f5);
}
[data-testid="stSidebar"] > div:first-child {
    background-color:#faf8ff;
}
</style>
"""
st.markdown(PASTEL_CSS, unsafe_allow_html=True)

st.markdown("# Credit‑Card Fraud Detection", unsafe_allow_html=True)

# helper functions
@st.cache_data(show_spinner=True)
def load_default_dataset() -> pd.DataFrame:
    """Load creditcard.csv from the app folder (works locally & when deployed)."""
    csv_path = "creditcard.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("creditcard.csv not found in the app directory.")
    return pd.read_csv(csv_path)

def pastelize_fig(fig):
    """Apply a pastel face‑color to matplotlib fig for a cohesive look."""
    fig.patch.set_facecolor("#fffbff")
    return fig

with st.expander("## What This App Does & Why It Matters", expanded=False):
    st.markdown("""
This dashboard demonstrates how machine learning can help detect fraudulent credit-card transactions in a real-world, highly imbalanced dataset. Credit card fraud is a costly problem that affects consumers and institutions alike.

In this app I:
- Load a Kaggle dataset of ~285,000 real transactions.
- Apply feature scaling and synthetic oversampling to fix the imbalance between legitimate and fraudulent transactions.
- Let the user choose between two machine learning models (Logistic Regression or Random Forest).
- Train, test, and evaluate the model with metrics like accuracy, AUC, and feature importance.
- Visualize results in a clean interface to make data science accessible, aesthetic, and intuitive.
""")

#  sidebar
st.sidebar.header("Data Source")
user_file = st.sidebar.file_uploader("Upload your own CSV (optional)", type=["csv"])

with st.sidebar.expander("ℹ️  How to use", expanded=False):
    st.markdown(
        "Upload a CSV **with the same schema** as the Kaggle fraud dataset or just leave this blank to use the default 285 k‑row dataset."
    )

fast_mode = st.sidebar.checkbox("Fast mode (train on 50 k rows)", value=False)

# load data
try:
    df = load_default_dataset() if user_file is None else pd.read_csv(user_file)
except Exception as e:
    st.error(f"Hmmm...It looks like we failed to load your dataset: {e}")
    st.stop()

# Preview
st.subheader("Quick Peek of Data")
st.dataframe(df.head(), use_container_width=True)

col_dist1, col_dist2 = st.columns(2)
with col_dist1:
    st.metric("Total Transactions", f"{len(df):,}")
with col_dist2:
    fraud_rate = df["Class"].mean() * 100
    st.metric("% Fraudulent", f"{fraud_rate:.2f}%")

# Class distribution bar
fig_bar, ax_bar = plt.subplots(figsize=(3, 2))
sns.countplot(x=df["Class"], palette=["#b4e0fe", "#ffb7ce"], ax=ax_bar)
ax_bar.set_title("Class Distribution (0 = Legit, 1 = Fraud)")
ax_bar.set_xlabel("")
ax_bar.set_ylabel("Count")
fig_bar = pastelize_fig(fig_bar)
st.pyplot(fig_bar, use_container_width=True)

# preprocess data
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale features – StdScaler handles anonymised PCA components well
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Oversample minority class
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Sample down for fast‑mode
if fast_mode and len(X_res) > 50_000:
    X_res = X_res[:50_000]
    y_res = y_res[:50_000]

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# choosing model option
st.subheader("Model Selection & Training")
model_name = st.radio("Choose model", ["Logistic Regression", "Random Forest"], horizontal=True)

if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=2000)
else:
    n_trees = st.slider("Number of Trees", min_value=20, max_value=200, value=80, step=20)
    model = RandomForestClassifier(
        n_estimators=n_trees, random_state=42, n_jobs=-1, max_depth=None
    )

train_btn = st.button("Click Here To Train Your Model!!")

if train_btn:
    with st.spinner("Training… this could take ~1–2 min on full data"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    st.success("Model training finished!")

    # the metrics section
    st.subheader("Performance Metrics")

    col_report, col_matrix = st.columns([1, 1])

    with col_report:
        st.markdown("**Classification Report**")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().round(2)
        st.dataframe(report_df, use_container_width=True)

    with col_matrix:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=sns.light_palette("#ffb7ce", as_cmap=True),
            cbar=False,
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        fig_cm = pastelize_fig(fig_cm)
        st.pyplot(fig_cm)

    # ROC Curve
    st.markdown("**ROC Curve**")
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color="#b39cfd", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#d0d0d0")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    fig_roc = pastelize_fig(fig_roc)
    st.pyplot(fig_roc)

    # Feature Importance
    if model_name == "Random Forest":
        st.markdown("**Top 10 Feature Importances (RF)**")
        importances = model.feature_importances_
        feat_imp = (
            pd.Series(importances, index=X.columns)
            .sort_values(ascending=False)
            .head(10)
            .sort_values()
        )
        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
        feat_imp.plot.barh(color="#a3e4d7", ax=ax_imp)
        ax_imp.set_xlabel("Gini Importance")
        fig_imp = pastelize_fig(fig_imp)
        st.pyplot(fig_imp, use_container_width=True)

st.caption("Made by Naushin Uddin · DS 2010 2025")
