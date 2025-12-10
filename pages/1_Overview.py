import math

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import get_dataset

try:
    df = get_dataset()
except FileNotFoundError:
    st.error("Packaged dataset missing. Please place 'Data_Sheet _Cleaned_Final.csv' beside app.py.")
    st.stop()

st.title("Overview Statistics")

# --- KPIs ---
col1, col2, col3 = st.columns(3)

col1.metric("Total Respondents", len(df))

if "Gender_num" in df.columns:
    male_pct = df["Gender_num"].mean() * 100
    col2.metric("Male (%)", f"{male_pct:.1f}%")
    col3.metric("Female (%)", f"{100 - male_pct:.1f}%")
else:
    col2.write("Gender unavailable")

# Age metric
if "Age" in df.columns:
    age = df["Age"].dropna()
    if not age.empty:
        st.metric("Age Range", f"{int(age.min())} – {int(age.max())}")

# --- Summary Statistics ---
st.subheader("Summary Statistics")

summary = df.describe().T
st.dataframe(summary)

# --- Histograms ---
st.subheader("Distribution Snapshots")

hist_targets = [
    ("Age", "Age"),
    ("Experience (Years)", "ExperienceYears"),
    ("Work Hours / Week", "HoursPerWeek"),
    ("Emotional Exhaustion", "EE"),
    ("Depersonalisation", "DP"),
    ("Personal Accomplishment", "PA"),
    ("Adaptability", "ADT"),
    ("Conscientiousness", "CST"),
    ("Positive Climate", "POS"),
    ("Autonomy", "AUT"),
    ("Workload", "WKL"),
    ("Neuroticism", "NEU"),
]

numeric_targets = [
    (label, col) for label, col in hist_targets
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
]

if numeric_targets:
    cols_per_row = 3
    rows = math.ceil(len(numeric_targets) / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 6.5, rows * 5.0))
    axes = axes.flatten()

    palette = sns.color_palette("viridis", len(numeric_targets))

    for ax, (label, col), color in zip(axes, numeric_targets, palette):
        sns.histplot(df[col].dropna(), kde=True, bins=20, ax=ax, color=color)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes[len(numeric_targets):]:
        ax.remove()

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
else:
    st.info("No numeric columns available for histogram view.")
