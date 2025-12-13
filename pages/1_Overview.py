import math

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# --- Scale Reliability (Cronbach's Alpha) ---
st.subheader("Scale Reliability (Cronbach's α)")

scale_definitions = {
    "Adaptability": "ADT",
    "Extraversion": "EXT",
    "Agreeableness": "AGR",
    "Conscientiousness": "CST",
    "Neuroticism": "NEU",
    "Openness": "OPE",
    "Emotional Exhaustion": "EE",
    "Depersonalisation": "DP",
    "Personal Accomplishment": "PA",
    "Autonomy": "AUT",
    "Workload": "WKL",
    "Perceived Organizational Support": "POS",
}

def cronbach_alpha(item_data: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for a set of items."""
    item_data = item_data.dropna()
    if item_data.empty or item_data.shape[1] < 2:
        return np.nan
    
    item_vars = item_data.var(axis=0, ddof=1)
    total_var = item_data.sum(axis=1).var(ddof=1)
    n_items = item_data.shape[1]
    
    if total_var == 0:
        return np.nan
    
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha

reliability_results = []
for scale_name, prefix in scale_definitions.items():
    item_cols = [c for c in df.columns if c.upper().startswith(prefix.upper()) and c != prefix and not c.endswith("_c")]
    if len(item_cols) >= 2:
        item_data = df[item_cols].apply(pd.to_numeric, errors="coerce")
        alpha = cronbach_alpha(item_data)
        reliability_results.append({
            "Scale": scale_name,
            "Items": len(item_cols),
            "Cronbach's α": f"{alpha:.3f}" if not np.isnan(alpha) else "N/A"
        })

if reliability_results:
    reliability_df = pd.DataFrame(reliability_results)
    st.dataframe(reliability_df, use_container_width=True, hide_index=True)

# --- Histograms ---
st.subheader("Distribution Snapshots")

hist_targets = [
    ("Age", "Age"),
    ("Experience (Years)", "ExperienceYears"),
    ("Work Hours / Week", "HoursPerWeek"),
    ("Emotional Exhaustion", "EE"),
    ("Depersonalisation", "DP"),
    ("Personal Accomplishment (PA)", "PA"),
    ("Adaptability (ADT)", "ADT"),
    ("Conscientiousness (CST)", "CST"),
    ("Perceived Organizational Support (POS)", "POS"),
    ("Autonomy (AUT)", "AUT"),
    ("Workload (WKL)", "WKL"),
    ("Neuroticism (NEU)", "NEU"),
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
