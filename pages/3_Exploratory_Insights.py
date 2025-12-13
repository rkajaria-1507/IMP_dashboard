import math

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import get_dataset

st.title("Exploratory Data Insights")

try:
    df = get_dataset()
except FileNotFoundError:
    st.error("Packaged dataset missing. Please place 'Data_Sheet _Cleaned_Final.csv' beside app.py.")
    st.stop()

# --- Baseline Relationships with Burnout Dimensions ---
st.subheader("Baseline Relationships with Burnout Dimensions")

# Predictor selector
predictor_options = {
    "Adaptability": ("ADT", "Higher adaptability is generally associated with lower burnout, though the strength of this relationship varies across burnout dimensions."),
    "Big Five Personality Traits": ("personality", "Personality traits exhibit distinct baseline relationships with burnout dimensions, particularly stronger associations for Neuroticism and Conscientiousness."),
    "Workload": ("WKL", "Higher workload is associated with increased Emotional Exhaustion, with weaker and more variable effects on other burnout dimensions."),
    "Autonomy": ("AUT", "Greater autonomy is associated with lower burnout, particularly in terms of Emotional Exhaustion and Personal Accomplishment."),
    "Perceived Organizational Support": ("POS", "Perceived organisational support shows a consistent protective relationship across burnout dimensions.")
}

selected_predictor = st.selectbox(
    "Select predictor to explore:",
    list(predictor_options.keys()),
    index=0
)

predictor_code, interpretation = predictor_options[selected_predictor]
burnout_dims = ["EE", "DP", "PA"]
burnout_labels = {
    "EE": "Emotional Exhaustion",
    "DP": "Depersonalisation", 
    "PA": "Personal Accomplishment"
}

if predictor_code == "personality":
    # Big Five subset: Neuroticism vs EE, Conscientiousness vs PA, Extraversion vs DP
    personality_pairs = [
        ("NEU", "EE", "Neuroticism vs Emotional Exhaustion"),
        ("CST", "PA", "Conscientiousness vs Personal Accomplishment"),
        ("EXT", "DP", "Extraversion vs Depersonalisation")
    ]
    
    valid_pairs = [
        (pred, outcome, title) for pred, outcome, title in personality_pairs
        if pred in df.columns and outcome in df.columns
    ]
    
    if valid_pairs:
        fig, axes = plt.subplots(1, len(valid_pairs), figsize=(len(valid_pairs) * 5.5, 4.8))
        if len(valid_pairs) == 1:
            axes = [axes]
        
        for ax, (pred, outcome, title) in zip(axes, valid_pairs):
            sns.regplot(
                x=df[pred], y=df[outcome], ax=ax,
                scatter_kws={"alpha": 0.5, "color": "#9b59b6"},
                line_kws={"color": "#e74c3c", "linewidth": 2},
                lowess=True
            )
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(pred)
            ax.set_ylabel(burnout_labels[outcome])
            ax.grid(axis="both", linestyle="--", alpha=0.3)
        
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.info(interpretation)
    else:
        st.warning("Required personality trait data not available.")
        
else:
    # Single predictor vs all three burnout dimensions
    if predictor_code in df.columns:
        available_burnout = [dim for dim in burnout_dims if dim in df.columns]
        
        if available_burnout:
            fig, axes = plt.subplots(1, len(available_burnout), figsize=(len(available_burnout) * 5.5, 4.8))
            if len(available_burnout) == 1:
                axes = [axes]
            
            colors_map = {"EE": "#e74c3c", "DP": "#e67e22", "PA": "#3498db"}
            
            for ax, outcome in zip(axes, available_burnout):
                sns.regplot(
                    x=df[predictor_code], y=df[outcome], ax=ax,
                    scatter_kws={"alpha": 0.5, "color": colors_map[outcome]},
                    line_kws={"color": "#2c3e50", "linewidth": 2},
                    lowess=True
                )
                ax.set_title(f"{burnout_labels[outcome]} vs {selected_predictor}", fontsize=11)
                ax.set_xlabel(selected_predictor)
                ax.set_ylabel(burnout_labels[outcome])
                ax.grid(axis="both", linestyle="--", alpha=0.3)
            
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.info(interpretation)
        else:
            st.warning("Burnout dimension data not available.")
    else:
        st.warning(f"{selected_predictor} data not available in the dataset.")

st.divider()

# Correlation Heatmap
st.subheader("Correlation Heatmap")
priority_cols = [
    "ADT",
    "EXT",
    "AGR",
    "CST",
    "NEU",
    "OPE",
    "EE",
    "DP",
    "PA",
    "AUT",
    "WKL",
    "POS",
    "HoursPerWeek",
    "ExperienceYears",
    "Age",
    "Gender_num",
]
available = [c for c in priority_cols if c in df.columns]
numeric_df = df[available].select_dtypes(include="number") if available else pd.DataFrame()

if numeric_df.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        ax=ax,
        annot_kws={"fontsize": 7},
    )
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
else:
    st.info("Not enough numeric columns to compute correlations.")

# Additional Relationships Overview
st.subheader("Additional Relationship Patterns")

st.markdown("""
This section provides a comprehensive view of bivariate relationships between predictors and burnout dimensions.
Use the baseline relationships above for focused exploration of specific predictors.
""")

relationships = [
    ("ADT", "EE", "Emotional Exhaustion vs Adaptability"),
    ("CST", "EE", "Emotional Exhaustion vs Conscientiousness"),
    ("NEU", "EE", "Emotional Exhaustion vs Neuroticism"),
    ("WKL", "EE", "Emotional Exhaustion vs Workload"),
    ("POS", "EE", "Emotional Exhaustion vs Perceived Organizational Support"),
    ("ADT", "DP", "Depersonalisation vs Adaptability"),
    ("CST", "DP", "Depersonalisation vs Conscientiousness"),
    ("NEU", "DP", "Depersonalisation vs Neuroticism"),
    ("POS", "DP", "Depersonalisation vs Perceived Organizational Support"),
    ("ADT", "PA", "Personal Accomplishment vs Adaptability"),
    ("EXT", "PA", "Personal Accomplishment vs Extraversion"),
    ("POS", "PA", "Personal Accomplishment vs Perceived Organizational Support"),
]

valid_relationships = [
    (x, y, title) for x, y, title in relationships
    if x in df.columns and y in df.columns
    and pd.api.types.is_numeric_dtype(df[x])
    and pd.api.types.is_numeric_dtype(df[y])
]

if valid_relationships:
    cols = 3
    rows = math.ceil(len(valid_relationships) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 5.2))
    axes = axes.flatten()

    scatter_palette = sns.color_palette("viridis", len(valid_relationships))

    for ax, (x, y, title), color in zip(axes, valid_relationships, scatter_palette):
        sns.regplot(
            x=df[x],
            y=df[y],
            ax=ax,
            scatter_kws={"alpha": 0.6, "color": color},
            line_kws={"color": "#ff6b6b"},
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="both", linestyle="--", alpha=0.3)

    for ax in axes[len(valid_relationships):]:
        ax.remove()

    fig.suptitle("Comprehensive Bivariate Relationships", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    st.pyplot(fig, use_container_width=True)
else:
    st.info("No numeric variable pairs available for regression plots.")
