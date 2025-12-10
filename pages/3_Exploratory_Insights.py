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

# Regression grid
st.subheader("Regression Relationships")

relationships = [
    ("ADT", "EE", "EE vs Adaptability (ADT)"),
    ("CST", "EE", "EE vs Conscientiousness (CST)"),
    ("NEU", "EE", "EE vs Neuroticism (NEU)"),
    ("WKL", "EE", "EE vs Workload (WKL)"),
    ("POS", "EE", "EE vs Positive Climate (POS)"),
    ("ADT", "DP", "DP vs Adaptability (ADT)"),
    ("CST", "DP", "DP vs Conscientiousness (CST)"),
    ("NEU", "DP", "DP vs Neuroticism (NEU)"),
    ("POS", "DP", "DP vs Positive Climate (POS)"),
    ("ADT", "PA", "PA vs Adaptability (ADT)"),
    ("EXT", "PA", "PA vs Extraversion (EXT)"),
    ("POS", "PA", "PA vs Positive Climate (POS)"),
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

    fig.suptitle("Scatter Plots of Key Burnout Relationships", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    st.pyplot(fig, use_container_width=True)
else:
    st.info("No numeric variable pairs available for regression plots.")
