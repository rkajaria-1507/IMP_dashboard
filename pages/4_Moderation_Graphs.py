import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import get_dataset

st.title("Moderation Graphs")

try:
    df = get_dataset()
except FileNotFoundError:
    st.error("Packaged dataset missing. Please place 'Data_Sheet _Cleaned_Final.csv' beside app.py.")
    st.stop()


def plot_interaction(data: pd.DataFrame, moderator: str) -> plt.Figure | None:
    series = data[moderator].dropna()
    if series.nunique() < 2:
        st.warning(f"Not enough variation in {moderator} to create bins.")
        return None

    bins = pd.qcut(series, 3, duplicates="drop")
    if bins.nunique() < 2:
        st.warning(f"Quantile binning collapsed for {moderator}; showing scatter instead.")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=data, x="ADT_c", y="EE", ax=ax, color="#6c5ce7")
        return fig

    unique_bins = bins.nunique()
    labels = ["Low", "Medium", "High"][:unique_bins]
    mod_levels = pd.Series(pd.Categorical(bins, ordered=True).rename_categories(labels), index=series.index)

    plot_df = data.loc[series.index].copy()
    plot_df["ModLevel"] = mod_levels

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=plot_df, x="ADT_c", y="EE", hue="ModLevel", palette="Set2", ax=ax)
    ax.set_title(f"Adaptability × {moderator} → EE")
    ax.set_xlabel("Adaptability (centered)")
    ax.set_ylabel("Emotional Exhaustion")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    return fig


mods = ["WKL_c", "AUT_c", "POS_c", "HoursPerWeek_c"]

for mod in mods:
    if mod in df.columns:
        st.subheader(f"Moderation: {mod}")
        interaction_fig = plot_interaction(df, mod)
        if interaction_fig is not None:
            st.pyplot(interaction_fig, use_container_width=True)


