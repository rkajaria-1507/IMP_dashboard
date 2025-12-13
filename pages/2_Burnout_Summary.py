import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data_loader import get_dataset

st.title("Burnout Summary")

try:
    df = get_dataset()
except FileNotFoundError:
    st.error("Packaged dataset missing. Please place 'Data_Sheet _Cleaned_Final.csv' beside app.py.")
    st.stop()

# --- Comparative Burnout Dimensions ---
st.subheader("Comparative Burnout Dimensions (Emotional Exhaustion, Depersonalisation, Personal Accomplishment)")

cols = ["EE", "DP", "PA"]
available_cols = [c for c in cols if c in df.columns]

if len(available_cols) >= 2:
    burnout_stats = []
    for col in available_cols:
        data = df[col].dropna()
        burnout_stats.append({
            "Dimension": col,
            "Mean": data.mean(),
            "Std": data.std()
        })
    
    stats_df = pd.DataFrame(burnout_stats)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(stats_df))
    
    bars = ax.bar(x_pos, stats_df["Mean"], yerr=stats_df["Std"], 
                   capsize=8, alpha=0.8, color=["#e74c3c", "#e67e22", "#3498db"])
    
    ax.set_xlabel("Burnout Dimension", fontsize=12)
    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_title("Comparative Burnout Dimensions with Variability", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_df["Dimension"])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    
    fig.tight_layout()
    st.pyplot(fig, width='stretch')

st.divider()

# --- Individual Distributions ---
st.subheader("Individual Burnout Distributions")

cols = ["EE", "DP", "PA"]
available_dist_cols = [c for c in cols if c in df.columns]

if available_dist_cols:
    fig, axes = plt.subplots(1, len(available_dist_cols), figsize=(len(available_dist_cols) * 5.5, 4.5))
    if len(available_dist_cols) == 1:
        axes = [axes]
    
    colors = ["#e74c3c", "#e67e22", "#3498db"]
    
    for idx, (ax, col) in enumerate(zip(axes, available_dist_cols)):
        sns.histplot(df[col].dropna(), kde=True, bins=20, ax=ax, color=colors[idx])
        ax.set_title(f"{col} Distribution", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    
    fig.tight_layout()
    st.pyplot(fig, width='stretch')

# Simple burnout risk (can be replaced with Maslach cutoffs)
st.divider()
st.subheader("Burnout Risk Indicators")

risk_cols = st.columns(3)

if "EE" in df.columns:
    high_ee = (df["EE"] > df["EE"].mean() + df["EE"].std()).mean() * 100
    risk_cols[0].metric("High Emotional Exhaustion (%)", f"{high_ee:.1f}%")

if "DP" in df.columns:
    high_dp = (df["DP"] > df["DP"].mean() + df["DP"].std()).mean() * 100
    risk_cols[1].metric("High Depersonalisation (%)", f"{high_dp:.1f}%")

if "PA" in df.columns:
    low_pa = (df["PA"] < df["PA"].mean() - df["PA"].std()).mean() * 100
    risk_cols[2].metric("Low Personal Accomplishment (%)", f"{low_pa:.1f}%")

# --- Burnout by Organisational Context ---
st.divider()
st.subheader("Burnout by Organisational Context (Key Moderators)")

moderators = [
    ("WKL", "Workload"),
    ("AUT", "Autonomy"),
    ("POS", "Perceived Organizational Support")
]

burnout_dimensions = [
    ("EE", "Emotional Exhaustion", "#e74c3c"),
    ("DP", "Depersonalisation", "#e67e22"),
    ("PA", "Personal Accomplishment", "#3498db")
]

available_moderators = [(col, label) for col, label in moderators if col in df.columns]
available_burnout = [(col, label, color) for col, label, color in burnout_dimensions if col in df.columns]

if available_moderators and available_burnout:
    for burnout_col, burnout_label, burnout_color in available_burnout:
        st.markdown(f"**{burnout_label}**")
        
        context_data = []
        mod_categories = []
        
        for mod_col, mod_label in available_moderators:
            mod_series = df[mod_col].dropna()
            if mod_series.nunique() < 2:
                continue
                
            median_val = mod_series.median()
            
            low_mask = df[mod_col] <= median_val
            high_mask = df[mod_col] > median_val
            
            low_burnout = df.loc[low_mask, burnout_col].mean()
            high_burnout = df.loc[high_mask, burnout_col].mean()
            
            context_data.append(low_burnout)
            context_data.append(high_burnout)
            mod_categories.extend([f"Low", f"High"])
        
        if context_data:
            # Create grouped structure
            n_mods = len(available_moderators)
            x_positions = np.arange(n_mods)
            width = 0.35
            
            low_values = [context_data[i*2] for i in range(n_mods)]
            high_values = [context_data[i*2 + 1] for i in range(n_mods)]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.bar(x_positions - width/2, low_values, width, label="Low", 
                   color="#3498db", alpha=0.8, edgecolor="black", linewidth=0.8)
            ax.bar(x_positions + width/2, high_values, width, label="High", 
                   color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.8)
            
            ax.set_xlabel("Organisational Factor", fontsize=12)
            ax.set_ylabel(f"Mean {burnout_label}", fontsize=12)
            ax.set_title(f"{burnout_label} Across Organisational Contexts", fontsize=14)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([label for _, label in available_moderators])
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            
            fig.tight_layout()
            st.pyplot(fig, width='stretch')
    
    st.info("Burnout prevalence is markedly higher under conditions of high workload and low organisational support, highlighting the role of contextual stressors.")
else:
    pass
    # st.warning("Required variables not available for organisational context analysis.")
