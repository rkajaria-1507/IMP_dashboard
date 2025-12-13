import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data_loader import get_dataset

try:
    import statsmodels.formula.api as smf
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

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
    ax.set_title(f"Adaptability × {moderator} → Emotional Exhaustion")
    ax.set_xlabel("Adaptability (Centered)")
    ax.set_ylabel("Emotional Exhaustion")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    return fig


mods = ["WKL_c", "AUT_c", "POS_c", "HoursPerWeek_c"]

for mod in mods:
    if mod in df.columns:
        st.subheader(f"Moderation: {mod}")
        interaction_fig = plot_interaction(df, mod)
        if interaction_fig is not None:
            st.pyplot(interaction_fig, width='stretch')

# --- Advanced Interaction Analysis ---
st.divider()
st.subheader("Advanced Interaction Analysis")

if not _STATSMODELS_AVAILABLE:
    st.warning("Statsmodels is required for advanced interaction analysis. Please install it to view these visualizations.")
    st.code("pip install statsmodels", language="bash")
else:
    # Define function to generate interaction plot
    def plot_advanced_interaction(dv_name, iv1_name, iv2_name, df_data):
        """
        Generates an interaction plot for adaptability and moderators on burnout dimensions.
        
        Args:
            dv_name (str): Dependent variable (e.g., 'EE', 'DP', 'PA')
            iv1_name (str): First independent variable (e.g., 'ADT_c')
            iv2_name (str): Second independent variable/moderator (e.g., 'HoursPerWeek_c')
            df_data (pd.DataFrame): Original DataFrame containing the data
        """
        # Build formula for regression
        controls = []
        for ctrl in ['Age_c', 'WorkExperienceYears_c', 'Gender_num']:
            if ctrl in df_data.columns and ctrl != iv2_name:
                controls.append(ctrl)
        
        control_str = " + " + " + ".join(controls) if controls else ""
        formula = f"{dv_name} ~ {iv1_name} * {iv2_name}{control_str}"
        
        try:
            # Fit model
            model = smf.ols(formula, data=df_data).fit()
            
            # Calculate levels for the moderator
            iv2_mean = df_data[iv2_name].mean()
            iv2_std = df_data[iv2_name].std()
            
            iv2_levels = {
                'Low': iv2_mean - iv2_std,
                'Mean': iv2_mean,
                'High': iv2_mean + iv2_std
            }
            
            # Create range for iv1
            iv1_range = np.linspace(df_data[iv1_name].min(), df_data[iv1_name].max(), 100)
            
            # Prepare prediction data
            plot_data = []
            for label, iv2_val in iv2_levels.items():
                predict_data = {
                    iv1_name: iv1_range,
                    iv2_name: iv2_val,
                }
                
                # Add control variables at their means
                for ctrl in controls:
                    if ctrl == 'Gender_num':
                        predict_data[ctrl] = df_data['Gender_num'].mode()[0] if 'Gender_num' in df_data.columns else 0.5
                    else:
                        predict_data[ctrl] = df_data[ctrl].mean()
                
                predict_df = pd.DataFrame(predict_data)
                predicted_dv = model.predict(predict_df)
                
                for i, val in enumerate(iv1_range):
                    plot_data.append({
                        iv1_name: val,
                        dv_name: predicted_dv.iloc[i],
                        'Level': label
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(
                data=plot_df,
                x=iv1_name,
                y=dv_name,
                hue='Level',
                palette='viridis',
                linewidth=2,
                ax=ax
            )
            
            # Create labels
            iv1_label = iv1_name.replace('_c', '').replace('ADT', 'Adaptability')
            iv2_label = (iv2_name.replace('_c', '')
                        .replace('HoursPerWeek', 'Hours Per Week')
                        .replace('WKL', 'Workload')
                        .replace('AUT', 'Autonomy')
                        .replace('POS', 'Perceived Organizational Support'))
            dv_label = (dv_name.replace('EE', 'Emotional Exhaustion')
                       .replace('DP', 'Depersonalisation')
                       .replace('PA', 'Personal Accomplishment'))
            
            ax.set_title(f'{iv1_label} × {iv2_label} → {dv_label}', fontsize=12, fontweight='bold')
            ax.set_xlabel(iv1_label)
            ax.set_ylabel(dv_label)
            ax.legend(title=iv2_label)
            ax.grid(True, linestyle='--', alpha=0.4)
            
            return fig
            
        except Exception as e:
            st.error(f"Error generating interaction plot: {str(e)}")
            return None
    
    # Define interactions to plot
    interactions = [
        ("HoursPerWeek_c", "Hours Per Week"),
        ("WKL_c", "Workload"),
        ("AUT_c", "Autonomy"),
        ("POS_c", "Perceived Organizational Support")
    ]
    
    burnout_dims = [
        ("EE", "Emotional Exhaustion"),
        ("DP", "Depersonalisation"),
        ("PA", "Personal Accomplishment")
    ]
    
    # Allow user to select which interaction to view
    selected_moderator = st.selectbox(
        "Select Moderator:",
        [label for _, label in interactions],
        index=0
    )
    
    # Find the corresponding column name
    moderator_col = next(col for col, label in interactions if label == selected_moderator)
    
    if moderator_col in df.columns and "ADT_c" in df.columns:
        # Create plots for all three burnout dimensions
        for dv_col, dv_label in burnout_dims:
            if dv_col in df.columns:
                st.markdown(f"**{dv_label}**")
                fig = plot_advanced_interaction(dv_col, "ADT_c", moderator_col, df)
                if fig is not None:
                    st.pyplot(fig, width='stretch')
    else:
        st.warning("Required variables not available for interaction analysis.")

