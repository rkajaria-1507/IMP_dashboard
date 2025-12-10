import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import get_dataset

st.title("Burnout Summary")

try:
    df = get_dataset()
except FileNotFoundError:
    st.error("Packaged dataset missing. Please place 'Data_Sheet _Cleaned_Final.csv' beside app.py.")
    st.stop()

cols = ["EE", "DP", "PA"]

for c in cols:
    if c in df.columns:
        st.subheader(f"{c} Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[c], kde=True, ax=ax, color="#ff7675")
        st.pyplot(fig, use_container_width=True)

# Simple burnout risk (can be replaced with Maslach cutoffs)
if "EE" in df.columns:
    high_ee = (df["EE"] > df["EE"].mean() + df["EE"].std()).mean() * 100
    st.metric("High EE (%)", f"{high_ee:.1f}%")

if "DP" in df.columns:
    high_dp = (df["DP"] > df["DP"].mean() + df["DP"].std()).mean() * 100
    st.metric("High DP (%)", f"{high_dp:.1f}%")

if "PA" in df.columns:
    low_pa = (df["PA"] < df["PA"].mean() - df["PA"].std()).mean() * 100
    st.metric("Low PA (%)", f"{low_pa:.1f}%")
