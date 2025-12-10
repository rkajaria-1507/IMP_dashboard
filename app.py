import streamlit as st

from data_loader import DATA_PATH, get_dataset

st.set_page_config(page_title="IMP Dashboard", layout="wide")

st.title("IMP Dashboard")

try:
    df = get_dataset()
except FileNotFoundError:
    st.error("The packaged dataset could not be found. Please add 'Data_Sheet _Cleaned_Final.csv' to the project.")
    st.stop()

if st.session_state.get("_dataset_loaded") is None:
    st.session_state["_dataset_loaded"] = True
    st.success("Dataset loaded from packaged resource.")

st.dataframe(df.head())

if not st.session_state.get("_navigated_overview") and hasattr(st, "switch_page"):
    st.session_state["_navigated_overview"] = True
    st.switch_page("pages/1_Overview.py")
else:
    st.info("Use the sidebar to open the Overview page.")



