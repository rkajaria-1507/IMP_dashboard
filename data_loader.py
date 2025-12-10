from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "dataset_dashboard.xlsx"


def _clean_columns(columns: Iterable[str]) -> list[str]:
    cleaned = [re.sub(r"[^0-9A-Za-z]+", "_", col or "").strip("_") for col in columns]
    return cleaned


def _encode_gender(df: pd.DataFrame) -> None:
    gender_cols = [c for c in df.columns if "gender" in c.lower()]
    if not gender_cols:
        return
    col = gender_cols[0]
    df[col] = df[col].astype(str).str.strip().str.lower()
    df["Gender_num"] = pd.to_numeric(
        df[col].replace({"male": 1, "m": 1, "female": 0, "f": 0}),
        errors="coerce",
    )


def _encode_age(df: pd.DataFrame) -> None:
    age_cols = [c for c in df.columns if c.lower().startswith("age")]
    if not age_cols:
        return
    col = age_cols[0]
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df.rename(columns={col: "Age"}, inplace=True)


def _encode_hours(df: pd.DataFrame) -> None:
    hours_cols = [c for c in df.columns if "hours_per_week" in c.lower()]
    if not hours_cols:
        return
    col = hours_cols[0]
    df.rename(columns={col: "HoursPerWeek"}, inplace=True)
    df["HoursPerWeek"] = pd.to_numeric(df["HoursPerWeek"], errors="coerce")


def _encode_experience(df: pd.DataFrame) -> None:
    exp_cols = [c for c in df.columns if "experience" in c.lower() and "year" in c.lower()]
    if not exp_cols:
        return
    col = exp_cols[0]
    df.rename(columns={col: "ExperienceYears"}, inplace=True)
    df["ExperienceYears"] = pd.to_numeric(df["ExperienceYears"], errors="coerce")
    df["WorkExperienceYears"] = df["ExperienceYears"]


def _compute_scale_means(df: pd.DataFrame, prefixes: Iterable[str]) -> Dict[str, pd.Series]:
    means: Dict[str, pd.Series] = {}
    for prefix in prefixes:
        candidates = [c for c in df.columns if c.upper().startswith(prefix.upper()) and c != prefix]
        if not candidates:
            continue
        numeric = df.loc[:, candidates].apply(pd.to_numeric, errors="coerce")
        df.loc[:, candidates] = numeric
        means[prefix] = numeric.mean(axis=1)
    return means


@st.cache_data(show_spinner=False)
def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_excel(path)

    df.columns = _clean_columns(df.columns)
    df = df.loc[:, [c for c in df.columns if c]]
    df = df.dropna(axis=1, how="all")
    df = df.dropna(how="all").reset_index(drop=True)

    _encode_gender(df)
    _encode_age(df)
    _encode_hours(df)
    _encode_experience(df)

    scale_means = _compute_scale_means(
        df,
        [
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
        ],
    )
    if scale_means:
        df = df.assign(**scale_means)

    centered: Dict[str, pd.Series] = {}
    for col in [
        *scale_means.keys(),
        "HoursPerWeek",
        "ExperienceYears",
        "WorkExperienceYears",
        "Age",
    ]:
        if col in df.columns:
            centered[f"{col}_c"] = df[col] - df[col].mean()
    if centered:
        df = df.assign(**centered)

    return df


def get_dataset() -> pd.DataFrame:
    if "df" not in st.session_state:
        st.session_state["df"] = load_dataset()
    return st.session_state["df"]
