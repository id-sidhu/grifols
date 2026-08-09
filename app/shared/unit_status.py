"""Cleaning rules for the Unit Status file, shared across sections."""

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def clean_unit_status(us_df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows from the unit status DataFrame based on status patterns.

    The original script removed rows where the ``Status`` column contained
    ``"ejec"`` or exactly the letter ``"S"`` (case insensitive).  This helper
    function performs the same filtering and returns a new DataFrame.

    Parameters
    ----------
    us_df: pd.DataFrame
        The unit status DataFrame with a column named ``"Status"``.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the undesired rows removed.
    """
    if "Status" not in us_df.columns:
        return us_df
    # Drop rows where the Status contains "ejec" (case insensitive).
    # Note: "Rejected" also contains "ejec", so ALL rejected rows match this.
    mask_ejec = us_df["Status"].astype(str).str.contains("ejec", case=False, na=False)
    # Exception: if the 'Reasons/ Notes' column says samples were actually
    # collected (contains "samples collected" but NOT "no samples collected"),
    # keep the row so it can be classified separately downstream.
    reasons_col = "Reasons/ Notes"
    if reasons_col in us_df.columns:
        notes_lower = us_df[reasons_col].fillna("").astype(str).str.lower()
        has_sc = notes_lower.str.contains("samples collected", na=False)
        has_no_sc = notes_lower.str.contains("no samples collected", na=False)
        samples_were_collected = has_sc & ~has_no_sc
        # Narrow the ejec mask: only drop ejec rows where samples were NOT collected
        mask_ejec = mask_ejec & ~samples_were_collected
    # Drop rows where the Status is exactly "S" (case insensitive).  We use
    # fullmatch to ensure we don't accidentally match longer strings.
    mask_s = us_df["Status"].astype(str).str.fullmatch(r"s", case=False, na=False)
    return us_df.loc[~(mask_ejec | mask_s)].copy()
