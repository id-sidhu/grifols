"""Classification of unit-status rows into the 'to be removed' sets."""

import re
from typing import List, Set

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def process_unit_status_all(us_df: pd.DataFrame, prefix: str) -> Set[str]:
    """
    Build a 'to be removed' set across ALL rows (no date filter) for a given prefix.

    - Drops rows with missing/blank Donor Status
    - Extracts numeric part from Donation # matching prefix and computes missing numbers
      between min and max seen for that prefix (no_bleeds)
    - Normalizes Status and classifies into rejected / sample_only
    - Returns a set containing: no_bleeds + rejected donation numbers + sample_only donation numbers
    """
    required_cols = {"Donor Status", "Donation #", "Status"}
    missing = required_cols - set(us_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in unit status DataFrame: {missing}")

    df = us_df.copy()

    # Compute no-bleed gaps from ALL rows matching the prefix (before any
    # filtering) so that samples with blank Donor Status do not create
    # false gaps.
    _all_donation_nums = (
        df["Donation #"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.extract(rf"^{re.escape(prefix)}(\d+)$")[0]
        .astype(float)
    )
    _all_nums = _all_donation_nums.dropna().astype(int).sort_values()
    missing_nums: List[int] = []
    if not _all_nums.empty:
        missing_nums = sorted(set(range(_all_nums.min(), _all_nums.max() + 1)) - set(_all_nums))
    no_bleeds = {f"{prefix}{n:06d}" for n in missing_nums}

    # Remove rows with missing/blank Donor Status for classification
    donor_status = df["Donor Status"].astype(str)
    df = df[donor_status.notna() & (donor_status.str.strip() != "")].copy()

    # Extract numeric donation number for the given prefix
    df["donation_num"] = (
        df["Donation #"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.extract(rf"^{re.escape(prefix)}(\d+)$")[0]
        .astype(float)
    )

    # Normalize status
    status_raw = df["Status"].fillna("").astype(str).str.strip()

    def normalize_status(s: str) -> str:
        s_lower = s.strip().lower()
        if s_lower == "quarantine":
            return "Quarantine"
        if s_lower.startswith("so"):
            return "SO (16 week)"
        return "Rejected"

    df["Status_normalized"] = status_raw.map(normalize_status)

    def classify(s_norm: str) -> str:
        if s_norm == "Quarantine":
            return "quarantine"
        if s_norm == "SO (16 week)":
            return "sample_only"
        return "rejected"

    df["Type"] = df["Status_normalized"].map(classify)

    # Build rejected set, but exclude units whose 'Reasons/ Notes' indicates
    # samples were actually collected (i.e. note contains "samples collected"
    # but NOT "no samples collected").
    rejected_mask = df["Type"] == "rejected"
    samples_collected_rejected: Set[str] = set()
    reasons_col = "Reasons/ Notes"
    if reasons_col in df.columns:
        notes_lower = df[reasons_col].fillna("").astype(str).str.lower()
        has_samples_collected = notes_lower.str.contains("samples collected", na=False)
        has_no_samples_collected = notes_lower.str.contains("no samples collected", na=False)
        # Keep (do not remove) rejected units where samples were collected
        samples_were_collected = has_samples_collected & ~has_no_samples_collected
        rejected_mask = rejected_mask & ~samples_were_collected
        # Track these kept IDs so callers can highlight them separately
        samples_collected_rejected = set(
            df.loc[(df["Type"] == "rejected") & samples_were_collected, "Donation #"]
            .dropna().astype(str).str.strip().tolist()
        )
    rejected = set(df.loc[rejected_mask, "Donation #"].dropna().astype(str).str.strip().tolist())
    sample_only = set(df.loc[df["Type"] == "sample_only", "Donation #"].dropna().astype(str).str.strip().tolist())

    return set().union(no_bleeds, rejected, sample_only), samples_collected_rejected
