"""Grouping donations into visual-inspection label batches."""

import re
from typing import Dict, List

import pandas as pd
import streamlit as st

from app.shared.dates import _parse_donation_date


@st.cache_data(show_spinner=False)
def build_vi_label_groups(
    us_df: pd.DataFrame,
    prefix: str,
    start_id: str,
    group_size: int = 12,
) -> List[Dict]:
    """Group donation IDs into batches for Visual Inspection labels.

    Each complete group contains exactly ``group_size`` Quarantine units.
    Rejected and SO units encountered within the sequential range are
    included in the group (they appear in the date/ID span) but do **not**
    count toward ``group_size``.  No-bleed gaps are absent from the data and
    therefore never appear in any group.

    Parameters
    ----------
    us_df : pd.DataFrame
        Raw (uncleaned) unit status DataFrame.
    prefix : str
        Donation ID prefix, e.g. ``"F26-"``.
    start_id : str
        The first Donation # to include.  Must exist in the data.
    group_size : int
        Number of Quarantine units per complete label group (default 12).

    Returns
    -------
    list of dict
        Each dict has keys: ``rows``, ``first_id``, ``last_id``,
        ``date_min``, ``date_max``, ``is_complete``, ``valid_count``.
    """
    required = {"Donation #", "Status"}
    missing = required - set(us_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = us_df.copy()
    donation_col = df["Donation #"].fillna("").astype(str).str.strip()

    def _extract_num(x: str) -> float:
        m = re.match(rf"^{re.escape(prefix)}(\d+)$", x, re.IGNORECASE)
        return float(m.group(1)) if m else float("inf")

    prefix_mask = donation_col.str.upper().str.startswith(prefix.upper())
    df = df.loc[prefix_mask].copy()
    df["_dn"] = donation_col.loc[prefix_mask].map(_extract_num)
    df = df.sort_values("_dn").reset_index(drop=True)

    if df.empty:
        return []

    # Find start position
    start_norm = start_id.strip()
    positions = df.index[df["Donation #"].astype(str).str.strip() == start_norm].tolist()
    if not positions:
        raise ValueError(f"Start ID '{start_id}' not found for prefix '{prefix}'")
    start_pos = positions[0]

    date_col = "Donation Date"
    has_dates = date_col in df.columns

    groups: List[Dict] = []
    cur_rows: List[Dict] = []
    valid_count = 0

    for idx in range(start_pos, len(df)):
        row = df.iloc[idx].to_dict()
        status = str(row.get("Status", "")).strip().lower()
        is_valid = status == "quarantine"
        cur_rows.append(row)
        if is_valid:
            valid_count += 1
        if valid_count == group_size:
            dates = [
                _parse_donation_date(r.get(date_col))
                for r in cur_rows
                if has_dates
            ]
            dates = [d for d in dates if d is not None]
            groups.append({
                "rows": cur_rows,
                "first_id": str(cur_rows[0]["Donation #"]).strip(),
                "last_id": str(cur_rows[-1]["Donation #"]).strip(),
                "date_min": min(dates) if dates else None,
                "date_max": max(dates) if dates else None,
                "is_complete": True,
                "valid_count": valid_count,
            })
            cur_rows = []
            valid_count = 0

    # Trailing partial group (fewer than group_size valid units)
    if cur_rows:
        dates = [
            _parse_donation_date(r.get(date_col))
            for r in cur_rows
            if has_dates
        ]
        dates = [d for d in dates if d is not None]
        groups.append({
            "rows": cur_rows,
            "first_id": str(cur_rows[0]["Donation #"]).strip(),
            "last_id": str(cur_rows[-1]["Donation #"]).strip(),
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "is_complete": False,
            "valid_count": valid_count,
        })

    return groups
