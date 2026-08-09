"""Pure pallet-slicing logic: markers, cleaning, and the report text."""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


def remove_packed(gs_df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows from the shipment DataFrame where samples have already been packed.

    The ``gs_df`` DataFrame is expected to have a column named ``"Samples Packed?"``
    containing values such as ``"y"`` or ``"yes"`` (case insensitive) to indicate
    that a sample has been packed.  Rows with those indicators are removed.

    Parameters
    ----------
    gs_df: pd.DataFrame
        The Grifols shipment DataFrame.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with packed samples removed.
    """
    if "Samples Packed?" not in gs_df.columns:
        return gs_df
    packed_mask = (
        gs_df["Samples Packed?"].fillna("").astype(str).str.strip().ne("")
    )
    return gs_df.loc[~packed_mask].copy()


def clean_grifols_shipment(
    gs_df: pd.DataFrame, pallet: int
) -> Tuple[pd.DataFrame, Optional[str], Optional[str], Optional[int], Optional[int]]:
    """Remove rows without sample IDs and find pallet start/end markers.

    This function performs two tasks:
    1. Identify the start and end rows for a given pallet within the
       ``Comments`` column using markers of the form ``START OF PALLET X``
       and ``END OF PALLET X`` (case insensitive).  It returns the indices
       of those rows and the corresponding Sample IDs (if present).
    2. Remove any rows in the returned DataFrame where ``"Sample ID"`` is ``NaN``.

    The original script returned a modified DataFrame along with the start
    and end Sample IDs and row indices.  We retain that behaviour here.

    Parameters
    ----------
    gs_df: pd.DataFrame
        The Grifols shipment DataFrame containing a ``"Comments"`` column and
        a ``"Sample ID"`` column.
    pallet: int
        The pallet number to search for within the comments.

    Returns
    -------
    tuple
        A tuple containing:
        ``(cleaned_df, sop_id, eop_id, sop_row, eop_row)`` where ``cleaned_df``
        has ``NaN`` values removed from ``"Sample ID"``; ``sop_id`` and
        ``eop_id`` are the Sample IDs found on the start/end marker rows
        (or ``None`` if not present); and ``sop_row``/``eop_row`` are the row
        indices of those markers (or ``None`` if not found).
    """
    gs_df = gs_df.copy()
    comments = gs_df["Comments"].fillna("").astype(str)

    # Compile regex patterns for start and end markers for the given pallet.
    # The patterns ignore leading/trailing whitespace and are case insensitive.
    sop_pat = re.compile(rf"^\s*START\s+OF\s+PALLET\s+{int(pallet)}\s*$", re.IGNORECASE)
    eop_pat = re.compile(rf"^\s*END\s+OF\s+PALLET\s+{int(pallet)}\s*$", re.IGNORECASE)

    # Find indices of rows matching the patterns.
    sop_rows = gs_df.index[comments.str.match(sop_pat)]
    eop_rows = gs_df.index[comments.str.match(eop_pat)]

    # Select the first occurrence of each marker if present.
    sop_row = int(sop_rows[0]) if len(sop_rows) else None
    eop_row = int(eop_rows[0]) if len(eop_rows) else None

    # Extract the Sample IDs on those marker rows.  They may be NaN.
    sop_id = gs_df.loc[sop_row, "Sample ID"] if sop_row is not None else None
    eop_id = gs_df.loc[eop_row, "Sample ID"] if eop_row is not None else None

    # Normalize missing values to None rather than np.nan
    sop_id = None if pd.isna(sop_id) else str(sop_id)
    eop_id = None if pd.isna(eop_id) else str(eop_id)

    # Drop rows where Sample ID is NaN for the cleaned DataFrame
    cleaned_df = gs_df.dropna(subset=["Sample ID"]).copy()

    return cleaned_df, sop_id, eop_id, sop_row, eop_row


def get_pallet_between_markers(
    df_original: pd.DataFrame, sop_row: Optional[int], eop_row: Optional[int]
) -> pd.DataFrame:
    """Return rows of ``df_original`` between the start and end marker indices.

    If either index is ``None``, a ``ValueError`` is raised.  If the start row
    occurs after the end row, the indices are swapped to ensure a correct
    slice, mirroring the behaviour of the original script.

    Parameters
    ----------
    df_original: pd.DataFrame
        The DataFrame to slice.  This should be the original shipment DataFrame
        with all rows intact (including those with missing ``Sample ID``).
    sop_row: int or None
        The index of the start marker row.
    eop_row: int or None
        The index of the end marker row.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing rows from ``sop_row`` through ``eop_row`` inclusive.
    """
    if sop_row is None or eop_row is None:
        raise ValueError("START/END OF PALLET marker row not found in original df.")
    # Ensure sop_row <= eop_row
    if sop_row > eop_row:
        sop_row, eop_row = eop_row, sop_row
    return df_original.loc[sop_row : eop_row].reset_index(drop=True)


def get_pallet_between_ids(
    df: pd.DataFrame, sop_id: Optional[str], eop_id: Optional[str]
) -> pd.DataFrame:
    """Return rows of ``df`` between rows where ``Sample ID`` equals the given IDs.

    This helper is not used in the core logic of the app but is provided for
    completeness.  It replicates the behaviour of the original script.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to slice.
    sop_id: str or None
        The Sample ID where the pallet starts.
    eop_id: str or None
        The Sample ID where the pallet ends.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing rows between the IDs.
    """
    if sop_id is None or eop_id is None:
        raise ValueError("sop_id or eop_id not provided.")
    df = df.copy()
    sop_idx = df.index[df["Sample ID"] == sop_id]
    eop_idx = df.index[df["Sample ID"] == eop_id]
    if sop_idx.empty or eop_idx.empty:
        raise ValueError("sop_id or eop_id not found in df['Sample ID'].")
    i_start = int(sop_idx[0])
    i_end = int(eop_idx[0])
    if i_start > i_end:
        i_start, i_end = i_end, i_start
    return df.loc[i_start : i_end].reset_index(drop=True)


def split_ids_by_prefix(pallet_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Separate the ``Sample ID`` column into F25 and F26 prefixes.

    Parameters
    ----------
    pallet_df: pd.DataFrame
        The DataFrame representing a pallet, with a column ``"Sample ID"``.

    Returns
    -------
    tuple
        Two lists: one of all sample IDs starting with ``"F25-"`` and one of all
        sample IDs starting with ``"F26-"``.  The IDs are sorted alphabetically.
    """
    ids = pallet_df["Sample ID"].dropna().astype(str).str.strip()
    f25_ids = ids[ids.str.startswith("F25-")].sort_values().tolist()
    f26_ids = ids[ids.str.startswith("F26-")].sort_values().tolist()
    return f25_ids, f26_ids


@st.cache_data(show_spinner=False)
def build_pallet_map(gs_df: pd.DataFrame) -> Dict[str, int]:
    """Return a dict mapping each Sample ID to its pallet number.

    Scans the ``Comments`` column for ``START OF PALLET N`` / ``END OF PALLET N``
    markers and assigns every Sample ID found between those markers to pallet N.

    Parameters
    ----------
    gs_df: pd.DataFrame
        The Grifols shipment DataFrame.

    Returns
    -------
    Dict[str, int]
        A mapping of sample ID string → pallet number integer.
    """
    if "Comments" not in gs_df.columns or "Sample ID" not in gs_df.columns:
        return {}
    comments = gs_df["Comments"].fillna("").astype(str)
    # Discover all pallet numbers present in the file
    pallet_nums: set = set()
    for comment in comments:
        m = re.match(r"^\s*START\s+OF\s+PALLET\s+(\d+)\s*$", comment, re.IGNORECASE)
        if m:
            pallet_nums.add(int(m.group(1)))
    pallet_map: Dict[str, int] = {}
    for pnum in sorted(pallet_nums):
        sop_pat = re.compile(rf"^\s*START\s+OF\s+PALLET\s+{pnum}\s*$", re.IGNORECASE)
        eop_pat = re.compile(rf"^\s*END\s+OF\s+PALLET\s+{pnum}\s*$", re.IGNORECASE)
        sop_rows = gs_df.index[comments.str.match(sop_pat)]
        eop_rows = gs_df.index[comments.str.match(eop_pat)]
        if sop_rows.empty or eop_rows.empty:
            continue
        sop_row, eop_row = int(sop_rows[0]), int(eop_rows[0])
        if sop_row > eop_row:
            sop_row, eop_row = eop_row, sop_row
        for sid in gs_df.loc[sop_row:eop_row, "Sample ID"].dropna().astype(str).str.strip():
            if sid:
                pallet_map[sid] = pnum
    return pallet_map


@st.cache_data(show_spinner=False)
def generate_report_text(
    pallet_df: pd.DataFrame,
    pallet_size: int,
    pallet_no: int,
    sop_id: Optional[str] = None,
    eop_id: Optional[str] = None,
) -> Tuple[str, List[str], List[str]]:
    """Generate a textual report summarising a pallet and return F25/F26 IDs.

    This helper assembles the same report text that the original script
    printed to stdout.  It returns the report as a string along with the
    lists of F25 and F26 IDs for optional display.

    Parameters
    ----------
    pallet_df: pd.DataFrame
        The DataFrame representing a pallet.
    pallet_size: int
        The number of rows in the pallet in the original DataFrame before
        removing packed rows or NaNs.  This mirrors the original behaviour
        where ``length_of_pallet`` was computed before filtering.
    pallet_no: int
        The pallet number being reported.
    sop_id: str or None
        The Sample ID where the pallet starts.
    eop_id: str or None
        The Sample ID where the pallet ends.

    Returns
    -------
    tuple
        A tuple ``(report_text, f25_ids, f26_ids)`` where ``report_text`` is
        the formatted report string and ``f25_ids``/``f26_ids`` are the lists
        of sample IDs beginning with the respective prefixes.
    """
    f25_ids, f26_ids = split_ids_by_prefix(pallet_df)
    total_ids = len(f25_ids) + len(f26_ids)
    ids_all = pallet_df["Sample ID"].dropna().astype(str).str.strip()
    first_id = ids_all.min() if not ids_all.empty else None
    last_id = ids_all.max() if not ids_all.empty else None
    title = f"PALLET {pallet_no} PACKING REPORT"
    line = "=" * len(title)
    # Build the report as a single formatted string
    lines = [
        line,
        title,
        line,
        f"Sample ID Where Pallet Starts: {sop_id}",
        f"Sample ID Where Pallet Ends: {eop_id}",
        "-" * len(line),
        f"Total number of samples in pallet: {pallet_size}",
        "-" * len(line),
        f"First sample ID to be packed: {first_id}",
        f"Last sample ID to be packed: {last_id}",
        "-" * len(line),
        f"F25 count: {len(f25_ids)}",
        f"F26 count: {len(f26_ids)}",
        "-" * len(line),
        f"Total samples to pack: {total_ids}",
        line,
    ]
    report_text = "\n".join(lines)
    return report_text, f25_ids, f26_ids
