"""
A Streamlit application that combines the original Grifols pallet packing
report with an additional report for analysing unit status data by date and
donation number prefix.  This combined app allows users to upload their
Grifols shipment CSV alongside a unit status CSV, generate pallet reports
for a specific pallet number, and compute missing donation numbers and
classifications (rejected, sample only) for donations on a particular date.

The original functionality is preserved: users can upload a shipment file,
select a pallet number, and optionally view lists of F25 and F26 sample IDs
within that pallet.  The new functionality adds inputs for selecting a
donation date and prefix to analyse the unit status file.  For the
specified date and prefix the app determines which donation numbers are
missing (no bleeds), and separates the donations into rejected or
"sample only" categories based on their status.

In addition to these features, this version introduces two new pieces of
functionality:

1.  When both a shipment file and a unit status file are provided, the app
    computes a set of donation IDs that appear in the unit status file
    but do **not** appear in the shipment manifest.  These are stored
    internally as ``not_in_manifest`` and may be queried later.  Any
    donation IDs classified as "to be removed" (e.g. rejected or
    sample‑only) are excluded when presenting results to the user.

2.  In the Manual racks/Unit Status Check section, the user may provide **two** control
    numbers separated by a comma.  When this occurs, instead of checking
    a single ID against the "to be removed" set, the app will display
    all IDs from the ``not_in_manifest`` set whose numeric portion falls
    between the two provided control numbers (inclusive) after removing
    any IDs flagged for removal.  This enables quick discovery of ranges
    of missing donation IDs.

To run this app locally, execute the following command from a terminal in
the directory containing this file::

    streamlit run grifols_combined_streamlit.py

The application expects you to upload a ``grifols_shipment.csv`` file (and
optionally a ``unit_status.csv`` file if you want to use the cleaning and
unit status helpers).  You will then be prompted for the pallet number you
are interested in, whether to see a verbose listing of F25 and F26 sample
IDs, and (if a unit status file is supplied) the donation prefix and
control number(s) for the unit status analysis.
"""

import base64
import datetime
import hashlib
import io
import json
import math
import re
import secrets as _secrets_mod
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import streamlit as st

import unit_status_db as usdb

# ---------------------------------------------------------------------------
# Feature flags
# Set a section to False to hide it from the navigation bar.
# Hidden sections remain accessible by adding ?unlock=<your_key> to the URL.
# ---------------------------------------------------------------------------
FEATURES: Dict[str, bool] = {
    "Pallet Report": True,
    "Manual racks/Unit Status Check": True,
    "Visual Inspection Labels": True,
    "QC Report PDF Extractor": True,
    "Master Sheet ": False,
    "Storage Manager": True,
    "Donation Processing": True,
}

# Default verbosity. Users can toggle this live from the sidebar.
VERBOSE_DEFAULT: bool = True

# UNLOCK_KEY is loaded at runtime from Streamlit secrets (see .streamlit/secrets.toml
# locally, or the Streamlit Cloud "Secrets" panel in production).
# It is intentionally NOT hardcoded here so it is never visible in the public repo.


def _verbose() -> bool:
    """Return current verbosity setting (True = show feedback messages)."""
    return st.session_state.get("_verbose", VERBOSE_DEFAULT)


def _show_success(msg: str, **kwargs) -> None:
    if _verbose():
        st.success(msg, **kwargs)


def _show_error(msg: str, **kwargs) -> None:
    if _verbose():
        st.error(msg, **kwargs)


def _show_warning(msg: str, **kwargs) -> None:
    if _verbose():
        st.warning(msg, **kwargs)


def _show_info(msg: str, **kwargs) -> None:
    if _verbose():
        st.info(msg, **kwargs)


def _show_caption(msg: str, **kwargs) -> None:
    if _verbose():
        st.caption(msg, **kwargs)


def _subheader(title: str) -> None:
    st.markdown(
        f'<div class="gf-section-header"><span>{title}</span></div>',
        unsafe_allow_html=True,
    )


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


# -----------------------------------------------------------------------------
# Donation date parsing helper
def _parse_donation_date(raw) -> Optional[datetime.date]:
    """Parse a raw Donation Date value into a datetime.date for comparison.

    Handles:
    - pandas Timestamp / datetime objects
    - Strings in DD.MM.YYYY, YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY formats
    - Excel serial numbers (numeric, > 1)

    Returns None when the value cannot be parsed.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (pd.Timestamp, datetime.datetime)):
        try:
            return pd.Timestamp(raw).date()
        except Exception:
            pass
    if isinstance(raw, datetime.date):
        return raw
    s = str(raw).strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
                "%d.%m.%y", "%Y%m%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    # Excel serial number (float/int stored as text)
    try:
        n = float(s)
        if n > 1:
            return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(n))).date()
    except (ValueError, OverflowError, OSError):
        pass
    return None


# -----------------------------------------------------------------------------
# Rack visualisation helper
@st.cache_data(show_spinner=False)
def build_rack_html(
    valid_ids: List[str],
    not_manifest_set: Set[str],
    samples_collected_set: Optional[Set[str]] = None,
    pallet_map: Optional[Dict[str, int]] = None,
    packed_set: Optional[Set[str]] = None,
    digits_to_show: int = 3,
    fill_value: str = "",
    title: str = "Rack Visualization (last three digits)",
    date_range_str: Optional[str] = None,
) -> str:
    """
    18×12 rack using a true 18-column CSS Grid (no spacer columns, no inserted blank IDs).
    Visible separators after columns 6 and 12 are drawn INSIDE the boundary cells
    (inset shadows), so they remain clearly visible regardless of background/theme.
    Cells whose sample IDs appear in ``packed_set`` receive a diagonal hatching
    overlay so they are visually marked as already packed while retaining their
    underlying colour.
    """
    total_positions = 216
    ids_padded = valid_ids[:total_positions] + [""] * max(0, total_positions - len(valid_ids))

    def display_text(sample_id: str) -> str:
        if not sample_id:
            return fill_value
        return sample_id[-digits_to_show:]

    _samples_collected = samples_collected_set or set()
    _pallet_map = pallet_map or {}
    _packed_set = packed_set or set()
    pallets_in_rack = sorted({_pallet_map[sid] for sid in ids_padded if sid and sid in _pallet_map})
    has_sc_in_rack = any(sid in _samples_collected for sid in ids_padded if sid)
    has_not_manifest_in_rack = any(sid in not_manifest_set for sid in ids_padded if sid)
    has_packed_in_rack = any(sid in _packed_set for sid in ids_padded if sid)
    # Build legend HTML as a Python string to avoid 4-space indentation being
    # misinterpreted as a Markdown code block by Streamlit's renderer.
    _legend_parts: List[str] = []
    if pallets_in_rack:
        for _p in pallets_in_rack:
            _legend_parts.append(f'<span class="legend-item"><span class="swatch pallet-{_p}"></span> Pallet {_p}</span>')
    else:
        _legend_parts.append('<span class="legend-item"><span class="swatch present"></span> In unit status</span>')
    if has_not_manifest_in_rack:
        _legend_parts.append('<span class="legend-item"><span class="swatch not-manifest"></span> Not in manifest</span>')
    if has_sc_in_rack:
        _legend_parts.append('<span class="legend-item"><span class="swatch samples-collected"></span> Rejected (samples collected)</span>')
    if has_packed_in_rack:
        _legend_parts.append('<span class="legend-item"><span class="swatch packed-swatch"></span> Already packed</span>')
    _legend_parts.append('<span class="legend-item"><span class="swatch blank"></span> Empty</span>')
    legend_html = "".join(_legend_parts)
    cells_html: List[str] = []
    for sample_id in ids_padded:
        is_blank = not bool(sample_id)
        is_not_manifest = (sample_id in not_manifest_set) if sample_id else False
        is_samples_collected = (sample_id in _samples_collected) if sample_id else False
        is_packed = (sample_id in _packed_set) if sample_id else False
        pallet_num = _pallet_map.get(sample_id) if sample_id else None

        classes = ["rack-cell"]
        if is_blank:
            classes.append("blank")
        elif is_not_manifest:
            classes.append("not-manifest")
        elif is_samples_collected:
            classes.append("samples-collected")
        elif pallet_num is not None:
            classes.append(f"pallet-{pallet_num}")
        else:
            classes.append("present")

        # packed is an additive overlay class – applied on top of the colour class
        if is_packed:
            classes.append("packed")

        tooltip = (sample_id + " [PACKED]") if is_packed and sample_id else (sample_id if sample_id else "Empty")
        cells_html.append(
            f'<div class="{" ".join(classes)}" title="{tooltip}">{display_text(sample_id)}</div>'
        )

    cells_joined = "\n".join(cells_html)

    _date_range_html = (
        f'<span class="rack-date-range">{date_range_str}</span>'
        if date_range_str
        else ""
    )

    html = f"""
<div class="rack-wrap">
  <div class="rack-title">{title}</div>

  <div class="rack-legend">
    {legend_html}
    {_date_range_html}
  </div>

  <div class="rack-grid">
    {cells_joined}
  </div>
</div>

<style>
  /* ── Rack container ────────────────────────────────── */
  .rack-wrap {{
    padding: 14px 14px 16px 14px;
    border: 1px solid #d0d7de;
    border-radius: 12px;
    background: #f6f8fa;
    width: fit-content;
    max-width: 100%;
    overflow-x: auto;
  }}

  .rack-title {{
    font-weight: 800;
    font-size: 15px;
    margin: 0 0 10px 0;
    color: #1f2328;
    letter-spacing: -0.02em;
  }}

  /* ── Legend ─────────────────────────────────────────── */
  .rack-legend {{
    display: flex;
    gap: 16px;
    align-items: center;
    font-size: 12px;
    font-weight: 600;
    color: #424a53;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }}

  .rack-date-range {{
    margin-left: auto;
    font-size: 11px;
    font-weight: 400;
    opacity: 0.6;
    white-space: nowrap;
    font-style: italic;
  }}

  .legend-item {{
    display: inline-flex;
    gap: 6px;
    align-items: center;
  }}

  .swatch {{
    width: 14px;
    height: 14px;
    border-radius: 4px;
    border: 1px solid rgba(0,0,0,0.25);
    display: inline-block;
    flex-shrink: 0;
  }}

  /* Status swatches */
  .swatch.present           {{ background: #4ade80; }}
  .swatch.not-manifest      {{ background: #fbbf24; }}
  .swatch.samples-collected {{ background: #f87171; }}
  .swatch.blank             {{ background: #e7ecf0; border-color: rgba(0,0,0,0.12); }}

  /* Pallet swatches — one per hue family */
  .swatch.pallet-1  {{ background: #60a5fa; }}   /* sky blue   */
  .swatch.pallet-2  {{ background: #c084fc; }}   /* violet     */
  .swatch.pallet-3  {{ background: #34d399; }}   /* emerald    */
  .swatch.pallet-4  {{ background: #fb923c; }}   /* orange     */
  .swatch.pallet-5  {{ background: #f472b6; }}   /* pink       */
  .swatch.pallet-6  {{ background: #a3e635; }}   /* lime       */

  /* ── Grid ───────────────────────────────────────────── */
  .rack-grid {{
    --cell-w: 42px;
    --gap: 5px;
    display: grid;
    grid-auto-rows: 36px;
    gap: var(--gap);
    grid-template-columns: repeat(18, var(--cell-w));
  }}

  /* ── Mobile: shrink cells so all 18 columns fit on screen ── */
  @media (max-width: 900px) {{
    .rack-grid {{
      --cell-w: clamp(18px, 4.8vw, 38px);
      --gap: 3px;
      grid-auto-rows: clamp(22px, 5.5vw, 34px);
    }}
    .rack-cell {{
      font-size: clamp(8px, 2.2vw, 13px);
      border-radius: 4px;
    }}
    .rack-wrap {{
      padding: 8px 6px 10px 6px;
    }}
    .rack-title {{
      font-size: 13px;
    }}
    .rack-legend {{
      font-size: 10px;
      gap: 8px;
    }}
  }}

  /* ── Base cell ──────────────────────────────────────── */
  .rack-cell {{
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 7px;
    border: 1px solid rgba(0,0,0,0.20);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 0.4px;
    user-select: none;
    transition: transform 0.07s ease, filter 0.07s ease;
    position: relative;
  }}

  /* ── Status cells ───────────────────────────────────── */
  .rack-cell.present {{
    background: #4ade80;
    color: #14532d;
  }}

  .rack-cell.not-manifest {{
    background: #fbbf24;
    color: #78350f;
  }}

  .rack-cell.samples-collected {{
    background: #f87171;
    color: #7f1d1d;
  }}

  /* ── Pallet cells — vibrant, maximally distinct ─────── */
  .rack-cell.pallet-1 {{ background: #60a5fa; color: #1e3a8a; }}  /* sky blue   */
  .rack-cell.pallet-2 {{ background: #c084fc; color: #3b0764; }}  /* violet     */
  .rack-cell.pallet-3 {{ background: #34d399; color: #064e3b; }}  /* emerald    */
  .rack-cell.pallet-4 {{ background: #fb923c; color: #7c2d12; }}  /* orange     */
  .rack-cell.pallet-5 {{ background: #f472b6; color: #831843; }}  /* pink       */
  .rack-cell.pallet-6 {{ background: #a3e635; color: #365314; }}  /* lime       */

  /* ── Empty cell ─────────────────────────────────────── */
  .rack-cell.blank {{
    background: #eef1f4;
    color: #c6ccd2;
    border-color: rgba(0,0,0,0.06);
  }}

  /* ── Packed strikethrough ───────────────────────────── */
  .rack-cell.packed::after {{
    content: '';
    position: absolute;
    left: 4px;
    right: 4px;
    top: 50%;
    height: 2.5px;
    background: rgba(0, 0, 0, 0.55);
    transform: translateY(-50%);
    pointer-events: none;
    border-radius: 2px;
  }}

  .swatch.packed-swatch {{
    background: #9ca3af;
    position: relative;
  }}
  .swatch.packed-swatch::after {{
    content: '';
    position: absolute;
    left: 1px; right: 1px;
    top: 50%;
    height: 2px;
    background: rgba(0, 0, 0, 0.6);
    transform: translateY(-50%);
    border-radius: 1px;
  }}

  /* ── Column-group separators (after col 6 and col 12) ── */
  /* Dark inset shadow = "wider gap" illusion — visible on every cell colour  */
  .rack-grid > .rack-cell:nth-child(18n + 6),
  .rack-grid > .rack-cell:nth-child(18n + 12) {{
    box-shadow: inset -2px 0 0 rgba(0,0,0,0.70);
  }}
  .rack-grid > .rack-cell:nth-child(18n + 7),
  .rack-grid > .rack-cell:nth-child(18n + 13) {{
    box-shadow: inset 2px 0 0 rgba(0,0,0,0.70);
  }}

  /* ── Hover ──────────────────────────────────────────── */
  .rack-cell:hover {{
    transform: translateY(-2px);
    filter: brightness(1.05) drop-shadow(0 4px 10px rgba(0,0,0,0.20));
    z-index: 1;
  }}
</style>
"""
    return html


# ---------------------------------------------------------------------------
# Rack fullscreen dialog (mobile-friendly expand view)
@st.dialog("🔍 Rack — Full View", width="large")
def _show_rack_fullscreen_dialog():
    """Render the saved rack HTML inside a large modal dialog."""
    _html = st.session_state.get("_rack_fs_html", "")
    if _html:
        st.markdown(_html, unsafe_allow_html=True)
    else:
        st.info("No rack to display.")


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


def generate_vi_labels_pdf(
    groups: List[Dict],
    tomorrow: datetime.date,
) -> bytes:
    """Render Visual Inspection label groups to a PDF.

    Each A4 page contains **two identical copies** of the same group label
    (top half and bottom half), so one sheet can be cut in two and one copy
    kept with the unit and one with the paperwork.

    * **Complete group** label: ``DD.MM.YYYY – DD.MM.YYYY`` date range and
      ``FIRST_ID – LAST_ID`` ID range.
    * **Last partial group** label: ``LATEST_DATE – TOMORROW`` date range
      and ``FIRST_ID –`` (end ID omitted because the group is open-ended).

    Parameters
    ----------
    groups : list of dict
        As returned by :func:`build_vi_label_groups`.
    tomorrow : datetime.date
        End date used for the last partial group label.

    Returns
    -------
    bytes
        Raw PDF content suitable for ``st.download_button``.
    """
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfbase.pdfmetrics import stringWidth
    except ImportError as exc:
        raise ImportError(
            "reportlab is required to generate PDFs. "
            "Install it with:  pip install reportlab"
        ) from exc

    buf = io.BytesIO()
    page_w, page_h = A4          # ~595 × 842 pt
    margin_x = 12 * mm
    margin_y = 14 * mm
    gap = 8 * mm

    label_w = page_w - 2 * margin_x
    label_h = (page_h - 2 * margin_y - gap) / 2

    c = rl_canvas.Canvas(buf, pagesize=A4)

    def _fit_size(text: str, font: str, max_w: float, start: int = 60) -> int:
        sz = start
        while sz > 8:
            if stringWidth(text, font, sz) <= max_w:
                return sz
            sz -= 1
        return sz

    def _draw_label(
        x: float, y: float, date_str: str,
        id_left: str, id_right: Optional[str] = None,
        label_num: Optional[int] = None,
    ) -> None:
        inner_w = label_w - 10 * mm
        cx = x + label_w / 2

        # Border
        c.setStrokeColorRGB(0.55, 0.55, 0.55)
        c.setLineWidth(0.8)
        c.rect(x, y, label_w, label_h)

        # Small label number in top-right corner
        if label_num is not None:
            c.setFont("Helvetica", 14)
            c.setFillColorRGB(0.55, 0.55, 0.55)
            num_str = str(label_num)
            num_w = stringWidth(num_str, "Helvetica", 14)
            c.drawString(x + label_w - num_w - 3 * mm, y + label_h - 4 * mm, num_str)
            c.setFillColorRGB(0, 0, 0)

        # Date range — normal font, large
        date_sz = _fit_size(date_str, "Helvetica", inner_w, start=60)
        c.setFont("Helvetica", date_sz)
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(cx, y + label_h * 0.56, date_str)

        # ID range — all normal font, same size; last 3 digits bold with 1 space before them
        sizing_str = (
            f"{id_left}    -    {id_right}" if id_right else f"{id_left}   -"
        )
        id_sz = _fit_size(sizing_str, "Helvetica", inner_w, start=28)
        space_w = stringWidth(" ", "Helvetica", id_sz)

        def _id_w(sid: str) -> float:
            pre = sid[:-3] if len(sid) > 3 else ""
            last3 = sid[-3:] if len(sid) >= 3 else sid
            return (
                stringWidth(pre, "Helvetica", id_sz)
                + space_w
                + stringWidth(last3, "Helvetica-Bold", id_sz)
            )

        gap_w = stringWidth("    ", "Helvetica", id_sz)
        dash_w = stringWidth("-", "Helvetica", id_sz)
        left_w = _id_w(id_left)
        right_w = _id_w(id_right) if id_right else 0

        total_w = left_w + gap_w + dash_w + (gap_w + right_w if id_right else 0)
        sx = cx - total_w / 2
        base_y = y + label_h * 0.28

        def _draw_id(draw_x: float, sid: str) -> float:
            pre = sid[:-3] if len(sid) > 3 else ""
            last3 = sid[-3:] if len(sid) >= 3 else sid
            if pre:
                c.setFont("Helvetica", id_sz)
                c.drawString(draw_x, base_y, pre)
                draw_x += stringWidth(pre, "Helvetica", id_sz)
            draw_x += space_w
            c.setFont("Helvetica-Bold", id_sz)
            c.drawString(draw_x, base_y, last3)
            return draw_x + stringWidth(last3, "Helvetica-Bold", id_sz)

        cur_x = _draw_id(sx, id_left)
        c.setFont("Helvetica", id_sz)
        c.drawString(cur_x + gap_w, base_y, "-")
        if id_right:
            _draw_id(cur_x + gap_w + dash_w + gap_w, id_right)

    def _label_parts(group: Dict):
        """Return (date_str, id_left, id_right) for a group."""
        if group["is_complete"]:
            d_min, d_max = group["date_min"], group["date_max"]
            date_str = (
                f"{d_min.strftime('%d.%m.%Y')} - {d_max.strftime('%d.%m.%Y')}"
                if d_min and d_max
                else (d_min or d_max).strftime("%d.%m.%Y") if (d_min or d_max) else "Date unknown"
            )
            return date_str, group["first_id"], group["last_id"]
        else:
            d_max = group["date_max"]
            date_str = (
                f"{d_max.strftime('%d.%m.%Y')} - {tomorrow.strftime('%d.%m.%Y')}"
                if d_max else f"? - {tomorrow.strftime('%d.%m.%Y')}"
            )
            return date_str, group["first_id"], None

    # Pair labels so cutting all pages in half and stacking gives sequential order:
    # top half of every page = labels 0..half-1, bottom half = labels half..n-1
    n_groups = len(groups)
    half = (n_groups + 1) // 2
    for i in range(half):
        top_ds, top_il, top_ir = _label_parts(groups[i])
        _draw_label(margin_x, margin_y + gap + label_h, top_ds, top_il, top_ir, label_num=i + 1)
        j = i + half
        if j < n_groups:
            bot_ds, bot_il, bot_ir = _label_parts(groups[j])
            _draw_label(margin_x, margin_y, bot_ds, bot_il, bot_ir, label_num=j + 1)
        c.showPage()

    c.save()
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# GitHub Gist state helpers for Visual Inspection label continuity
# ---------------------------------------------------------------------------

def _vi_gist_load() -> dict:
    """Read VI label state JSON from a GitHub Gist.

    Requires ``GITHUB_TOKEN`` and ``GIST_ID`` to be set in Streamlit secrets.
    Returns an empty dict if the Gist is unreachable or secrets are missing.
    The Gist file is named ``vi_state.json``.
    """
    import requests as _req
    try:
        token = st.secrets["GITHUB_TOKEN"]
        gist_id = st.secrets["GIST_ID"]
    except (KeyError, Exception):
        return {}
    try:
        resp = _req.get(
            f"https://api.github.com/gists/{gist_id}",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        content = resp.json().get("files", {}).get("vi_state.json", {}).get("content", "{}")
        return json.loads(content)
    except Exception:
        return {}


def _vi_gist_save(state: dict) -> bool:
    """Write VI label state JSON back to the GitHub Gist.

    Returns ``True`` on success, ``False`` otherwise.
    """
    import requests as _req
    try:
        token = st.secrets["GITHUB_TOKEN"]
        gist_id = st.secrets["GIST_ID"]
    except (KeyError, Exception):
        return False
    try:
        resp = _req.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json={"files": {"vi_state.json": {"content": json.dumps(state, indent=2)}}},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _vi_hash_id(salt: str, donation_id: str) -> str:
    """Return the SHA-256 hex digest of ``salt + donation_id``."""
    return hashlib.sha256(f"{salt}{donation_id}".encode()).hexdigest()


def _vi_find_next_start(
    us_df: pd.DataFrame, prefix: str, salt: str, target_hash: str
) -> Optional[str]:
    """Return the donation ID that comes immediately after the one matching ``target_hash``.

    Hashes every candidate ID (those starting with ``prefix``) using ``salt``
    and compares against ``target_hash``.  Returns ``None`` if the hash is not
    found or the matching ID is the last in the sorted list.
    """
    dn_col = us_df.get("Donation #", pd.Series(dtype=str)).fillna("").astype(str).str.strip()

    def _num(x: str) -> int:
        m = re.match(rf"^{re.escape(prefix)}(\d+)$", x, re.IGNORECASE)
        return int(m.group(1)) if m else int(1e18)

    candidates = sorted(
        {x for x in dn_col if x.upper().startswith(prefix.upper()) and _num(x) < int(1e18)},
        key=_num,
    )
    for idx, cid in enumerate(candidates):
        if _vi_hash_id(salt, cid) == target_hash:
            return candidates[idx + 1] if idx + 1 < len(candidates) else None
    return None


@st.cache_data(show_spinner=False)
def build_qc_release_comparison(
    qc_df: pd.DataFrame,
    us_df: pd.DataFrame,
    gs_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Compare QC-report releases against unit-status for each donation date.

    For every unique donation date found in *qc_df* the function counts:

    * **Released (QC)** – units that appear in the QC report for that date.
    * **Valid units (US)** – units in the unit-status file for that date whose
      Status is ``"Quarantine"`` (i.e. a real donation that is not a no-bleed,
      not rejected, and not sample-only).
    * **Pending** – the difference (Valid − Released), clamped to 0.
    * **Packed** *(optional)* – QC units for that date already packed in the
      shipment file (``"Samples Packed?"`` is non-empty).
    * **To pack** *(optional)* – QC units present in the shipment file but
      not yet packed.
    * **Not in manifest** *(optional)* – QC units absent from the shipment
      file entirely.

    Parameters
    ----------
    qc_df : pd.DataFrame
        DataFrame with at least columns ``Unit ID`` and ``Don. date`` as
        returned by :func:`parse_qc_report_pdf`.
    us_df : pd.DataFrame
        The raw unit-status DataFrame.  Must contain ``"Donation Date"`` and
        ``"Status"`` columns.
    gs_df : pd.DataFrame or None
        Optional Grifols shipment DataFrame.  When supplied, packed/to-pack/
        not-in-manifest counts and ID lists are included in the output.

    Returns
    -------
    tuple
        ``(summary_df, detail)`` where *summary_df* is one row per donation
        date and *detail* is a dict mapping ``date_str →
        {"packed": [...], "to_pack": [...], "not_manifest": [...]}``.
        The detail dict is empty when *gs_df* is ``None``.
    """
    if "Donation Date" not in us_df.columns:
        raise ValueError("Column 'Donation Date' not found in unit status file.")
    if "Status" not in us_df.columns:
        raise ValueError("Column 'Status' not found in unit status file.")

    # Keep only Quarantine units from unit status.
    quarantine_mask = (
        us_df["Status"].fillna("").astype(str).str.strip().str.lower() == "quarantine"
    )
    us_valid = us_df.loc[quarantine_mask].copy()
    us_valid["_date"] = us_valid["Donation Date"].map(_parse_donation_date)

    # Parse QC dates
    qc = qc_df.copy()
    qc["_date"] = qc["Don. date"].map(_parse_donation_date)

    # Units in the shipment file = already packed (the file is the packing manifest).
    # Units in QC but NOT in the shipment file = still need to be packed.
    shipment_ids: Set[str] = set()
    has_shipment = gs_df is not None and "Sample ID" in gs_df.columns
    if has_shipment:
        shipment_ids = set(gs_df["Sample ID"].dropna().astype(str).str.strip())

    rows = []
    detail: Dict[str, Dict] = {}
    for date_val in sorted(qc["_date"].dropna().unique()):
        date_str = date_val.strftime("%d.%m.%Y")
        date_uids = qc.loc[qc["_date"] == date_val, "Unit ID"].astype(str).str.strip().tolist()
        x = len(date_uids)
        y = int((us_valid["_date"] == date_val).sum())
        pending = max(0, y - x)
        row: Dict = {
            "Don. date": date_str,
            "Released (QC)": x,
            "Valid units (US)": y,
            "Pending": pending,
            "Summary": f"{x} out of {y} units released" if y > 0 else f"{x} released (date not in US file)",
        }
        if has_shipment:
            packed_list = sorted(u for u in date_uids if u in shipment_ids)
            still_to_pack_list = sorted(u for u in date_uids if u not in shipment_ids)
            row["Packed"] = len(packed_list)
            row["Still to pack"] = len(still_to_pack_list)
            detail[date_str] = {
                "packed": packed_list,
                "still_to_pack": still_to_pack_list,
            }
        rows.append(row)

    return pd.DataFrame(rows), detail


@st.cache_data(show_spinner=False)
def build_qc_packing_comparison(
    qc_df: pd.DataFrame,
    gs_df: pd.DataFrame,
    us_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Compare QC unit IDs against the Grifols shipment manifest by direct ID match.

    Packed     = QC unit ID found in shipment file ``Sample ID`` column.
    Not packed = QC unit ID absent from shipment file.

    Donation dates for unpacked units come from the unit-status file because
    the shipment-file dates can be unreliable.

    Returns
    -------
    unpacked_df : pd.DataFrame
        Columns ``Unit ID`` and ``Donation Date``.  Sorted by date then ID.
    stats : dict
        ``total_qc``, ``packed``, ``not_packed``,
        ``not_packed_by_date`` (dict date_str → count).
    """
    qc_ids: List[str] = (
        qc_df["Unit ID"].dropna().astype(str).str.strip().tolist()
        if "Unit ID" in qc_df.columns else []
    )

    shipment_ids: Set[str] = set()
    if gs_df is not None and "Sample ID" in gs_df.columns:
        shipment_ids = set(gs_df["Sample ID"].dropna().astype(str).str.strip())

    # Build date lookup from unit status (Donation # → "dd.mm.yyyy")
    us_date_map: Dict[str, str] = {}
    if (
        us_df is not None
        and "Donation #" in us_df.columns
        and "Donation Date" in us_df.columns
    ):
        for uid, raw in zip(
            us_df["Donation #"].fillna("").astype(str).str.strip(),
            us_df["Donation Date"],
        ):
            if uid:
                d = _parse_donation_date(raw)
                if d:
                    us_date_map[uid] = d.strftime("%d.%m.%Y")

    not_packed_ids = [u for u in qc_ids if u not in shipment_ids]

    not_packed_by_date: Dict[str, int] = {}
    not_packed_rows: List[Dict] = []
    for uid in not_packed_ids:
        date_str = us_date_map.get(uid, "—")
        not_packed_rows.append({"Unit ID": uid, "Donation Date": date_str})
        not_packed_by_date[date_str] = not_packed_by_date.get(date_str, 0) + 1

    not_packed_rows.sort(key=lambda r: (r["Donation Date"], r["Unit ID"]))

    unpacked_df = (
        pd.DataFrame(not_packed_rows)
        if not_packed_rows
        else pd.DataFrame(columns=["Unit ID", "Donation Date"])
    )

    return unpacked_df, {
        "total_qc": len(qc_ids),
        "packed": len(qc_ids) - len(not_packed_ids),
        "not_packed": len(not_packed_ids),
        "not_packed_by_date": not_packed_by_date,
    }


def parse_qc_report_pdf(pdf_file) -> pd.DataFrame:
    """Extract Unit ID and Don. date columns from a Grifols QC Report PDF.

    Uses word-position extraction so it works even when pdfplumber cannot
    detect a formal table structure in the PDF.  Each word is placed into a
    visual row by snapping its vertical (top) coordinate to a small grid; any
    row that contains both a Unit ID token (``F##-######``) and a date token
    (``DD.MM.YYYY``) is kept.

    Parameters
    ----------
    pdf_file : file-like object
        An uploaded PDF file (e.g. from ``st.file_uploader``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Unit ID`` and ``Don. date``.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required to read PDFs. "
            "Install it with:  pip install pdfplumber"
        )

    unit_id_pat = re.compile(r"^F\d{2}-\d{6}$")
    # The date may be fused to an adjacent token (e.g. "1004153720.03.2026"),
    # so we search *within* the token rather than requiring a full match.
    date_search_pat = re.compile(r"\d{2}\.\d{2}\.\d{4}")

    all_rows: List[Dict] = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # extract_words returns each word with its bounding box;
            # x_tolerance/y_tolerance control how close characters must be
            # to be merged into a single word.
            words = page.extract_words(x_tolerance=5, y_tolerance=5)

            # Group words into visual rows by snapping the top-coordinate
            # to a 3-pixel grid (absorbs minor vertical misalignments).
            row_buckets: Dict[int, List] = {}
            for w in words:
                bucket = round(w["top"] / 3) * 3
                row_buckets.setdefault(bucket, []).append(w)

            for bucket_y in sorted(row_buckets):
                # Sort words left-to-right within the row
                row_words = sorted(row_buckets[bucket_y], key=lambda w: w["x0"])
                texts = [w["text"].strip() for w in row_words]

                uid = next((t for t in texts if unit_id_pat.match(t)), None)

                # Search each token for an embedded date (handles the case
                # where the Donor ID and Don. date are merged without a space,
                # e.g. "1004153720.03.2026").
                don_date = None
                for t in texts:
                    m = date_search_pat.search(t)
                    if m:
                        don_date = m.group(0)
                        break

                # Only keep rows that have BOTH a Unit ID and a date on them.
                # Header/footer rows contain dates but no Unit ID, so they
                # are naturally excluded by this condition.
                if uid and don_date:
                    all_rows.append({"Unit ID": uid, "Don. date": don_date})

    return pd.DataFrame(all_rows, columns=["Unit ID", "Don. date"])


# ---------------------------------------------------------------------------
# Supabase storage helpers
# All files live inside one bucket whose name is set here or overridden via
# the SUPABASE_BUCKET secret.
# ---------------------------------------------------------------------------
_SUPABASE_BUCKET = "grifols"

# Storage folder holding the canonical Unit Status CSV database (one file per
# year, named "Unit Status(UNIT STATUS <year>) ...csv").
_US_FOLDER = "unit-status"


@st.cache_resource(show_spinner=False)
def _get_supabase_client():
    """Return a Supabase client if SUPABASE_URL / SUPABASE_KEY are in secrets.

    Decorated with @st.cache_resource so the connection is created once and
    reused across all reruns (avoids repeated TCP handshakes).
    """
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        if not url or not key:
            return None
        _bucket_override = st.secrets.get("SUPABASE_BUCKET", "")
        global _SUPABASE_BUCKET
        if _bucket_override:
            _SUPABASE_BUCKET = _bucket_override
        return create_client(url, key)
    except Exception:
        return None


def _get_supabase_error() -> str:
    """Return a human-readable reason why Supabase failed to connect, or empty string."""
    try:
        from supabase import create_client  # noqa: F401
    except ImportError:
        return "supabase package not installed (run: pip install supabase)"
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    if not url:
        return "SUPABASE_URL missing from secrets.toml"
    if not key:
        return "SUPABASE_KEY missing from secrets.toml"
    try:
        from supabase import create_client
        create_client(url, key)
        return ""
    except Exception as e:
        return str(e)


def _sb_list_files(client, folder: str) -> List[str]:
    """Return sorted file names inside a Supabase storage folder."""
    try:
        items = client.storage.from_(_SUPABASE_BUCKET).list(folder) or []
        return sorted(
            item["name"] for item in items
            if item.get("name") and not item["name"].startswith(".")
        )
    except Exception:
        return []


def _sb_download(client, path: str) -> Optional[bytes]:
    """Download raw bytes from Supabase storage. Returns None on failure."""
    try:
        return client.storage.from_(_SUPABASE_BUCKET).download(path)
    except Exception:
        return None


def _sb_upload(client, path: str, data: bytes, mime: str = "text/csv"):
    """Upsert a file to Supabase storage. Returns True on success, error string on failure."""
    try:
        client.storage.from_(_SUPABASE_BUCKET).upload(
            path, data, file_options={"content-type": mime, "upsert": "true"},
        )
        return True
    except Exception as e:
        return str(e)


def _sb_delete(client, path: str):
    """Delete a file from Supabase storage. Returns True on success, error string on failure."""
    try:
        client.storage.from_(_SUPABASE_BUCKET).remove([path])
        return True
    except Exception as e:
        return str(e)


def _sb_get_qc_cache(client, pdf_name: str) -> Optional[pd.DataFrame]:
    """Load cached QC extraction for pdf_name from Supabase (qc-cache/ folder).

    Returns a DataFrame on hit, None on miss or error.
    """
    raw = _sb_download(client, f"qc-cache/{pdf_name}.json")
    if raw is None:
        return None
    try:
        return pd.read_json(io.BytesIO(raw), orient="records")
    except Exception:
        return None


def _sb_save_qc_cache(client, pdf_name: str, df: pd.DataFrame) -> bool:
    """Persist QC extraction result to Supabase (qc-cache/{pdf_name}.json).

    Returns True on success, False on failure.
    """
    try:
        json_bytes = df.to_json(orient="records").encode("utf-8")
        return _sb_upload(client, f"qc-cache/{pdf_name}.json", json_bytes, "application/json") is True
    except Exception:
        return False


def _sb_delete_qc_cache(client, pdf_name: str) -> bool:
    """Delete the cached extraction for pdf_name. Returns True on success."""
    return _sb_delete(client, f"qc-cache/{pdf_name}.json") is True


# ---------------------------------------------------------------------------
# Excel multi-sheet helper
# ---------------------------------------------------------------------------
_EXCEL_PRIORITY_COLS: Set[str] = {
    "Donor ID", "Donation Date", "Status", "Donation #",
    "Donor Status", "Redo Panel",
    "Sample ID", "Pallet", "Comments",
}


def _score_sheet_name(sheets_raw: Dict[str, pd.DataFrame]) -> str:
    """Return the sheet name whose first 10 rows contain the most priority column values."""
    _best, _bscore = None, -1
    for _sn, _sdf in sheets_raw.items():
        _vals: Set[str] = set()
        for _si in range(min(10, len(_sdf))):
            _vals.update(
                str(v).strip() for v in _sdf.iloc[_si] if pd.notna(v) and str(v).strip()
            )
        _sc = len(_vals & _EXCEL_PRIORITY_COLS)
        if _sc > _bscore:
            _bscore = _sc
            _best = _sn
    return _best or next(iter(sheets_raw))


def _read_excel_smart(
    buf, dtype=str, sheet_strategy: str = "score", explicit_sheet: str = ""
) -> pd.DataFrame:
    """Read an Excel file, selecting the best sheet and auto-detecting the header row.

    Two-pass approach:
      1. Read all sheets with ``header=None, dtype=str`` for structure analysis
         (finds the right sheet AND the real header row even when decorative
         title rows appear above the column headers).
      2. Re-read only the chosen sheet with the caller's ``dtype`` and the
         detected ``header=`` offset, preserving original column dtypes.

    explicit_sheet: when non-empty, load that sheet name exactly (case-
                    insensitive).  Falls through to sheet_strategy on miss.
    sheet_strategy:
        "score"       — pick the sheet whose first rows contain the most
                        known column names.
        "latest"      — sort sheet names alphabetically, pick the last one.
        "unit_status" — prefer "UNIT STATUS {year}", then any "UNIT STATUS *",
                        then fall back to score.
    """
    # Slurp bytes once so we can read the buffer twice
    if isinstance(buf, io.BytesIO):
        _xb = buf.getvalue()
    elif isinstance(buf, (bytes, bytearray)):
        _xb = bytes(buf)
    else:
        _xb = buf.read()
        if hasattr(buf, "seek"):
            buf.seek(0)

    # Pass 1 — all sheets, no header assumption, everything as strings
    try:
        _raw: Dict[str, pd.DataFrame] = pd.read_excel(
            io.BytesIO(_xb), sheet_name=None, header=None, dtype=str
        )
    except Exception:
        return pd.DataFrame()
    if not _raw:
        return pd.DataFrame()

    # --- pick target sheet name ---
    _chosen: Optional[str] = None
    if explicit_sheet:
        _tgt = explicit_sheet.strip().upper()
        _chosen = next((n for n in _raw if n.strip().upper() == _tgt), None)

    if _chosen is None:
        if sheet_strategy == "latest":
            _chosen = sorted(_raw)[-1]
        elif sheet_strategy == "unit_status":
            _yr = str(datetime.date.today().year)
            for _n in _raw:
                if _n.strip().upper() == f"UNIT STATUS {_yr}":
                    _chosen = _n
                    break
            if _chosen is None:
                for _n in _raw:
                    if "UNIT STATUS" in _n.strip().upper():
                        _chosen = _n
                        break
            if _chosen is None:
                _chosen = _score_sheet_name(_raw)
        else:
            _chosen = _score_sheet_name(_raw)

    # --- detect real header row (handles decorative title rows above data) ---
    _sheet_raw = _raw[_chosen]
    _hdr_row = 0
    for _ri in range(min(10, len(_sheet_raw))):
        _rv = {
            str(v).strip() for v in _sheet_raw.iloc[_ri]
            if pd.notna(v) and str(v).strip()
        }
        if len(_rv & _EXCEL_PRIORITY_COLS) >= 2:
            _hdr_row = _ri
            break

    # Pass 2 — re-read only the chosen sheet with correct dtype and header offset
    _kw = {"dtype": dtype} if dtype is not None else {}
    try:
        _df = pd.read_excel(
            io.BytesIO(_xb), sheet_name=_chosen, header=_hdr_row, **_kw
        )
        return _df.fillna("")
    except Exception:
        # Fallback: build from the raw string data already in memory
        _data = _sheet_raw.iloc[_hdr_row + 1:].copy().reset_index(drop=True)
        _data.columns = [
            str(v).strip() if (pd.notna(v) and str(v).strip()) else f"_c{j}"
            for j, v in enumerate(_sheet_raw.iloc[_hdr_row])
        ]
        return _data.fillna("")


def _sb_file_widget(
    label: str,
    folder: str,
    uploader_key: str,
    file_types: List[str],
    client,
    accept_multiple: bool = False,
    save_mime: str = "text/csv",
):
    """File uploader augmented with optional Supabase storage.

    Without Supabase (``client=None``) behaves identically to
    ``st.file_uploader``.  When connected:

    * An uploaded file shows a **"Save to storage"** button.
    * A selectbox / multiselect lets the user load a previously-saved file
      instead of uploading again.  Downloaded bytes are cached in session
      state so repeated reruns don't re-fetch from Supabase.
    * An uploaded file always takes priority over a storage selection.
    """
    # --- standard uploader (always shown) ---
    uploaded = st.file_uploader(
        label, type=file_types, key=uploader_key,
        accept_multiple_files=accept_multiple,
    )

    if client is None:
        return uploaded

    # ---- keys used for session-state caching ----
    _ls_key = f"_sb_ls_{uploader_key}"   # cached file listing
    _ln_key = f"_sb_ln_{uploader_key}"   # last loaded name(s)
    _ld_key = f"_sb_ld_{uploader_key}"   # loaded bytes (or dict)

    # Populate the file listing once per session (invalidated after a save)
    if _ls_key not in st.session_state:
        st.session_state[_ls_key] = _sb_list_files(client, folder)
    _sb_files: List[str] = st.session_state[_ls_key]

    # ---- save button(s) for freshly uploaded files ----
    if uploaded and not accept_multiple:
        if st.button(
            f"Save '{uploaded.name}' to storage",
            key=f"{uploader_key}_sb_save",
        ):
            uploaded.seek(0)
            _raw = uploaded.read()
            uploaded.seek(0)
            _result = _sb_upload(client, f"{folder}/{uploaded.name}", _raw, mime=save_mime)
            if _result is True:
                _show_success(f"Saved to storage: **{uploaded.name}**")
                st.session_state[_ls_key] = _sb_list_files(client, folder)
            else:
                _show_error(f"Save failed: {_result}")

    elif uploaded and accept_multiple:
        if st.button(
            f"Save {len(uploaded)} file(s) to storage",
            key=f"{uploader_key}_sb_save",
        ):
            _saved, _failed = [], []
            for _uf in uploaded:
                _uf.seek(0)
                _raw = _uf.read()
                _uf.seek(0)
                if _sb_upload(client, f"{folder}/{_uf.name}", _raw, save_mime):
                    _saved.append(_uf.name)
                else:
                    _failed.append(_uf.name)
            if _saved:
                _show_success(f"Saved: {', '.join(_saved)}")
                st.session_state[_ls_key] = _sb_list_files(client, folder)
            if _failed:
                _show_warning(f"Failed to save: {', '.join(_failed)}")

    # ---- load from storage ----
    if _sb_files:
        if not accept_multiple:
            _opts = ["— select from storage —"] + _sb_files
            _sel = st.selectbox(
                "Or load from storage:",
                _opts,
                key=f"{uploader_key}_sb_pick",
            )
            if _sel and _sel != "— select from storage —":
                if st.session_state.get(_ln_key) != _sel:
                    _data = _sb_download(client, f"{folder}/{_sel}")
                    st.session_state[_ln_key] = _sel
                    st.session_state[_ld_key] = _data
        else:
            _sel_multi = st.multiselect(
                "Or load from storage:",
                _sb_files,
                key=f"{uploader_key}_sb_pick",
            )
            if _sel_multi:
                if st.session_state.get(_ln_key) != _sel_multi:
                    _multi_data: Dict[str, bytes] = {}
                    for _fn in _sel_multi:
                        _d = _sb_download(client, f"{folder}/{_fn}")
                        if _d:
                            _multi_data[_fn] = _d
                    st.session_state[_ln_key] = _sel_multi
                    st.session_state[_ld_key] = _multi_data
            else:
                st.session_state.pop(_ln_key, None)
                st.session_state.pop(_ld_key, None)
    elif not uploaded:
        _show_caption(f"No files in storage folder `{folder}/` yet — upload one above.")

    # ---- determine return value ----
    if uploaded:
        # Uploaded file wins; clear any stale storage cache so the next
        # time the uploader is empty the storage pick still works.
        st.session_state.pop(_ln_key, None)
        st.session_state.pop(_ld_key, None)
        return uploaded

    _cached_data = st.session_state.get(_ld_key)
    _cached_name = st.session_state.get(_ln_key)
    if _cached_data and _cached_name:
        import io as _io_sb
        if not accept_multiple:
            _f = _io_sb.BytesIO(_cached_data)
            _f.name = _cached_name
            _show_info(f"Using from storage: **{_cached_name}**")
            return _f
        else:
            _files_sb: List = []
            for _fn, _fd in _cached_data.items():
                _f = _io_sb.BytesIO(_fd)
                _f.name = _fn
                _files_sb.append(_f)
            if _files_sb:
                _show_info(f"Using from storage: **{', '.join(_cached_data.keys())}**")
            return _files_sb

    return [] if accept_multiple else None


def parse_master_sheet(ms_file) -> Dict:
    """Parse a master sheet CSV into structured data keyed by freezer ID.

    The CSV is expected to contain two stacked freezer tables, each starting
    with a row whose first cell begins with "Freezer ID:".  Within each table
    the structure is:
        row 0 – Freezer ID
        row 1 – Maximum Operation Capacity
        row 2 – Date (DD.MM.YYYY) followed by date strings in subsequent columns
        row 3 – Category of plasma units (header, skipped)
        rows 4-11 – 8 data category rows (columns a-h)
        row 12 – Total
        row 13 – Units remained to reach Maximum Operation Capacity

    Returns a dict mapping freezer_id → dict with keys:
        freezer_id, date_row_idx, cat_start_idx, dates, date_col_indices,
        row_labels, categories (label → {date_str: int})
    """
    ms_file.seek(0)
    raw_bytes = ms_file.read()
    for _enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            import io as _io
            raw = pd.read_csv(
                _io.BytesIO(raw_bytes), header=None, dtype=str, encoding=_enc
            ).fillna("")
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        raise ValueError(
            "Could not decode the master sheet CSV. "
            "Try saving it as UTF-8 in Excel (File → Save As → CSV UTF-8)."
        )
    n_rows, n_cols = raw.shape
    result: Dict = {}

    i = 0
    while i < n_rows:
        cell = str(raw.iloc[i, 0]).strip()
        if "Freezer ID:" in cell:
            freezer_id = cell.replace("Freezer ID:", "").strip()

            # Locate the Date row within the next 5 rows
            date_row_idx = None
            for j in range(i + 1, min(i + 6, n_rows)):
                if "Date" in str(raw.iloc[j, 0]) and "DD.MM" in str(raw.iloc[j, 0]):
                    date_row_idx = j
                    break

            if date_row_idx is None:
                i += 1
                continue

            # Collect non-empty date strings and their column positions
            dates: List[str] = []
            date_col_indices: List[int] = []
            for col in range(1, n_cols):
                v = str(raw.iloc[date_row_idx, col]).strip()
                if v and v.lower() != "nan":
                    dates.append(v)
                    date_col_indices.append(col)

            # Category data starts 2 rows after the date row
            # (skipping the "Category of plasma units" header row)
            cat_start_idx = date_row_idx + 2

            row_labels: List[str] = []
            categories: Dict[str, Dict[str, int]] = {}
            for k in range(8):
                ridx = cat_start_idx + k
                if ridx >= n_rows:
                    break
                label = str(raw.iloc[ridx, 0]).strip()
                row_labels.append(label)
                vals: Dict[str, int] = {}
                for date, col in zip(dates, date_col_indices):
                    v = str(raw.iloc[ridx, col]).strip()
                    try:
                        vals[date] = int(float(v)) if v and v.lower() not in ("nan", "") else 0
                    except (ValueError, TypeError):
                        vals[date] = 0
                categories[label] = vals

            result[freezer_id] = {
                "freezer_id": freezer_id,
                "freezer_start": i,
                "date_row_idx": date_row_idx,
                "cat_start_idx": cat_start_idx,
                "dates": dates,
                "date_col_indices": date_col_indices,
                "row_labels": row_labels,
                "categories": categories,
            }
            i = cat_start_idx + 8
        else:
            i += 1

    return result


# ---------------------------------------------------------------------------
# Donation Processing Dashboard helpers (ported from the Pavlo HTML tool)
# ---------------------------------------------------------------------------

_DP_BARCODE_RE = re.compile(r"^=C0703(\d{2})(\d{6})00$")

# PC-Blut pipe-separated column indices per role
_DP_COLS: Dict[str, Dict[str, int]] = {
    "supervisor": {
        "product": 39, "donation_num": 47, "donor_id": 38, "raw_date": 48,
        "start5": 76, "quarantine": 42, "prod_vol": 50, "vol": 51,
    },
    "staff": {
        "product": 9, "donation_num": 17, "donor_id": 8, "raw_date": 18,
        "start5": 46, "quarantine": 12, "prod_vol": 20, "vol": 21,
    },
}


def dp_parse_barcodes(raw_text: str) -> List[str]:
    """Convert scanned barcode lines into donation IDs.

    Lines matching ``=C0703XXYYYYYY00`` become ``FXX-YYYYYY``; any other
    non-empty line is kept as-is.
    """
    out: List[str] = []
    for line in (raw_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _DP_BARCODE_RE.match(line)
        out.append(f"F{m.group(1)}-{m.group(2)}" if m else line)
    return out


def dp_iso_week(date_str: str) -> str:
    """ISO week number for a DD-MM-YYYY / DD.MM.YYYY date string."""
    parts = re.split(r"[-.]", date_str.strip()) if date_str else []
    if len(parts) != 3:
        return ""
    try:
        return str(
            datetime.date(int(parts[2]), int(parts[1]), int(parts[0])).isocalendar()[1]
        )
    except (ValueError, OverflowError):
        return ""


def _dp_parse_float(s: str) -> float:
    """Mimic JS ``parseFloat``: leading numeric prefix or NaN."""
    m = re.match(r"^[+-]?(\d+\.?\d*|\.\d+)", s.strip())
    return float(m.group(0)) if m else float("nan")


def dp_parse_time(raw: str) -> Optional[int]:
    """Parse ``H:MM`` / ``H.MM`` (single-digit minutes padded to tens, e.g.
    ``9:3`` → ``9:30``) into minutes since midnight, or None if invalid."""
    s = raw.strip().replace(".", ":", 1)
    s = re.sub(r"^(\d{1,2}):(\d)$", lambda m: f"{m.group(1)}:{m.group(2)}0", s)
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    h, mn = int(m.group(1)), int(m.group(2))
    if h > 23 or mn > 59:
        return None
    return h * 60 + mn


def dp_is_zero_cycle(raw: str) -> bool:
    """True for placeholder lines like ``0``, ``00:00``, ``0.0``."""
    return re.fullmatch(r"0+([.:][0]*)?", raw.strip()) is not None


def dp_fmt_time(total_min: int) -> str:
    return f"{(total_min // 60) % 24:02d}:{total_min % 60:02d}"


def dp_validate_freeze_tracker(
    times: List[str], max_units: int, window_min: int, offset_hrs: int
) -> Dict:
    """Validate freezer loading cycles from START 5 timestamps.

    Groups timestamps into loading cycles of at most ``max_units`` units per
    ``window_min``-minute window, enforcing a cooldown between cycles.
    ``offset_hrs`` shifts all timestamps (e.g. -1 for timezone correction).

    Returns a dict with ``status`` ('empty' | 'ok' | 'error'), ``message``,
    ``rows`` (list of (time_str, count)), ``total``, ``cycle_errors``,
    ``format_errors`` and ``skipped``.
    """
    result: Dict = {
        "status": "empty", "message": "", "rows": [], "total": 0,
        "cycle_errors": [], "format_errors": [], "skipped": 0,
    }
    if not times:
        result["message"] = "No timestamps found."
        return result

    offset = offset_hrs * 60
    count_map: Dict[int, int] = {}
    bad_lines: List[str] = []
    skipped = 0
    for line in times:
        if dp_is_zero_cycle(line):
            skipped += 1
            continue
        raw = dp_parse_time(line)
        if raw is None:
            bad_lines.append(line)
            continue
        t = ((raw + offset) % 1440 + 1440) % 1440
        count_map[t] = count_map.get(t, 0) + 1
    result["skipped"] = skipped

    sorted_entries = sorted(count_map.items())
    if not sorted_entries:
        result["message"] = (
            f"Skipped {skipped} zero-cycle lines. No data."
            if skipped else "No valid timestamps."
        )
        return result

    result["rows"] = [(dp_fmt_time(t), c) for t, c in sorted_entries]
    result["total"] = sum(c for _, c in sorted_entries)

    has_errors = bool(bad_lines)
    cycles: List[Dict] = []
    cur: Optional[Dict] = None
    running_total = 0
    cycle_window_end: Optional[int] = None
    cooldown_until: Optional[int] = None

    def _close_cycle() -> None:
        nonlocal cur
        if cur is not None:
            cycles.append(cur)
            cur = None

    def _open_cycle(t: int, too_early: bool, prev_cooldown: Optional[int]) -> None:
        nonlocal cur
        _close_cycle()
        cur = {
            "index": len(cycles) + 1, "start": t, "window_end": t + window_min,
            "too_early": too_early, "prev_cooldown": prev_cooldown,
            "over_capacity": False, "over_at": None, "over_total": None,
            "next_allowed": None,
        }

    for t, count in sorted_entries:
        too_early = cooldown_until is not None and t < cooldown_until
        if too_early:
            has_errors = True

        need_new_cycle = (
            cur is None or too_early
            or (cooldown_until is not None and t >= cooldown_until)
        )
        if need_new_cycle:
            if not too_early and cooldown_until is not None and t >= cooldown_until:
                running_total = 0
            _open_cycle(t, too_early, cooldown_until)
            if too_early:
                running_total = 0
                cooldown_until = None
            cycle_window_end = t + window_min
        else:
            if cycle_window_end is not None and t >= cycle_window_end:
                _open_cycle(t, False, None)
                cycle_window_end = t + window_min

        if running_total + count > max_units:
            has_errors = True
            cur["over_capacity"] = True
            cur["over_at"] = t
            cur["over_total"] = running_total + count
            cur["next_allowed"] = t + window_min
            cooldown_until = t + window_min
            running_total = 0
            _close_cycle()
        else:
            running_total += count
            if running_total == max_units:
                cooldown_until = t + window_min
                running_total = 0
                _close_cycle()
                cycle_window_end = None
    _close_cycle()

    for cyc in cycles:
        if not (cyc["too_early"] or cyc["over_capacity"]):
            continue
        header = (
            f"Cycle #{cyc['index']} "
            f"({dp_fmt_time(cyc['start'])} – {dp_fmt_time(cyc['window_end'] - 1)})"
        )
        if cyc["over_capacity"]:
            detail = (
                f"Capacity exceeded at {dp_fmt_time(cyc['over_at'])}: "
                f"{cyc['over_total']}/{max_units} units. "
                f"Next allowed: {dp_fmt_time(cyc['next_allowed'])}."
            )
        else:
            detail = (
                f"Started too early — cooldown active until "
                f"{dp_fmt_time(cyc['prev_cooldown'])}."
            )
        result["cycle_errors"].append(f"{header}: {detail}")

    result["format_errors"] = [
        f'Format error: "{bl}" was skipped.' for bl in bad_lines
    ]
    result["status"] = "error" if has_errors else "ok"
    result["message"] = (
        "Validation failed — see errors below."
        if has_errors else "All loading cycles are valid."
    )
    return result


def dp_process_pc_blut(
    pc_blut_raw: str,
    vmt_list: List[str],
    ser_list: List[str],
    absc_list: List[str],
    role: str,
) -> Dict:
    """Process pasted PC-Blut rows against the scanned barcode lists.

    Returns a dict with ``excel_rows`` (tab-separated), ``start5_values``,
    ``missing`` (VMT-NAT IDs absent from PC-Blut), ``comp_text``,
    ``rejected_units`` and ``incomplete_units``.
    """
    cols = _DP_COLS[role]
    min_cols = max(cols.values()) + 1
    vmt_set, ser_set, absc_set = set(vmt_list), set(ser_list), set(absc_list)

    excel_rows: List[str] = []
    start5_values: List[str] = []
    processed: Set[str] = set()
    comp_date = ""
    rejected_units: List[str] = []
    incomplete_units: List[str] = []

    for line in pc_blut_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        columns = line.split("|")
        if len(columns) < min_cols:
            continue

        donor_id = columns[cols["donor_id"]].strip()
        product = columns[cols["product"]].strip()
        quarantine_raw = columns[cols["quarantine"]].strip()
        donation_num = columns[cols["donation_num"]].strip()
        raw_date = columns[cols["raw_date"]].strip()
        prod_vol = _dp_parse_float(columns[cols["prod_vol"]])
        vol = _dp_parse_float(columns[cols["vol"]])
        start5_value = columns[cols["start5"]].strip()

        if product == "No bleed":
            continue
        processed.add(donation_num)

        if not comp_date and raw_date:
            comp_date = raw_date.replace("-", ".")

        clean_reason = quarantine_raw.replace("*", "").strip()
        is_rejected = False
        if clean_reason:
            rejected_units.append(f"{donation_num}/ {donor_id} ({clean_reason})")
            is_rejected = True

        if (
            not is_rejected
            and not math.isnan(prod_vol)
            and not math.isnan(vol)
            and prod_vol - vol >= 5
        ):
            incomplete_units.append(f"{donation_num}/ {donor_id}")

        in_vmt = donation_num in vmt_set
        in_ser = donation_num in ser_set
        in_absc = donation_num in absc_set

        if product == "Test sample":
            donor_status = "SO (16 week)"
            status = "SO (16 week)" if (in_vmt and in_ser) else "Rejected"
        elif product == "SP/PPH":
            if not in_vmt:
                donor_status = "Q"
                status = "Rejected"
            else:
                status = "Quarantine"
                if in_ser and in_absc:
                    donor_status = "1"
                elif in_ser:
                    donor_status = "16"
                else:
                    donor_status = "Q"
        else:
            donor_status = "Q"
            status = "Quarantine"

        formatted_date = raw_date.replace("-", ".")
        iso_week = dp_iso_week(raw_date)
        excel_rows.append(
            f"{donation_num}\t{donor_id}\t{donor_status}\t{formatted_date}\t"
            f"{iso_week}\t{status}\t{clean_reason}"
        )
        start5_values.append(start5_value)

    comp_text = (
        f"Hello! Please see Compensation details for {comp_date or 'Unknown Date'}:\n"
        f"Total Rejected: {len(rejected_units)}\n\n"
    )
    if rejected_units:
        comp_text += "\n".join(rejected_units) + "\n"
    comp_text += f"\nTotal Incompletes: {len(incomplete_units)}\n\n"
    if incomplete_units:
        comp_text += "\n".join(incomplete_units) + "\n"

    return {
        "excel_rows": excel_rows,
        "start5_values": start5_values,
        "missing": [num for num in vmt_list if num not in processed],
        "comp_text": comp_text.strip(),
        "rejected_units": rejected_units,
        "incomplete_units": incomplete_units,
    }


_DP_RACK_CSS = """
<style>
  .dp-rack { display: inline-grid; gap: 3px; background: #f6f8fa; padding: 8px; border-radius: 8px; border: 1px solid #d0d7de; margin-bottom: 14px; }
  .dp-cell { background: #eef1f4; border: 1px solid rgba(0,0,0,0.08); width: 36px; height: 34px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #6b7280; box-sizing: border-box; border-radius: 4px; }
  .dp-filled { background: #60a5fa; border-color: #3b82f6; color: #1e3a8a; font-weight: 700; }
  .dp-rack-title { font-weight: 700; font-size: 13px; color: #0056b3; margin: 6px 0; }
</style>
"""


def dp_build_rack_html(title: str, data_list: List[str], rows: int, cols: int) -> str:
    """Render Pavlo-style rack grids (filled top-to-bottom, row-major)."""
    capacity = rows * cols
    num_racks = max(1, -(-len(data_list) // capacity))
    blocks: List[str] = []
    for r in range(num_racks):
        cells: List[str] = []
        for i in range(capacity):
            idx = r * capacity + i
            if idx < len(data_list):
                full = data_list[idx]
                parts = full.split("-")
                text = parts[1] if len(parts) > 1 and parts[1] else full
                cells.append(
                    f'<div class="dp-cell dp-filled" title="{full}">{text}</div>'
                )
            else:
                cells.append('<div class="dp-cell"></div>')
        blocks.append(
            f'<div class="dp-rack-title">{title} (Rack {r + 1} of {num_racks})</div>'
            f'<div class="dp-rack" style="grid-template-columns: repeat({cols}, 36px);">'
            + "".join(cells)
            + "</div>"
        )
    return "".join(blocks)


def render_vi_labels(us_df: pd.DataFrame, key_ns: str = "nav") -> None:
    """Render the Visual Inspection Labels UI for ``us_df``.

    Called from two places: the "Visual Inspection Labels" nav section, and
    the Donation Processing section immediately after a successful append to
    the unit status database.  ``key_ns`` namespaces every widget key and the
    session-state result slot so the two instances never collide.

    The Gist continuation state is deliberately *not* namespaced -- it is
    keyed on donation prefix, so labels resume from the right ID whichever
    entry point was used.
    """
    _vi_result_key = f"{key_ns}_vi_pdf_result"
    _subheader("Visual Inspection Labels")
    st.write(
        "Groups Quarantine units into batches of N and generates a printable "
        "PDF.  Each page has two different labels (top and bottom half). "
        "Rejected and SO units are included in the printed range but do **not** "
        "count toward the group size."
    )

    vi_c1, vi_c2, vi_c3 = st.columns([2, 3, 1])
    with vi_c1:
        vi_prefix = st.text_input(
            "Donation prefix", value="F26-", key=f"{key_ns}_vi_prefix", max_chars=20
        ).strip()
    with vi_c3:
        vi_group_size = st.number_input(
            "Group size", min_value=1, max_value=216, value=12, step=1,
            key=f"{key_ns}_vi_group_size",
        )

    # Load Gist state once per session (cached in session_state)
    if "vi_gist_state" not in st.session_state:
        st.session_state["vi_gist_state"] = _vi_gist_load()
    _vi_gist_state = st.session_state["vi_gist_state"]
    _vi_prefix_state = _vi_gist_state.get(vi_prefix, {})
    _vi_gist_configured = bool(
        _vi_prefix_state.get("salt") and _vi_prefix_state.get("last_complete_hash")
    )

    # Auto-detect start: find the ID after the last printed complete group
    _vi_auto_start = ""
    if _vi_gist_configured:
        _vi_auto_start = _vi_find_next_start(
            us_df, vi_prefix,
            _vi_prefix_state["salt"],
            _vi_prefix_state["last_complete_hash"],
        ) or ""

    # Compute default start ID: first Quarantine unit from the latest date
    _vi_default_start = ""
    try:
        _vi_dn = us_df.get("Donation #", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
        _vi_pm = _vi_dn.str.upper().str.startswith(vi_prefix.upper())
        _vi_qs = us_df.get("Status", pd.Series(dtype=str)).fillna("").astype(str).str.strip().str.lower() == "quarantine"
        _vi_filt = us_df.loc[_vi_pm & _vi_qs].copy()
        if "Donation Date" in _vi_filt.columns and not _vi_filt.empty:
            _vi_filt["_pd"] = _vi_filt["Donation Date"].map(_parse_donation_date)
            _vi_max_d = _vi_filt["_pd"].dropna().max()
            if _vi_max_d is not None:
                _vi_on_max = _vi_filt[_vi_filt["_pd"] == _vi_max_d]
                def _vi_sort_num(x):
                    m = re.match(rf"^{re.escape(vi_prefix)}(\d+)$", str(x), re.IGNORECASE)
                    return int(m.group(1)) if m else int(1e18)
                _vi_sorted = _vi_on_max.sort_values(
                    by="Donation #", key=lambda col: col.map(_vi_sort_num)
                )
                if not _vi_sorted.empty:
                    _vi_default_start = str(_vi_sorted.iloc[0]["Donation #"]).strip()
    except Exception:
        pass

    # Auto-detected takes priority over date-based default
    _vi_effective_start = _vi_auto_start or _vi_default_start

    # Show auto-detect info / reset button
    if _vi_auto_start:
        _last_updated = _vi_prefix_state.get("last_updated", "unknown date")
        _info_col, _reset_col = st.columns([5, 1])
        with _info_col:
            _show_info(
                f"Auto-detected start: **{_vi_auto_start}** "
                f"— last complete group covered donations up to {_last_updated}"
            )
        with _reset_col:
            st.write("")
            if st.button("Reset", key=f"{key_ns}_vi_reset_state", help="Clear saved state for this prefix"):
                _vi_gist_state.pop(vi_prefix, None)
                _vi_gist_save(_vi_gist_state)
                st.session_state["vi_gist_state"] = _vi_gist_state
                st.rerun()
    elif not _vi_gist_configured:
        _show_caption(
            "No saved state found for this prefix. "
            "After generating a PDF the start position will be saved automatically."
        )

    with vi_c2:
        vi_start_id = st.text_input(
            "Start from donation ID",
            value=_vi_effective_start,
            key=f"{key_ns}_vi_start_id",
            placeholder="e.g. F26-012401",
        ).strip()

    if st.button("Generate Visual Inspection Labels PDF", key=f"{key_ns}_btn_vi_labels"):
        st.session_state.pop(_vi_result_key, None)
        if not vi_start_id:
            _show_error("Please enter a start donation ID.")
        else:
            try:
                vi_groups = build_vi_label_groups(
                    us_df, vi_prefix, vi_start_id, group_size=int(vi_group_size)
                )
                if not vi_groups:
                    _show_warning("No groups found from the specified start ID.")
                else:
                    # End date = last donation date in the file + 1 day
                    _all_dates = (
                        us_df.get("Donation Date", pd.Series(dtype=str))
                        .map(_parse_donation_date)
                        .dropna()
                    )
                    _max_date = _all_dates.max() if not _all_dates.empty else None
                    _tomorrow = (
                        _max_date + datetime.timedelta(days=1)
                        if _max_date
                        else datetime.date.today()
                    )
                    vi_pdf = generate_vi_labels_pdf(vi_groups, tomorrow=_tomorrow)

                    # Save state: hash of last complete group's last ID
                    _vi_last_complete = next(
                        (g for g in reversed(vi_groups) if g["is_complete"]), None
                    )
                    if _vi_last_complete:
                        _vi_state = st.session_state.get("vi_gist_state", {})
                        _vi_ps = _vi_state.get(vi_prefix, {})
                        _vi_salt = _vi_ps.get("salt") or _secrets_mod.token_hex(16)
                        _vi_lc_date = _vi_last_complete.get("date_max")
                        _vi_state[vi_prefix] = {
                            "salt": _vi_salt,
                            "last_complete_hash": _vi_hash_id(_vi_salt, _vi_last_complete["last_id"]),
                            "last_updated": (
                                _vi_lc_date.strftime("%d.%m.%Y")
                                if _vi_lc_date
                                else datetime.date.today().strftime("%d.%m.%Y")
                            ),
                        }
                        _saved_ok = _vi_gist_save(_vi_state)
                        st.session_state["vi_gist_state"] = _vi_state
                        if _saved_ok:
                            _vi_msg = (
                                f"Generated {len(vi_groups)} label(s). "
                                f"Next session will auto-start from the continuation point."
                            )
                            _vi_msg_kind = "success"
                        else:
                            _vi_msg = (
                                f"Generated {len(vi_groups)} label(s) but could not save state "
                                f"(check GITHUB_TOKEN and GIST_ID in Streamlit secrets)."
                            )
                            _vi_msg_kind = "warning"
                    else:
                        _vi_msg = f"Generated {len(vi_groups)} label(s)."
                        _vi_msg_kind = "success"
                    # Preview table
                    _prev = []
                    for _g in vi_groups:
                        _dm, _dx = _g["date_min"], _g["date_max"]
                        if _g["is_complete"]:
                            _dr = (
                                f"{_dm.strftime('%d.%m.%Y')} – {_dx.strftime('%d.%m.%Y')}"
                                if _dm and _dx else "—"
                            )
                            _ir = f"{_g['first_id']} – {_g['last_id']}"
                        else:
                            _dr = (
                                f"{_dx.strftime('%d.%m.%Y')} – {_tomorrow.strftime('%d.%m.%Y')}"
                                if _dx else f"? – {_tomorrow.strftime('%d.%m.%Y')}"
                            )
                            _ir = f"{_g['first_id']} –"
                        _prev.append({
                            "Date range": _dr,
                            "ID range": _ir,
                            "Quarantine": _g["valid_count"],
                            "Total rows": len(_g["rows"]),
                            "Complete": "✓" if _g["is_complete"] else "(partial)",
                        })
                    # Keep result in session state so the Download / Print
                    # buttons and preview survive Streamlit reruns
                    st.session_state[_vi_result_key] = {
                        "pdf": vi_pdf,
                        "n_groups": len(vi_groups),
                        "msg": _vi_msg,
                        "msg_kind": _vi_msg_kind,
                        "preview": _prev,
                    }
            except ImportError as _e:
                _show_error(str(_e))
            except ValueError as _e:
                _show_error(str(_e))
            except Exception as _e:
                st.exception(_e)

    _vi_res = st.session_state.get(_vi_result_key)
    if _vi_res:
        if _vi_res["msg_kind"] == "warning":
            _show_warning(_vi_res["msg"])
        else:
            _show_success(_vi_res["msg"])
        _dl_col, _pr_col = st.columns([1, 1])
        with _dl_col:
            st.download_button(
                label=f"⬇ Download PDF ({_vi_res['n_groups']} label pages)",
                data=_vi_res["pdf"],
                file_name="vi_labels.pdf",
                mime="application/pdf",
                key=f"{key_ns}_vi_pdf_dl",
            )
        with _pr_col:
            _vi_b64 = base64.b64encode(_vi_res["pdf"]).decode()
            _vi_print_html = """
<button onclick="printVI()" style="
  background:#1e6fbf;color:#fff;border:none;padding:0.45rem 1rem;
  border-radius:0.375rem;cursor:pointer;font-size:0.875rem;
  font-family:sans-serif;width:100%;margin-top:4px;">
  &#128438;&nbsp;Print PDF
</button>
<script>
var _viUrl=null;
function _viBlobUrl(){
  if(_viUrl)return _viUrl;
  var bin=atob("__VI_B64__");
  var arr=new Uint8Array(bin.length);
  for(var i=0;i<bin.length;i++)arr[i]=bin.charCodeAt(i);
  _viUrl=URL.createObjectURL(new Blob([arr],{type:"application/pdf"}));
  return _viUrl;
}
function printVI(){
  var url=_viBlobUrl();
  // Firefox cannot print a PDF from a hidden iframe - open a tab instead
  if(navigator.userAgent.toLowerCase().indexOf("firefox")>-1){
    window.open(url,"_blank");
    return;
  }
  var old=document.getElementById("__VI_FRAME__");
  if(old)old.parentNode.removeChild(old);
  var f=document.createElement("iframe");
  f.id="__VI_FRAME__";
  f.style.cssText="position:fixed;right:0;bottom:0;width:0;height:0;border:0;";
  f.onload=function(){
    setTimeout(function(){
      try{f.contentWindow.focus();f.contentWindow.print();}
      catch(e){window.open(url,"_blank");}
    },200);
  };
  f.src=url;
  document.body.appendChild(f);
}
</script>
""".replace("__VI_B64__", _vi_b64).replace(
                "__VI_FRAME__", f"vi-print-frame-{key_ns}"
            )
            st.components.v1.html(_vi_print_html, height=48)
        st.table(pd.DataFrame(_vi_res["preview"]))


def main() -> None:
    """Run the Streamlit application.

    This function builds the Streamlit UI, handles user input, performs data
    processing using the helper functions defined above, and displays the
    resulting reports and data tables.  It covers both the pallet packing
    report and the unit status analysis.
    """
    st.set_page_config(page_title="Grifols", layout="wide", page_icon=None, initial_sidebar_state="expanded")

    st.markdown("""
<style>
:root {
    --accent:    #3b82f6;
    --accent-bg: rgba(59,130,246,0.10);
    --accent-hi: #1d4ed8;
    --bg-card:   #f6f8fa;
    --border:    #d0d7de;
    --text:      #1f2328;
    --text-dim:  #57606a;
}

/* ── Layout ──────────────────────────────────────────── */
.main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1300px; }

/* ── Hide Streamlit chrome ───────────────────────────── */
#MainMenu, footer { visibility: hidden; }
header { visibility: visible; }

/* ── Custom title bar ────────────────────────────────── */
.gf-title-bar {
    display: flex; align-items: center; gap: 12px;
    padding: 0.4rem 0 0.9rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}
.gf-title-accent { width: 4px; height: 28px; background: var(--accent); border-radius: 2px; flex-shrink: 0; }
.gf-title-text { font-size: 1.3rem; font-weight: 800; color: #1f2328; letter-spacing: -0.03em; line-height: 1; }
.gf-title-sub { font-size: 0.7rem; color: var(--text-dim); margin-top: 3px; letter-spacing: 0.02em; }

/* ── Section headers ─────────────────────────────────── */
.gf-section-header {
    margin: 0.25rem 0 1rem;
    padding: 0.45rem 0.75rem;
    background: var(--bg-card);
    border-left: 3px solid var(--accent);
    border-radius: 0 5px 5px 0;
}
.gf-section-header span { font-size: 1rem; font-weight: 700; color: #1f2328; letter-spacing: -0.01em; }
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="gf-title-bar">
  <div class="gf-title-accent"></div>
  <div>
    <div class="gf-title-text">Grifols</div>
    <div class="gf-title-sub">Winnipeg — Operations</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # _get_supabase_client is @st.cache_resource — returns the same instance every call.
    _sb_client = _get_supabase_client()
    _sb_error = _get_supabase_error() if _sb_client is None else ""

    shipment_file = _sb_file_widget(
        "Upload Grifols shipment file",
        "shipment",
        "shipment",
        ["csv", "xlsx", "xls"],
        _sb_client,
    )
    # Shipment ID input — only shown when an Excel file is loaded
    _gs_explicit_sheet = ""
    if shipment_file is not None:
        _gs_fname = getattr(shipment_file, "name", "") or ""
        if _gs_fname.lower().endswith((".xlsx", ".xls")):
            try:
                _gs_peek = shipment_file.read()
                shipment_file.seek(0)
                _gs_xl_names = pd.ExcelFile(io.BytesIO(_gs_peek)).sheet_names
                _gs_shipment_sheets = [n for n in _gs_xl_names if "SHIPMENT" in n.upper()]
                _gs_sheet_hint = "Available: " + ", ".join(_gs_shipment_sheets) if _gs_shipment_sheets else ""
            except Exception:
                _gs_sheet_hint = ""
            _gs_id_raw = st.text_input(
                "Shipment ID",
                key="gs_shipment_id",
                placeholder="e.g. 4000079",
                help=_gs_sheet_hint or "Enter the shipment number to load sheet 'SHIPMENT <ID>'.",
            ).strip()
            if _gs_id_raw:
                _gs_explicit_sheet = f"SHIPMENT {_gs_id_raw}"

    unit_status_file = _sb_file_widget(
        "Upload unit status file (optional)",
        "unit-status",
        "unit_status",
        ["csv", "xlsx", "xls"],
        _sb_client,
    )

    def _read_upload(
        uploaded_file, dtype=None, sheet_strategy: str = "score", explicit_sheet: str = ""
    ) -> Optional[pd.DataFrame]:
        """Read an uploaded file as a DataFrame, supporting CSV and Excel."""
        if uploaded_file is None:
            return None
        name = uploaded_file.name.lower()
        try:
            if name.endswith((".xlsx", ".xls")):
                return _read_excel_smart(
                    uploaded_file, dtype=dtype,
                    sheet_strategy=sheet_strategy, explicit_sheet=explicit_sheet,
                )
            else:
                kwargs = {"dtype": dtype} if dtype else {}
                return pd.read_csv(uploaded_file, **kwargs)
        except Exception as e:
            _show_error(f"Failed to read {uploaded_file.name}: {e}")
            return None

    # Load the DataFrames
    gs_df = _read_upload(shipment_file, sheet_strategy="latest", explicit_sheet=_gs_explicit_sheet)
    us_df = _read_upload(unit_status_file, dtype=str, sheet_strategy="unit_status")

    # Normalize whitespace in key ID columns right after loading so all
    # downstream comparisons are consistent.
    if gs_df is not None and "Sample ID" in gs_df.columns:
        gs_df["Sample ID"] = gs_df["Sample ID"].astype(str).str.strip()
        gs_df.loc[gs_df["Sample ID"] == "nan", "Sample ID"] = np.nan
    if us_df is not None and "Donation #" in us_df.columns:
        us_df["Donation #"] = us_df["Donation #"].astype(str).str.strip()
        us_df.loc[us_df["Donation #"] == "nan", "Donation #"] = np.nan

    # --- Duplicate ID warnings (shown immediately after upload, dismissible) ---
    if gs_df is not None and "Sample ID" in gs_df.columns:
        _gs_ids = gs_df["Sample ID"].dropna()
        _gs_dupes = _gs_ids[_gs_ids.duplicated(keep=False)].unique().tolist()
        if _gs_dupes:
            _gs_key = f"dismiss_gs_dupes_{shipment_file.name}_{len(_gs_dupes)}"
            if not st.session_state.get(_gs_key, False):
                _ec1, _ec2 = st.columns([0.97, 0.03])
                with _ec1:
                    _show_error(
                        f"🚨 **DUPLICATE SAMPLE IDs DETECTED in shipment file** — {len(_gs_dupes)} duplicate(s):\n\n"
                        + ", ".join(str(x) for x in sorted(_gs_dupes)),
                        icon="🚨",
                    )
                with _ec2:
                    st.write("")
                    if st.button("✕", key=f"btn_{_gs_key}", help="Dismiss"):
                        st.session_state[_gs_key] = True
                        st.rerun()

    if us_df is not None and "Donation #" in us_df.columns:
        _us_ids = us_df["Donation #"].dropna()
        _us_dupes = _us_ids[_us_ids.duplicated(keep=False)].unique().tolist()
        if _us_dupes:
            _us_key = f"dismiss_us_dupes_{unit_status_file.name}_{len(_us_dupes)}"
            if not st.session_state.get(_us_key, False):
                _uc1, _uc2 = st.columns([0.97, 0.03])
                with _uc1:
                    _show_error(
                        f"**DUPLICATE DONATION #s DETECTED in unit status file** — {len(_us_dupes)} duplicate(s):\n\n"
                        + ", ".join(str(x) for x in sorted(_us_dupes)),
                    )
                with _uc2:
                    st.write("")
                    if st.button("✕", key=f"btn_{_us_key}", help="Dismiss"):
                        st.session_state[_us_key] = True
                        st.rerun()

    # Compute and store a not_in_manifest set if both DataFrames are present
    if gs_df is not None and us_df is not None:
        try:
            # Clean unit status to remove undesired rows
            cleaned_us_df_all = clean_unit_status(us_df)
            # Prepare lists of IDs from each DataFrame
            us_ids = cleaned_us_df_all.get("Donation #", pd.Series(dtype=str)).dropna().astype(str).str.strip()
            shipment_ids = gs_df.get("Sample ID", pd.Series(dtype=str)).dropna().astype(str).str.strip()
            not_in_manifest_set = set(us_ids) - set(shipment_ids)
            st.session_state["not_in_manifest"] = sorted(not_in_manifest_set)
        except Exception:
            # If any error occurs, simply clear the not_in_manifest variable
            st.session_state["not_in_manifest"] = []
    else:
        st.session_state["not_in_manifest"] = []

    # Check URL for the unlock key — exposes all hidden sections when present
    _unlock_key = st.secrets.get("UNLOCK_KEY", "")
    _is_unlocked = bool(_unlock_key) and st.query_params.get("unlock", "") == _unlock_key

    _all_sections = [
        "Pallet Report",
        "Manual racks/Unit Status Check",
        "Visual Inspection Labels",
        "QC Report PDF Extractor",
        "Master Sheet ",
        "Storage Manager",
        "Donation Processing",
    ]
    _visible_sections = (
        _all_sections if _is_unlocked
        else [s for s in _all_sections if FEATURES.get(s, True)]
    )

    # Sidebar: navigation + section-specific controls
    with st.sidebar:
        st.header("Navigation")
        if _is_unlocked:
            _show_caption("All sections unlocked")
        st.toggle(
            "Verbose",
            value=st.session_state.get("_verbose", VERBOSE_DEFAULT),
            key="_verbose",
        )
        if _sb_client:
            _show_caption("Storage: connected")
        elif _sb_error:
            _show_caption(f"Storage: {_sb_error}")
        st.markdown("---")
        nav_section = st.radio(
            "Go to section",
            _visible_sections,
            key="nav_section",
        )
        st.markdown("---")
        if nav_section == "Pallet Report":
            st.header("Pallet Report Inputs")
            pallet_no = st.number_input(
                "Pallet number", min_value=1, step=1, value=1, format="%d"
            )
            verbose = st.checkbox("Show full F25/F26 lists", value=False)
        else:
            pallet_no = st.session_state.get("pallet_no_last", 1)
            verbose = False

    # Display shipment DataFrame preview and generate pallet report
    if nav_section == "Pallet Report" and gs_df is not None:
        _subheader("Shipment Data Preview")
        st.write(
            "Below is a preview of the shipment DataFrame (first 5 rows). "
            "Ensure the columns include at least 'Sample ID', 'Comments' and 'Samples Packed?'."
        )
        st.dataframe(gs_df.head())

        # ── Box # validation ──────────────────────────────────────────────
        if "Box #" in gs_df.columns:
            _subheader("Box Count Validation")
            _box_col = (
                gs_df["Box #"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            _box_col = _box_col[_box_col != ""]          # drop blank cells
            _box_counts = _box_col.value_counts().sort_index()
            _bad_boxes = _box_counts[_box_counts != 12]
            _total_boxes = len(_box_counts)

            if _bad_boxes.empty:
                _show_success(
                    f"All {_total_boxes} box(es) have exactly 12 samples."
                )
            else:
                st.error(
                    f"{len(_bad_boxes)} of {_total_boxes} box(es) "
                    "do not have exactly 12 samples."
                )
                _bad_df = _bad_boxes.reset_index()
                _bad_df.columns = ["Box #", "Count"]
                _bad_df["Issue"] = _bad_df["Count"].apply(
                    lambda c: f"Too many — {c}" if c > 12 else f"Too few — {c}"
                )
                st.dataframe(
                    _bad_df[["Box #", "Count", "Issue"]],
                    use_container_width=True,
                    hide_index=True,
                )


        # Optionally show a preview of the unit status file if uploaded
        if us_df is not None:
            _subheader("Unit Status Data Preview")
            st.write("First 5 rows of the unit status DataFrame after cleaning:")
            cleaned_us_df_preview = clean_unit_status(us_df)
            st.dataframe(cleaned_us_df_preview.head())
        # Process the shipment DataFrame when the user clicks the button
        if st.button("Generate Pallet Report", key="btn_gen_pallet"):
            try:
                # Find markers on the original DataFrame and get cleaned DataFrame
                cleaned_df, sop_id, eop_id, sop_row, eop_row = clean_grifols_shipment(
                    gs_df, pallet=int(pallet_no)
                )
                # Slice pallet on ORIGINAL df using marker indices
                pallet_df_raw = get_pallet_between_markers(gs_df, sop_row, eop_row)
                length_of_pallet = len(pallet_df_raw)
                # Remove packed rows and drop NaN Sample IDs
                pallet_df = remove_packed(pallet_df_raw)
                pallet_df = pallet_df.dropna(subset=["Sample ID"]).reset_index(drop=True)
                # Generate report text and get ID lists
                report_text, f25_ids, f26_ids = generate_report_text(
                    pallet_df,
                    pallet_size=length_of_pallet,
                    pallet_no=int(pallet_no),
                    sop_id=sop_id,
                    eop_id=eop_id,
                )
                # Store results in session state so they persist across reruns
                st.session_state["pallet_report_text"] = report_text
                st.session_state["pallet_f25_ids"] = f25_ids
                st.session_state["pallet_f26_ids"] = f26_ids
                st.session_state["pallet_no_last"] = int(pallet_no)

                # ── Donation date breakdown for this pallet ────────────────
                # Dates come from the unit status file (Donation # → Donation
                # Date) because shipment-file dates can be unreliable.
                _pallet_date_bd = None
                if (
                    us_df is not None
                    and "Donation #" in us_df.columns
                    and "Donation Date" in us_df.columns
                ):
                    _pd_date_map = dict(zip(
                        us_df["Donation #"].fillna("").astype(str).str.strip(),
                        us_df["Donation Date"].map(_parse_donation_date),
                    ))
                    _pd_all_ids = (
                        pallet_df_raw["Sample ID"].dropna().astype(str).str.strip().tolist()
                    )
                    _pd_to_pack_ids = set(
                        pallet_df["Sample ID"].dropna().astype(str).str.strip()
                    )
                    _pd_total: Dict[str, int] = {}
                    _pd_to_pack: Dict[str, int] = {}
                    _pd_parsed: List[datetime.date] = []
                    for _sid in _pd_all_ids:
                        if not _sid:
                            continue
                        _d = _pd_date_map.get(_sid)
                        if _d:
                            _pd_parsed.append(_d)
                        _dk = _d.strftime("%d.%m.%Y") if _d else "Unknown"
                        _pd_total[_dk] = _pd_total.get(_dk, 0) + 1
                        if _sid in _pd_to_pack_ids:
                            _pd_to_pack[_dk] = _pd_to_pack.get(_dk, 0) + 1
                    if _pd_total:
                        _pd_keys = sorted(
                            (k for k in _pd_total if k != "Unknown"),
                            key=lambda s: datetime.datetime.strptime(s, "%d.%m.%Y"),
                        )
                        if "Unknown" in _pd_total:
                            _pd_keys.append("Unknown")
                        _pallet_date_bd = {
                            "range": (
                                f"{min(_pd_parsed).strftime('%d.%m.%Y')} – "
                                f"{max(_pd_parsed).strftime('%d.%m.%Y')}"
                                if _pd_parsed else "Unknown"
                            ),
                            "rows": [
                                {
                                    "Donation Date": _dk,
                                    "Samples in pallet": _pd_total[_dk],
                                    "To pack": _pd_to_pack.get(_dk, 0),
                                }
                                for _dk in _pd_keys
                            ],
                        }
                st.session_state["pallet_date_breakdown"] = _pallet_date_bd
            except ValueError as ve:
                _show_error(str(ve))
            except Exception as e:
                st.exception(e)
        # Display the last generated pallet report if available
        if "pallet_report_text" in st.session_state:
            last_no = st.session_state.get("pallet_no_last", pallet_no)
            _subheader(f"Pallet Report (Pallet {last_no})")

        # ---- BEAUTIFUL OUTPUT (no logic changes; just parsing the existing report_text) ----
        if "pallet_report_text" in st.session_state:
            report_text = st.session_state["pallet_report_text"]
            
            def _pick(label: str) -> str:
                for line in report_text.splitlines():
                    if line.strip().startswith(label):
                        return line.split(":", 1)[1].strip()
                return ""
            
            sop = _pick("Sample ID Where Pallet Starts")
            eop = _pick("Sample ID Where Pallet Ends")
            pallet_size_val = _pick("Total number of samples in pallet")
            first_id = _pick("First sample ID to be packed")
            last_id = _pick("Last sample ID to be packed")
            f25_count = _pick("F25 count")
            f26_count = _pick("F26 count")
            total_to_pack = _pick("Total samples to pack")
            
            # KPI row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pallet size", pallet_size_val or "—")
            c2.metric("To pack", total_to_pack or "—")
            c3.metric("F25", f25_count or "—")
            c4.metric("F26", f26_count or "—")
            
            # Clean summary table
            summary_df = pd.DataFrame(
                [
                    ["Start marker ID", sop or "—"],
                    ["End marker ID", eop or "—"],
                    ["First ID to pack", first_id or "—"],
                    ["Last ID to pack", last_id or "—"],
                ],
                columns=["Field", "Value"],
            )
            st.table(summary_df)

            # ── Donation date breakdown ─────────────────────────────────
            _pd_bd = st.session_state.get("pallet_date_breakdown")
            if _pd_bd:
                st.markdown(f"**Samples from:** {_pd_bd['range']}")
                st.dataframe(
                    pd.DataFrame(_pd_bd["rows"]),
                    hide_index=True,
                    use_container_width=True,
                )
            elif us_df is None:
                _show_caption(
                    "Upload a unit status file to see the donation date range "
                    "and per-day sample counts for this pallet."
                )

            # Optional: keep the original text (collapsed)
            with st.expander("Show raw report text"):
                st.code(report_text)
                
                if verbose:
                    f25_ids = st.session_state.get("pallet_f25_ids", [])
                    f26_ids = st.session_state.get("pallet_f26_ids", [])
                    st.markdown("### F25 Sample IDs")
                    if f25_ids:
                        f25_df = pd.DataFrame({"Sample ID": f25_ids})
                        st.dataframe(f25_df)
                    else:
                        st.write("No F25 IDs found.")
                    st.markdown("### F26 Sample IDs")
                    if f26_ids:
                        f26_df = pd.DataFrame({"Sample ID": f26_ids})
                        st.dataframe(f26_df)
                    else:
                        st.write("No F26 IDs found.")
    elif nav_section == "Pallet Report":
        _show_info("Please upload a Grifols shipment file to begin the pallet report.")

    # ── Box Number Generator — always visible in Pallet Report ───────────────
    if nav_section == "Pallet Report":
        _subheader("Box Number Generator")
        _bg_col1, _bg_col2 = st.columns(2)
        with _bg_col1:
            _bg_start = st.number_input(
                "Starting box number", min_value=1, step=1, value=1,
                key="bg_start", format="%d",
            )
        with _bg_col2:
            _bg_count = st.number_input(
                "Number of boxes to generate", min_value=1, step=1, value=10,
                key="bg_count", format="%d",
            )
        _show_caption(
            f"Will generate {int(_bg_count)} box(es) × 12 = {int(_bg_count) * 12} rows  "
            f"(Box {int(_bg_start)} → Box {int(_bg_start) + int(_bg_count) - 1})"
        )
        if st.button("Generate Box Numbers", key="btn_box_gen"):
            # Copy-friendly block — Streamlit renders st.code() with a built-in
            # copy button so the user can paste numbers straight into Excel.
            _copy_nums = []
            for _bn in range(int(_bg_start), int(_bg_start) + int(_bg_count)):
                _copy_nums.extend([str(_bn)] * 12)
            st.session_state["_bg_copy_text"] = "\n".join(_copy_nums)
        if "_bg_copy_text" in st.session_state:
            with st.expander("Copy box numbers (paste into Excel)", expanded=True):
                st.code(st.session_state["_bg_copy_text"], language=None)

    # If unit status CSV is loaded, provide inputs and allow control number checks
    if nav_section == "Manual racks/Unit Status Check" and us_df is not None:
        _subheader("Manual racks/Unit Status Check")

        prefix_input = st.text_input(
            "Donation prefix", value="F26-", max_chars=20
        ).strip()

        # Allow users to enter one control number (suffix) or two separated by a comma.
        suffix_input = st.text_input(
            "Control number(s) – enter one six‑digit number or two separated by a comma",
            value="",
            placeholder="002035 or 002030,002040",
        ).strip()

        check_btn = st.button("Check Control Number(s)")

        if check_btn:
            if not suffix_input:
                _show_error("Please enter the control number(s).")
            else:
                try:
                    # Clean the unit status DataFrame once per check
                    cleaned_us_df = clean_unit_status(us_df)
                    # Determine if range or single value
                    if "," in suffix_input:
                        # Range mode: expect exactly two values
                        parts = [p.strip() for p in suffix_input.split(",") if p.strip()]
                        if len(parts) != 2:
                            _show_error("Please enter exactly two control numbers separated by a comma.")
                        else:
                            # Build full IDs (with prefix) for the range bounds
                            full_ids: List[str] = []
                            for part in parts:
                                part_upper = part.upper()
                                prefix_upper = prefix_input.upper()
                                # If the user provided the full ID (starting with prefix), accept it
                                if part_upper.startswith(prefix_upper):
                                    full_ids.append(part.strip())
                                else:
                                    # Otherwise treat as suffix and pad if numeric
                                    if part.isdigit():
                                        suffix_norm = part.zfill(6)
                                    else:
                                        suffix_norm = part
                                    full_ids.append(f"{prefix_input}{suffix_norm}")
                            # Extract start and end IDs
                            start_id, end_id = full_ids[0], full_ids[1]
                            # Function to extract numeric portion of an ID for ordering/comparison
                            def extract_num(x: str) -> int:
                                m = re.match(rf"^{re.escape(prefix_input)}(\d+)$", x)
                                if m:
                                    return int(m.group(1))
                                # If prefix does not match, return a large number to exclude
                                return int(1e18)
                            start_num = extract_num(start_id)
                            end_num = extract_num(end_id)
                            # Ensure start <= end
                            if start_num > end_num:
                                start_num, end_num = end_num, start_num
                            # Build to_remove_set for the given prefix
                            to_remove_set, samples_collected_set = process_unit_status_all(cleaned_us_df, prefix_input)
                            # Filter not_in_manifest IDs for current prefix and remove those in to_remove_set
                            not_manifest_ids = [iid for iid in st.session_state.get("not_in_manifest", []) if iid.upper().startswith(prefix_input.upper())]
                            not_manifest_ids_filtered = [iid for iid in not_manifest_ids if iid not in to_remove_set]
                            # Collect IDs within the numeric range
                            ids_between = [iid for iid in not_manifest_ids_filtered if start_num <= extract_num(iid) <= end_num]
                            ids_between_sorted = sorted(ids_between, key=extract_num)
                            if ids_between_sorted:
                                _show_success(f"IDs in not_in_manifest between {start_id} and {end_id} (excluding removed):")
                                ids_df = pd.DataFrame({"Missing IDs": ids_between_sorted})
                                st.dataframe(ids_df)
                            else:
                                _show_info("No IDs in not_in_manifest found within the specified range after excluding removed IDs.")
                            # Show rejected units whose samples were collected, within the range
                            sc_in_range = sorted(
                                [iid for iid in samples_collected_set if start_num <= extract_num(iid) <= end_num],
                                key=extract_num,
                            )
                            if sc_in_range:
                                _show_warning(f"{len(sc_in_range)} rejected unit(s) in this range have samples collected and are NOT removed:")
                                st.dataframe(pd.DataFrame({"Rejected – Samples Collected": sc_in_range}))

                            # ------------------------------------------------------------------
                            # Additional functionality: always show Rack visualization
                            try:
                                # Recompute cleaned unit status and removal set
                                cleaned_us_full = clean_unit_status(us_df)
                                to_remove_set_full, samples_collected_set_full = process_unit_status_all(cleaned_us_full, prefix_input)
                                us_ids_cleaned_set = set(
                                    cleaned_us_full.get("Donation #", pd.Series(dtype=str)).dropna().astype(str).str.strip()
                                )
                                not_manifest_set_full = set(st.session_state.get("not_in_manifest", []))
                                # Build the full range of numeric IDs
                                range_numbers_full = list(range(start_num, end_num + 1))
                                # Collect IDs present in unit_status (cleaned) and not removed
                                valid_ids_full: List[str] = []
                                for num_val in range_numbers_full:
                                    full_id_val = f"{prefix_input}{num_val:06d}"
                                    # Exclude IDs marked for removal
                                    if full_id_val in to_remove_set_full:
                                        continue
                                    # Include only those present in the cleaned unit status file
                                    if full_id_val in us_ids_cleaned_set:
                                        valid_ids_full.append(full_id_val)
                                # Construct the rack HTML and display it.
                                # Cells are coloured by pallet (if shipment file loaded),
                                # with special overrides for not-in-manifest and samples-collected.
                                pallet_map = build_pallet_map(gs_df) if gs_df is not None else {}
                                # Build packed_set: sample IDs from the shipment file
                                # where "Samples Packed?" is non-empty.
                                packed_set_rack: Set[str] = set()
                                if gs_df is not None and "Samples Packed?" in gs_df.columns and "Sample ID" in gs_df.columns:
                                    _packed_mask = gs_df["Samples Packed?"].fillna("").astype(str).str.strip().ne("")
                                    packed_set_rack = set(
                                        gs_df.loc[_packed_mask, "Sample ID"].dropna().astype(str).str.strip()
                                    )
                                st.session_state["rack_data"] = {
                                    "valid_ids": valid_ids_full,
                                    "not_manifest_set": not_manifest_set_full,
                                    "samples_collected_set": samples_collected_set_full,
                                    "pallet_map": pallet_map,
                                    "packed_set": packed_set_rack,
                                }

                            except Exception as e:
                                _show_error(f"Failed to generate rack visualisation: {e}")
                    else:
                        # Single control number check
                        part = suffix_input
                        # Normalize suffix to 6 digits if numeric
                        if part.isdigit():
                            suffix_norm = part.zfill(6)
                        else:
                            suffix_norm = part
                        control_id = f"{prefix_input}{suffix_norm}"
                        # Build the removal set
                        to_remove_set, samples_collected_set = process_unit_status_all(cleaned_us_df, prefix_input)
                        if control_id in samples_collected_set:
                            _show_warning(
                                f"{control_id} is **Rejected** but samples were collected."
                            )
                        elif control_id in to_remove_set:
                            _show_success(f"{control_id} is classified as No Bleed, Sample Only, or Rejected.")
                        else:
                            _show_error(f"{control_id} is not classified as No Bleed, Sample Only, or Rejected.")
                except Exception as e:
                    st.exception(e)

        # Rack toggle + render — lives outside the button-click guard so that
        # toggling instantly re-renders without requiring another button click.
        if "rack_data" in st.session_state:
            hide_packed = st.toggle(
                "Hide packed samples", value=False, key="rack_hide_packed"
            )
            _rd = st.session_state["rack_data"]
            _display_ids = _rd["valid_ids"]
            _display_packed = _rd["packed_set"]
            if hide_packed:
                _display_ids = [
                    "" if sid in _rd["packed_set"] else sid for sid in _display_ids
                ]
                _display_packed = set()
            else:
                show_strikethrough = st.toggle(
                    "Show strikethrough on packed samples",
                    value=True,
                    key="rack_show_strikethrough",
                )
                if not show_strikethrough:
                    _display_packed = set()
            _rack_html = build_rack_html(
                _display_ids,
                _rd["not_manifest_set"],
                samples_collected_set=_rd["samples_collected_set"],
                pallet_map=_rd["pallet_map"],
                packed_set=_display_packed,
                digits_to_show=3,
                fill_value="–",
            )
            st.markdown(_rack_html, unsafe_allow_html=True)

            # ── Date breakdown — how many samples per donation date need packing ──
            if us_df is not None and "Donation #" in us_df.columns and "Donation Date" in us_df.columns:
                _us_date_map = dict(zip(
                    us_df["Donation #"].fillna("").astype(str).str.strip(),
                    us_df["Donation Date"].map(_parse_donation_date),
                ))
                _packed_set_rack = _rd["packed_set"]
                _date_total: Dict[str, int] = {}
                _date_to_pack: Dict[str, int] = {}
                for _sid in _rd["valid_ids"]:
                    if not _sid:
                        continue
                    _d = _us_date_map.get(_sid)
                    _dk = _d.strftime("%d.%m.%Y") if _d else "Unknown"
                    _date_total[_dk] = _date_total.get(_dk, 0) + 1
                    if _sid not in _packed_set_rack:
                        _date_to_pack[_dk] = _date_to_pack.get(_dk, 0) + 1
                if _date_total:
                    _total_samples = sum(_date_total.values())
                    _total_to_pack = sum(_date_to_pack.values())
                    with st.expander(
                        f"Rack samples by donation date — "
                        f"{_total_to_pack} to pack / {_total_samples} total"
                    ):
                        _date_rows = [
                            {
                                "Donation Date": _dk,
                                "Total in rack": _date_total[_dk],
                                "Already packed": _date_total[_dk] - _date_to_pack.get(_dk, 0),
                                "To pack": _date_to_pack.get(_dk, _date_total[_dk]),
                            }
                            for _dk in sorted(_date_total)
                        ]
                        st.dataframe(
                            pd.DataFrame(_date_rows),
                            hide_index=True,
                            use_container_width=True,
                        )

            # ── Fullscreen / expand button (especially useful on mobile) ──
            if st.button(
                "⛶  Expand rack (fullscreen view)",
                key="rack_expand_btn",
                help="Opens the rack in a large modal — tap to fit your screen.",
                use_container_width=False,
            ):
                st.session_state["_rack_fs_html"] = _rack_html
                _show_rack_fullscreen_dialog()

    if nav_section == "Visual Inspection Labels" and us_df is not None:
        render_vi_labels(us_df, key_ns="nav")

    if nav_section == "QC Report PDF Extractor":
        _subheader("QC Report PDF Extractor")
        st.write(
            "Upload one or more Grifols QC Report PDFs to extract **Unit ID** and "
            "**Don. date** for each donation on the report.  Results from all files "
            "are combined and deduplicated."
        )

        # ── local upload ────────────────────────────────────────────────────
        qc_pdf_local = st.file_uploader(
            "Upload QC Report PDF(s)", type=["pdf"], key="qc_pdf",
            accept_multiple_files=True,
        )

        # ── Supabase: save + load from storage ──────────────────────────────
        _qc_ls_key = "_sb_ls_qc_pdf"
        qc_pdf_files = list(qc_pdf_local) if qc_pdf_local else []

        if _sb_client:
            # refresh listing once per session (or after a save)
            if _qc_ls_key not in st.session_state:
                st.session_state[_qc_ls_key] = _sb_list_files(_sb_client, "qc-reports")
            _qc_sb_files: List[str] = st.session_state[_qc_ls_key]

            # save freshly uploaded PDFs to storage
            if qc_pdf_local:
                if st.button(
                    f"Save {len(qc_pdf_local)} PDF(s) to storage",
                    key="qc_pdf_sb_save",
                ):
                    _qc_saved, _qc_failed = [], []
                    for _qf in qc_pdf_local:
                        _qf.seek(0)
                        _res = _sb_upload(
                            _sb_client, f"qc-reports/{_qf.name}",
                            _qf.read(), "application/pdf",
                        )
                        _qf.seek(0)
                        (_qc_saved if _res is True else _qc_failed).append(_qf.name)
                    if _qc_saved:
                        _show_success(f"Saved: {', '.join(_qc_saved)}")
                        st.session_state[_qc_ls_key] = _sb_list_files(_sb_client, "qc-reports")
                    if _qc_failed:
                        _show_error(f"Failed: {', '.join(_qc_failed)}")

            # load previously stored PDFs (only when nothing is locally uploaded)
            if _qc_sb_files:
                _qc_sel = st.multiselect(
                    "Or load from storage",
                    options=_qc_sb_files,
                    key="qc_pdf_sb_sel",
                    disabled=bool(qc_pdf_local),
                    help="Disabled while a file is uploaded above — clear the uploader first.",
                )
                if _qc_sel and not qc_pdf_local:
                    for _qfn in _qc_sel:
                        _qc_cache = f"_sb_qc_{_qfn}"
                        if _qc_cache not in st.session_state:
                            _raw = _sb_download(_sb_client, f"qc-reports/{_qfn}")
                            if _raw:
                                st.session_state[_qc_cache] = _raw
                        if _qc_cache in st.session_state:
                            # wrap bytes in a BytesIO-like object with a .name attribute
                            class _NamedBytesIO(io.BytesIO):
                                pass
                            _bio = _NamedBytesIO(st.session_state[_qc_cache])
                            _bio.name = _qfn
                            qc_pdf_files.append(_bio)
            else:
                _show_caption("No PDFs saved in storage yet.")
        else:
            _show_caption("Connect Supabase to save/load PDFs from storage.")

        if qc_pdf_files:
            debug_mode = st.checkbox("Show debug info", value=False, key="qc_debug")

            # ── Cache status preview (checks Supabase once per file set) ────────
            if _sb_client:
                _cache_key = "_qc_cache_status_" + "_".join(
                    sorted(f.name for f in qc_pdf_files)
                )
                if _cache_key not in st.session_state:
                    st.session_state[_cache_key] = {
                        f.name: _sb_get_qc_cache(_sb_client, f.name) is not None
                        for f in qc_pdf_files
                    }
                _cs: Dict[str, bool] = st.session_state[_cache_key]
                _n_cached = sum(_cs.values())
                _n_new = len(qc_pdf_files) - _n_cached
                if _n_cached:
                    _show_info(
                        f"**{_n_cached}** file(s) already cached — "
                        f"only **{_n_new}** file(s) will be re-extracted from PDF."
                        if _n_new else
                        f"All {_n_cached} file(s) are cached — results load instantly."
                    )
                _cs_df = pd.DataFrame([
                    {"File": name, "Status": "✓ cached" if hit else "⚡ will extract"}
                    for name, hit in _cs.items()
                ])
                st.dataframe(_cs_df, hide_index=True, use_container_width=True)

                # Per-file cache clear buttons
                with st.expander("Re-extract a specific file (clear its cache)"):
                    for _fname, _hit in _cs.items():
                        if _hit:
                            if st.button(f"Clear cache for {_fname}", key=f"qc_clr_{_fname}"):
                                _sb_delete_qc_cache(_sb_client, _fname)
                                st.session_state.pop(_cache_key, None)
                                st.rerun()
                        else:
                            st.caption(f"{_fname} — not cached yet")

            if st.button("Extract Data", key="btn_extract_qc_pdf"):
                try:
                    import pdfplumber as _plumber
                    all_frames: List[pd.DataFrame] = []
                    _n_from_cache = 0
                    _n_from_pdf = 0
                    for qc_pdf_file in qc_pdf_files:
                        # ── Try cache first ──────────────────────────────────
                        if _sb_client:
                            _cached_df = _sb_get_qc_cache(_sb_client, qc_pdf_file.name)
                            if _cached_df is not None and not _cached_df.empty:
                                all_frames.append(_cached_df)
                                _n_from_cache += 1
                                continue

                        # ── Extract from PDF ─────────────────────────────────
                        qc_pdf_file.seek(0)

                        if debug_mode:
                            st.markdown(f"#### Debug: {qc_pdf_file.name}")
                            unit_id_pat_dbg = re.compile(r"^F\d{2}-\d{6}$")
                            date_pat_dbg = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
                            with _plumber.open(qc_pdf_file) as _pdf:
                                for _pi, _page in enumerate(_pdf.pages[:2]):
                                    st.markdown(f"**Page {_pi + 1}**")
                                    _words = _page.extract_words(x_tolerance=5, y_tolerance=5)
                                    st.write(f"Total words extracted: {len(_words)}")
                                    _uids = [w["text"] for w in _words if unit_id_pat_dbg.match(w["text"].strip())]
                                    _dates = [w["text"] for w in _words if date_pat_dbg.match(w["text"].strip())]
                                    st.write(f"Unit IDs found: {_uids}")
                                    st.write(f"Dates found: {_dates}")
                                    if _words:
                                        st.write("First 30 words (text → top):",
                                                 [(w["text"], round(w["top"])) for w in _words[:30]])
                                    _raw = _page.extract_text() or ""
                                    with st.expander("Raw page text"):
                                        st.text(_raw[:3000])
                            qc_pdf_file.seek(0)

                        _df = parse_qc_report_pdf(qc_pdf_file)
                        if _df.empty:
                            _show_warning(f"No records found in **{qc_pdf_file.name}**.")
                        else:
                            all_frames.append(_df)
                            _n_from_pdf += 1
                            # ── Save to cache ────────────────────────────────
                            if _sb_client:
                                _sb_save_qc_cache(_sb_client, qc_pdf_file.name, _df)
                                # Invalidate the status preview so it refreshes
                                st.session_state.pop(
                                    "_qc_cache_status_" + "_".join(
                                        sorted(f.name for f in qc_pdf_files)
                                    ), None
                                )

                    if not all_frames:
                        _show_warning(
                            "No records were found in any of the uploaded PDFs. "
                            "Enable **Show debug info** and click Extract Data again "
                            "to see what pdfplumber is reading from these files."
                        )
                        st.session_state.pop("qc_extracted_df", None)
                    else:
                        qc_df = pd.concat(all_frames, ignore_index=True).drop_duplicates()
                        st.session_state["qc_extracted_df"] = qc_df
                        _parts = []
                        if _n_from_cache:
                            _parts.append(f"{_n_from_cache} from cache")
                        if _n_from_pdf:
                            _parts.append(f"{_n_from_pdf} extracted from PDF")
                        _show_info(
                            f"Combined {len(qc_pdf_files)} file(s) "
                            f"({', '.join(_parts)}): "
                            f"{len(qc_df)} record(s) after deduplication."
                        )
                except ImportError as _ie:
                    _show_error(str(_ie))
                except Exception as _pe:
                    st.exception(_pe)

        if "qc_extracted_df" in st.session_state:
            qc_result = st.session_state["qc_extracted_df"]
            _show_success(f"Extracted {len(qc_result)} donation record(s).")
            st.dataframe(qc_result, use_container_width=True)
            csv_bytes = qc_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download as CSV",
                data=csv_bytes,
                file_name="qc_report_extracted.csv",
                mime="text/csv",
                key="qc_csv_dl",
            )

            # ------------------------------------------------------------------
            # Packing Status — compare QC unit IDs against shipment manifest
            # ------------------------------------------------------------------
            st.markdown("---")
            _subheader("Packing Status")
            if gs_df is None:
                _show_info("Upload a **Grifols shipment file** to check which QC units are packed.")
            else:
                if st.button("Check Packing Status", key="btn_qc_pack_check"):
                    try:
                        _pack_df, _pack_stats = build_qc_packing_comparison(
                            qc_result, gs_df, us_df=us_df
                        )
                        st.session_state["qc_pack_df"] = _pack_df
                        st.session_state["qc_pack_stats"] = _pack_stats
                    except Exception as _pe:
                        st.exception(_pe)

                if "qc_pack_stats" in st.session_state:
                    _ps = st.session_state["qc_pack_stats"]
                    _pc1, _pc2, _pc3 = st.columns(3)
                    _pc1.metric("Total QC Units", _ps["total_qc"])
                    _pc2.metric("Packed (in manifest)", _ps["packed"])
                    _pc3.metric("Not packed", _ps["not_packed"])

                    _pack_df = st.session_state["qc_pack_df"]
                    if not _pack_df.empty:
                        _npbd = _ps["not_packed_by_date"]
                        _date_sum = pd.DataFrame(
                            [{"Donation Date": k, "Not packed": v}
                             for k, v in sorted(_npbd.items())]
                        )
                        st.markdown("**Not packed — by donation date (from unit status file)**")
                        st.dataframe(_date_sum, hide_index=True, use_container_width=True)
                        with st.expander(f"Full list — {_ps['not_packed']} unpacked unit(s)"):
                            st.dataframe(_pack_df, hide_index=True, use_container_width=True)
                        st.download_button(
                            "Download not-packed list as CSV",
                            data=_pack_df.to_csv(index=False).encode("utf-8"),
                            file_name="qc_not_packed.csv",
                            mime="text/csv",
                            key="qc_not_packed_csv_dl",
                        )
                    else:
                        _show_success("All QC units are present in the shipment manifest!")

            # ------------------------------------------------------------------
            # Release Comparison by Date — QC releases vs unit-status counts
            # ------------------------------------------------------------------
            st.markdown("---")
            _subheader("Release Comparison by Date")
            if us_df is None:
                _show_info(
                    "Upload a **unit status file** (above) to compare QC releases "
                    "against your unit status data."
                )
            else:
                if st.button("Run Comparison", key="btn_qc_compare"):
                    try:
                        cmp_df, cmp_detail = build_qc_release_comparison(
                            qc_result, us_df, gs_df=gs_df
                        )
                        st.session_state["qc_comparison_df"] = cmp_df
                        st.session_state["qc_comparison_detail"] = cmp_detail
                    except ValueError as _ve:
                        _show_error(str(_ve))
                    except Exception as _ce:
                        st.exception(_ce)

                if "qc_comparison_df" in st.session_state:
                    cmp = st.session_state["qc_comparison_df"]

                    def _row_style(row):
                        if row["Valid units (US)"] == 0:
                            # amber — date not in US file or partial
                            return ["background-color: #fef3c7; color: #92400e"] * len(row)
                        elif row["Released (QC)"] >= row["Valid units (US)"]:
                            # green — fully released
                            return ["background-color: #d1fae5; color: #065f46"] * len(row)
                        else:
                            # amber — partially released
                            return ["background-color: #fef3c7; color: #92400e"] * len(row)

                    st.dataframe(
                        cmp.style.apply(_row_style, axis=1),
                        use_container_width=True,
                        hide_index=True,
                    )

                    tc1, tc2, tc3 = st.columns(3)
                    tc1.metric("Total Released (QC)", int(cmp["Released (QC)"].sum()))
                    tc2.metric("Valid units (US)", int(cmp["Valid units (US)"].sum()))
                    tc3.metric("Still Pending", int(cmp["Pending"].sum()))

                    st.download_button(
                        label="Download comparison as CSV",
                        data=cmp.to_csv(index=False).encode("utf-8"),
                        file_name="qc_release_comparison.csv",
                        mime="text/csv",
                        key="qc_cmp_csv_dl",
                    )

    if nav_section in ["Manual racks/Unit Status Check", "Visual Inspection Labels"] and us_df is None:
        _show_info("Please upload a unit status file to use this section.")

    # =========================================================================
    # Master Sheet 
    # =========================================================================
    if nav_section == "Master Sheet ":
        _subheader("Master Sheet ")
        st.write(
            "Automatically fills the Master Sheet daily inventory table. "
            "Upload the files below, select a freezer and date, then review "
            "the calculated values before downloading the updated CSV."
        )

        _ms_c1, _ms_c2 = st.columns(2)
        with _ms_c1:
            _ms_file = _sb_file_widget(
                "Master Sheet CSV",
                "master-sheet",
                "ms_master_file",
                ["csv"],
                _sb_client,
                save_mime="text/csv",
            )
        with _ms_c2:
            _us_2025_file = _sb_file_widget(
                "2025 Unit Status (for Donor ID columns e & f)",
                "unit-status-2025",
                "ms_us_2025_file",
                ["csv", "xlsx", "xls"],
                _sb_client,
            )

        _ms_qc_pdfs = _sb_file_widget(
            "QC Report PDF(s) — required for column (c). "
            "Alternatively, extract them in the QC Report PDF Extractor section first.",
            "qc-reports",
            "ms_qc_pdf",
            ["pdf"],
            _sb_client,
            accept_multiple=True,
            save_mime="application/pdf",
        )

        # Read 2025 unit status
        _us_df_2025: Optional[pd.DataFrame] = _read_upload(_us_2025_file, dtype=str, sheet_strategy="unit_status")
        if _us_df_2025 is not None and "Donation #" in _us_df_2025.columns:
            _us_df_2025["Donation #"] = _us_df_2025["Donation #"].astype(str).str.strip()

        # Parse QC PDFs uploaded in this section (reuse existing parser)
        _qc_df_ms: Optional[pd.DataFrame] = None
        if _ms_qc_pdfs:
            try:
                import pdfplumber as _pl_ms  # noqa: F401 – just check it's available
                _ms_qc_frames: List[pd.DataFrame] = []
                for _qpdf in _ms_qc_pdfs:
                    _qpdf.seek(0)
                    _qdf = parse_qc_report_pdf(_qpdf)
                    if not _qdf.empty:
                        _ms_qc_frames.append(_qdf)
                if _ms_qc_frames:
                    _qc_df_ms = pd.concat(_ms_qc_frames, ignore_index=True).drop_duplicates()
                    _show_success(f"QC data loaded from uploaded PDF(s): {len(_qc_df_ms)} unit record(s).")
            except ImportError:
                _show_warning("pdfplumber is not installed — QC PDF cannot be parsed. Column (c) will be 0.")
            except Exception as _qe:
                _show_error(f"Failed to parse QC PDF: {_qe}")

        # Fall back to QC data extracted in the QC Report PDF Extractor section
        if _qc_df_ms is None and "qc_extracted_df" in st.session_state:
            _qc_df_ms = st.session_state["qc_extracted_df"]
            _show_info("Using QC data from the QC Report PDF Extractor section.")

        if _ms_file is None:
            _show_info("Please upload the Master Sheet CSV to begin.")
        else:
            try:
                _ms_data = parse_master_sheet(_ms_file)
            except Exception as _mse:
                _show_error(f"Failed to parse master sheet: {_mse}")
                _ms_data = {}

            if not _ms_data:
                _show_error(
                    "Could not find any freezer sections in the master sheet. "
                    "Ensure the CSV contains rows beginning with 'Freezer ID:'."
                )
            else:
                _freezer_ids = list(_ms_data.keys())

                # Freezer selection
                _sel_freezer = st.radio(
                    "Select Freezer",
                    _freezer_ids,
                    horizontal=True,
                    key="ms_freezer_select",
                    help=(
                        "Only one freezer is active at a time. "
                        "The inactive freezer shows all zeros in the master sheet."
                    ),
                )
                _fd = _ms_data[_sel_freezer]
                _ms_dates = _fd["dates"]

                if not _ms_dates:
                    _show_error("No dates found in the master sheet for the selected freezer.")
                else:
                    # Date selection — default to today if present
                    _today_str = datetime.date.today().strftime("%d.%m.%Y")
                    _default_date_idx = (
                        _ms_dates.index(_today_str)
                        if _today_str in _ms_dates
                        else len(_ms_dates) - 1
                    )
                    _sel_date_str = st.selectbox(
                        "Select Date",
                        _ms_dates,
                        index=_default_date_idx,
                        key="ms_date_select",
                    )
                    _sel_date = datetime.datetime.strptime(_sel_date_str, "%d.%m.%Y").date()

                    st.markdown("---")

                    # ----------------------------------------------------------
                    # Column (a): Quarantine units from unit status for the date
                    # ----------------------------------------------------------
                    _col_a = 0
                    _col_a_note = (
                        "Upload the current unit status file (top of page) with "
                        "'Donation Date' and 'Status' columns to auto-calculate."
                    )
                    if (
                        us_df is not None
                        and "Status" in us_df.columns
                        and "Donation Date" in us_df.columns
                    ):
                        _dm_a = us_df["Donation Date"].map(_parse_donation_date) == _sel_date
                        _sm_a = (
                            us_df["Status"].fillna("").astype(str).str.strip().str.lower()
                            == "quarantine"
                        )
                        _col_a = int((_dm_a & _sm_a).sum())
                        _col_a_note = (
                            f"Quarantine units on {_sel_date_str} from unit status: **{_col_a}**"
                        )

                    # ----------------------------------------------------------
                    # Column (b): Units packed for shipment on the selected date
                    # ----------------------------------------------------------
                    _col_b_auto = 0
                    _col_b_auto_note = (
                        "Upload shipment file and unit status file to auto-calculate."
                    )
                    if (
                        gs_df is not None
                        and us_df is not None
                        and "Sample ID" in gs_df.columns
                        and "Samples Packed?" in gs_df.columns
                        and "Donation Date" in us_df.columns
                        and "Donation #" in us_df.columns
                    ):
                        _packed_mask_b = (
                            gs_df["Samples Packed?"].fillna("").astype(str).str.strip().ne("")
                        )
                        _packed_ids_b: Set[str] = set(
                            gs_df.loc[_packed_mask_b, "Sample ID"]
                            .dropna().astype(str).str.strip()
                        )
                        _dm_b = us_df["Donation Date"].map(_parse_donation_date) == _sel_date
                        _day_ids_b: Set[str] = set(
                            us_df.loc[_dm_b, "Donation #"].dropna().astype(str).str.strip()
                        )
                        _col_b_auto = len(_packed_ids_b & _day_ids_b)
                        _col_b_auto_note = (
                            f"Packed units whose donation date is {_sel_date_str}: **{_col_b_auto}**"
                        )

                    with st.expander(
                        "Column (b) — Plasma units packed for the shipment #", expanded=True
                    ):
                        _show_caption(_col_b_auto_note)
                        _ms_manual_b = st.checkbox(
                            "Enter manually instead",
                            value=False,
                            key="ms_b_manual_cb",
                        )
                        if _ms_manual_b:
                            _col_b: int = st.number_input(
                                "Packed units (manual entry)",
                                min_value=0,
                                value=_col_b_auto,
                                step=1,
                                key="ms_col_b",
                            )
                        else:
                            _col_b = _col_b_auto

                    # ----------------------------------------------------------
                    # Column (c): QC released – packed (unpacked but released)
                    # ----------------------------------------------------------
                    _col_c = 0
                    _col_c_note = (
                        "Upload QC Report PDF(s) above (or use the QC Report PDF Extractor "
                        "section) to auto-calculate. Defaults to 0."
                    )
                    if _qc_df_ms is not None and "Don. date" in _qc_df_ms.columns:
                        _dm_c = _qc_df_ms["Don. date"].map(_parse_donation_date) == _sel_date
                        _qc_count_c = int(_dm_c.sum())
                        _col_c = max(0, _qc_count_c - _col_b)
                        _col_c_note = (
                            f"QC released on {_sel_date_str}: {_qc_count_c}. "
                            f"Packed (col b): {_col_b}. "
                            f"Unpacked & released: **{_col_c}**"
                        )

                    # ----------------------------------------------------------
                    # Column (d): a(yesterday) + d(yesterday)
                    # ----------------------------------------------------------
                    _col_d_prev_a = 0
                    _col_d_prev_d = 0
                    _col_d_note = "First date in master sheet — no previous date available, d = 0"
                    if _sel_date_str in _ms_dates:
                        _date_idx_d = _ms_dates.index(_sel_date_str)
                        if _date_idx_d > 0:
                            _prev_date_str = _ms_dates[_date_idx_d - 1]
                            for _lbl, _lbl_vals in _fd["categories"].items():
                                if "day 1 freezing" in _lbl.lower():
                                    _col_d_prev_a = _lbl_vals.get(_prev_date_str, 0)
                                elif "waiting test results" in _lbl.lower():
                                    _col_d_prev_d = _lbl_vals.get(_prev_date_str, 0)
                            _col_d_note = (
                                f"a({_prev_date_str}) [{_col_d_prev_a}] + "
                                f"d({_prev_date_str}) [{_col_d_prev_d}]"
                            )
                    _col_d = _col_d_prev_a + _col_d_prev_d

                    # ----------------------------------------------------------
                    # Columns (e) & (f): Donor ID analysis
                    # Combine current unit status + 2025 unit status
                    # ----------------------------------------------------------
                    _col_e = 0
                    _col_e_note = (
                        "Upload unit status file(s) with a 'Donor ID' column to calculate."
                    )
                    _col_f = 0
                    _col_f_note = (
                        "Upload unit status file(s) with a 'Donor ID' column to calculate."
                    )

                    _dfs_donor: List[pd.DataFrame] = []
                    if us_df is not None and "Donor ID" in us_df.columns:
                        _dfs_donor.append(us_df)
                    if _us_df_2025 is not None and "Donor ID" in _us_df_2025.columns:
                        _dfs_donor.append(_us_df_2025)

                    if _dfs_donor:
                        _combined_donor = pd.concat(_dfs_donor, ignore_index=True)

                        # Determine which date column to use
                        _date_col_donor: Optional[str] = None
                        for _dc in ("Donation Date", "Date"):
                            if _dc in _combined_donor.columns:
                                _date_col_donor = _dc
                                break

                        # Column (e): donors appearing once in the last 6 months
                        _six_months_ago = _sel_date - datetime.timedelta(days=182)
                        if _date_col_donor:
                            _combined_donor["_d"] = (
                                _combined_donor[_date_col_donor].map(_parse_donation_date)
                            )
                            _recent = _combined_donor[
                                _combined_donor["_d"].notna()
                                & (_combined_donor["_d"] >= _six_months_ago)
                                & (_combined_donor["_d"] <= _sel_date)
                            ]
                            _donor_ids_recent = (
                                _recent["Donor ID"]
                                .dropna().astype(str).str.strip()
                            )
                            _donor_ids_recent = _donor_ids_recent[_donor_ids_recent != ""]
                            _counts_recent = _donor_ids_recent.value_counts()
                            _col_e = int((_counts_recent == 1).sum())
                            _col_e_note = (
                                f"Donors with exactly 1 donation in last 6 months "
                                f"({_six_months_ago.strftime('%d.%m.%Y')} – {_sel_date_str}): "
                                f"**{_col_e}** "
                                f"(from {len(_dfs_donor)} file(s))"
                            )
                        else:
                            _donor_ids_all_e = (
                                _combined_donor["Donor ID"]
                                .dropna().astype(str).str.strip()
                            )
                            _donor_ids_all_e = _donor_ids_all_e[_donor_ids_all_e != ""]
                            _counts_all_e = _donor_ids_all_e.value_counts()
                            _col_e = int((_counts_all_e == 1).sum())
                            _col_e_note = (
                                f"No 'Donation Date' column found — counted donors appearing "
                                f"once across all data (no 6-month filter): **{_col_e}**"
                            )

                        # Column (f): donors once overall – col_e
                        _donor_ids_all_f = (
                            _combined_donor["Donor ID"]
                            .dropna().astype(str).str.strip()
                        )
                        _donor_ids_all_f = _donor_ids_all_f[_donor_ids_all_f != ""]
                        _counts_all_f = _donor_ids_all_f.value_counts()
                        _donors_once_total = int((_counts_all_f == 1).sum())
                        _col_f = max(0, _donors_once_total - _col_e)
                        _col_f_note = (
                            f"Donors appearing once in all data: {_donors_once_total}. "
                            f"Minus waiting 2nd donation (col e): {_col_e}. "
                            f"Orphan units (beyond 6 months): **{_col_f}**"
                        )

                    # ----------------------------------------------------------
                    # Columns (g) and (h): manual entry
                    # ----------------------------------------------------------
                    _gh_c1, _gh_c2 = st.columns(2)
                    with _gh_c1:
                        _col_g: int = st.number_input(
                            "g. OUP units (excluding orphan units)",
                            min_value=0,
                            value=0,
                            step=1,
                            key="ms_col_g",
                        )
                    with _gh_c2:
                        _col_h: int = st.number_input(
                            "h. Plasma units quarantined by Quality",
                            min_value=0,
                            value=0,
                            step=1,
                            key="ms_col_h",
                        )

                    _col_values_ms = [
                        _col_a, _col_b, _col_c, _col_d,
                        _col_e, _col_f, _col_g, _col_h,
                    ]
                    _col_labels_ms = [
                        "a. Plasma units for Day 1 Freezing",
                        "b. Plasma units packed for the shipment #",
                        "c. Plasma units unpacked but released by Quality",
                        "d. Plasma units waiting test results",
                        "e. Plasma units waiting 2nd donation for qualification",
                        "f. Lapsed Units without qualification beyond 6 months (orphan units)",
                        "g. OUP units (excluding orphan units)",
                        "h. Plasma units quarantined by Quality",
                    ]
                    _col_notes_ms = [
                        _col_a_note, _col_b_auto_note, _col_c_note, _col_d_note,
                        _col_e_note, _col_f_note, "(manual)", "(manual)",
                    ]

                    # Pull existing master sheet values for comparison
                    _existing_ms: List[int] = []
                    for _rl in _fd["row_labels"]:
                        _existing_ms.append(_fd["categories"].get(_rl, {}).get(_sel_date_str, 0))
                    while len(_existing_ms) < 8:
                        _existing_ms.append(0)

                    # ----------------------------------------------------------
                    # Summary display
                    # ----------------------------------------------------------
                    st.markdown("---")
                    _subheader(f"Preview — {_sel_date_str} · {_sel_freezer}")

                    _ms_kpi_cols = st.columns(4)
                    _ms_kpi_cols[0].metric("Day 1 Freezing (a)", _col_a)
                    _ms_kpi_cols[1].metric("Waiting test results (d)", _col_d)
                    _ms_kpi_cols[2].metric("Waiting 2nd donation (e)", _col_e)
                    _ms_kpi_cols[3].metric("Orphan units (f)", _col_f)

                    _ms_total = sum(_col_values_ms)
                    st.metric("Total", _ms_total)

                    _summary_ms = pd.DataFrame({
                        "Column": _col_labels_ms,
                        "Calculated": _col_values_ms,
                        "Currently in Master Sheet": _existing_ms[:8],
                        "Notes": _col_notes_ms,
                    })
                    st.dataframe(_summary_ms, use_container_width=True, hide_index=True)

                    # ----------------------------------------------------------
                    # Apply & download
                    # ----------------------------------------------------------
                    st.markdown("---")
                    if st.button(
                        "Apply to Master Sheet & Download", key="ms_apply_btn"
                    ):
                        try:
                            _ms_file.seek(0)
                            _raw_bytes_apply = _ms_file.read()
                            _raw_ms = None
                            for _enc_apply in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                                try:
                                    import io as _io_apply
                                    _raw_ms = pd.read_csv(
                                        _io_apply.BytesIO(_raw_bytes_apply),
                                        header=None, dtype=str, encoding=_enc_apply,
                                    ).fillna("")
                                    break
                                except (UnicodeDecodeError, Exception):
                                    continue
                            if _raw_ms is None:
                                _show_error("Failed to re-read master sheet for writing.")
                                raise RuntimeError("Encoding error on re-read.")

                            # Locate column index for the selected date
                            _date_row_raw = _raw_ms.iloc[_fd["date_row_idx"]]
                            _ms_col_idx: Optional[int] = None
                            for _ci, _cv in enumerate(_date_row_raw):
                                if str(_cv).strip() == _sel_date_str:
                                    _ms_col_idx = _ci
                                    break

                            if _ms_col_idx is None:
                                _show_error(
                                    f"Date {_sel_date_str} not found in master sheet headers. "
                                    "The date may be outside the range of this CSV."
                                )
                            else:
                                _cat_s = _fd["cat_start_idx"]
                                for _ki, _kv in enumerate(_col_values_ms):
                                    if _cat_s + _ki < len(_raw_ms):
                                        _raw_ms.iloc[_cat_s + _ki, _ms_col_idx] = str(_kv)

                                # Update Total row (cat_start + 8)
                                _total_ridx = _cat_s + 8
                                if _total_ridx < len(_raw_ms):
                                    _raw_ms.iloc[_total_ridx, _ms_col_idx] = str(_ms_total)

                                # Update "Units remained to reach Maximum Operation Capacity"
                                # Max capacity row is 2 rows before the date row
                                _max_cap_ridx = _fd["date_row_idx"] - 1
                                _remained_ridx = _cat_s + 9
                                if _max_cap_ridx >= 0 and _remained_ridx < len(_raw_ms):
                                    try:
                                        _max_cap = int(
                                            float(
                                                str(_raw_ms.iloc[_max_cap_ridx, _ms_col_idx]).strip()
                                            )
                                        )
                                        _raw_ms.iloc[_remained_ridx, _ms_col_idx] = str(
                                            max(0, _max_cap - _ms_total)
                                        )
                                    except (ValueError, TypeError):
                                        pass

                                _updated_csv = _raw_ms.to_csv(
                                    index=False, header=False
                                ).encode("utf-8")
                                _show_success(
                                    f"Values for {_sel_date_str} ({_sel_freezer}) applied. "
                                    "Download the updated file below."
                                )
                                st.download_button(
                                    label="⬇ Download Updated Master Sheet",
                                    data=_updated_csv,
                                    file_name="master_sheet_updated.csv",
                                    mime="text/csv",
                                    key="ms_dl_btn",
                                )
                                if _sb_client:
                                    if st.button(
                                        "Save updated Master Sheet to storage",
                                        key="ms_sb_save_updated",
                                        help="Overwrites master-sheet/master_sheet.csv in Supabase",
                                    ):
                                        _ms_save_result = _sb_upload(
                                            _sb_client,
                                            "master-sheet/master_sheet.csv",
                                            _updated_csv,
                                            "text/csv",
                                        )
                                        if _ms_save_result is True:
                                            _show_success(
                                                "Master sheet saved to Supabase storage as "
                                                "**master_sheet.csv**."
                                            )
                                            st.session_state.pop("_sb_ls_ms_master_file", None)
                                        else:
                                            _show_error(f"Failed to save to Supabase: {_ms_save_result}")
                        except Exception as _ms_apply_err:
                            st.exception(_ms_apply_err)

    # =========================================================================
    # Storage Manager
    # =========================================================================
    if nav_section == "Storage Manager":
        _subheader("Storage Manager")

        if _sb_client is None:
            st.error(
                "Supabase is not connected. "
                + (f"**Reason:** {_sb_error}" if _sb_error else "Check SUPABASE_URL and SUPABASE_KEY in secrets.")
            )
            if st.button("Retry connection", key="_sm_retry_conn"):
                _get_supabase_client.clear()
                st.rerun()
        else:
            _sm_folders = {
                "Unit Status (2026)": ("unit-status", ["csv", "xlsx", "xls"]),
                "Grifols Shipment": ("shipment", ["csv", "xlsx", "xls"]),
                "Master Sheet": ("master-sheet", ["csv"]),
                "QC Reports": ("qc-reports", ["pdf"]),
            }

            def _sm_file_mime(fname: str) -> str:
                if fname.endswith(".xlsx"):
                    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if fname.endswith(".xls"):
                    return "application/vnd.ms-excel"
                return "text/csv"

            def _sm_load_df(raw: bytes, fname: str, sheet_strategy: str = "score") -> Optional[pd.DataFrame]:
                import io as _io_sm
                _f = _io_sm.BytesIO(raw)
                try:
                    if fname.endswith((".xlsx", ".xls")):
                        return _read_excel_smart(_f, sheet_strategy=sheet_strategy)
                    for _enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                        try:
                            _f.seek(0)
                            return pd.read_csv(_f, dtype=str, encoding=_enc).fillna("")
                        except UnicodeDecodeError:
                            continue
                except Exception:
                    pass
                return None

            def _sm_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
                return df.to_csv(index=False).encode("utf-8")

            for _sm_label, (_sm_folder, _sm_types) in _sm_folders.items():
                st.markdown(f"### {_sm_label}")
                _sm_ls_key = f"_sm_ls_{_sm_folder}"
                if _sm_ls_key not in st.session_state:
                    st.session_state[_sm_ls_key] = _sb_list_files(_sb_client, _sm_folder)
                _sm_files = st.session_state[_sm_ls_key]

                if _sm_files:
                    for _sm_fname in _sm_files:
                        _sm_col1, _sm_col2, _sm_col3 = st.columns([4, 1, 1])
                        _sm_col1.write(_sm_fname)

                        # Replace button
                        _sm_replace_file = _sm_col2.file_uploader(
                            "Replace",
                            type=_sm_types,
                            key=f"_sm_replace_{_sm_folder}_{_sm_fname}",
                            label_visibility="collapsed",
                        )
                        if _sm_replace_file is not None:
                            _sm_replace_file.seek(0)
                            _sm_res = _sb_upload(
                                _sb_client,
                                f"{_sm_folder}/{_sm_fname}",
                                _sm_replace_file.read(),
                                _sm_file_mime(_sm_replace_file.name),
                            )
                            if _sm_res is True:
                                _show_success(f"Replaced **{_sm_fname}** successfully.")
                                st.session_state[_sm_ls_key] = _sb_list_files(_sb_client, _sm_folder)
                                for _k in list(st.session_state.keys()):
                                    if _sm_folder.replace("-", "_") in _k and "_sb_ln_" in _k:
                                        st.session_state.pop(_k, None)
                                st.rerun()
                            else:
                                _show_error(f"Replace failed: {_sm_res}")

                        # Delete button
                        if _sm_col3.button("Delete", key=f"_sm_del_{_sm_folder}_{_sm_fname}"):
                            _sm_del_res = _sb_delete(_sb_client, f"{_sm_folder}/{_sm_fname}")
                            if _sm_del_res is True:
                                _show_success(f"Deleted **{_sm_fname}**.")
                                st.session_state[_sm_ls_key] = _sb_list_files(_sb_client, _sm_folder)
                                for _k in list(st.session_state.keys()):
                                    if _sm_folder.replace("-", "_") in _k and "_sb_ln_" in _k:
                                        st.session_state.pop(_k, None)
                                st.rerun()
                            else:
                                _show_error(f"Delete failed: {_sm_del_res}")
                else:
                    _show_caption(f"No files stored in `{_sm_folder}/` yet.")

                # Upload new file to this folder
                with st.expander(f"Upload new file to {_sm_label}"):
                    _sm_new_file = st.file_uploader(
                        "Choose file", type=_sm_types, key=f"_sm_new_{_sm_folder}"
                    )
                    if _sm_new_file is not None:
                        if st.button("Save to storage", key=f"_sm_new_save_{_sm_folder}"):
                            _sm_new_file.seek(0)
                            _sm_new_res = _sb_upload(
                                _sb_client,
                                f"{_sm_folder}/{_sm_new_file.name}",
                                _sm_new_file.read(),
                                _sm_file_mime(_sm_new_file.name),
                            )
                            if _sm_new_res is True:
                                _show_success(f"Saved **{_sm_new_file.name}** to storage.")
                                st.session_state[_sm_ls_key] = _sb_list_files(_sb_client, _sm_folder)
                                st.rerun()
                            else:
                                _show_error(f"Upload failed: {_sm_new_res}")

                # ── Unit Status row editing ──────────────────────────────────
                if _sm_folder == "unit-status" and _sm_files:
                    with st.expander("Edit Rows in Unit Status File"):
                        _us_edit_sel = st.selectbox(
                            "Select file to edit",
                            _sm_files,
                            key="_sm_us_edit_sel",
                        )
                        _us_df_key = f"_sm_us_df_{_us_edit_sel}"

                        # Auto-load when selection changes or not yet loaded
                        if _us_df_key not in st.session_state:
                            _us_raw = _sb_download(_sb_client, f"unit-status/{_us_edit_sel}")
                            if _us_raw:
                                _loaded = _sm_load_df(_us_raw, _us_edit_sel, sheet_strategy="unit_status")
                                if _loaded is not None:
                                    st.session_state[_us_df_key] = _loaded
                                else:
                                    _show_error("Could not parse the selected file.")
                            else:
                                _show_error("Could not download the selected file from storage.")

                        if _us_df_key in st.session_state:
                            _us_df_edit: pd.DataFrame = st.session_state[_us_df_key]
                            _us_cols = list(_us_df_edit.columns)
                            _show_caption(f"{len(_us_df_edit)} rows · {len(_us_cols)} columns")

                            with st.expander("Preview last 10 rows", expanded=False):
                                st.dataframe(
                                    _us_df_edit.tail(10),
                                    use_container_width=True,
                                    hide_index=False,
                                )

                            _tab_add, _tab_find, _tab_ctrl_us = st.tabs(
                                ["Add Rows", "Edit by Date", "Edit by Control #"]
                            )

                            # ── Add Rows (paste) ─────────────────────────────
                            with _tab_add:
                                _show_caption(
                                    f"Paste rows copied from Excel. Expected column order: "
                                    f"`{'  |  '.join(_us_cols)}`"
                                )
                                _paste_text = st.text_area(
                                    "Paste rows here (tab-separated, no header)",
                                    height=200,
                                    key="_sm_paste_rows",
                                    placeholder="F26-000001\t10057852\tQ\t02.01.2026\t1\tQuarantine\t\t\tTO BE SHIPPED ON 05.01.2026",
                                )
                                if st.button("Preview & Append", key="_sm_paste_preview_btn"):
                                    if not _paste_text.strip():
                                        _show_warning("Nothing pasted.")
                                    else:
                                        import io as _io_paste
                                        try:
                                            _pasted_df = pd.read_csv(
                                                _io_paste.StringIO(_paste_text.strip()),
                                                sep="\t",
                                                header=None,
                                                dtype=str,
                                            ).fillna("")
                                            _n_pasted_cols = _pasted_df.shape[1]
                                            _n_expected = len(_us_cols)
                                            if _n_pasted_cols != _n_expected:
                                                _show_warning(
                                                    f"Pasted data has {_n_pasted_cols} columns "
                                                    f"but file has {_n_expected}. "
                                                    "Extra columns will be dropped; missing ones filled blank."
                                                )
                                            # Align columns
                                            _pasted_df.columns = (
                                                _us_cols[:_n_pasted_cols]
                                                + [f"_extra_{i}" for i in range(max(0, _n_pasted_cols - _n_expected))]
                                            )
                                            for _mc in _us_cols:
                                                if _mc not in _pasted_df.columns:
                                                    _pasted_df[_mc] = ""
                                            _pasted_df = _pasted_df[_us_cols]
                                            st.session_state["_sm_paste_preview"] = _pasted_df
                                        except Exception as _pe:
                                            _show_error(f"Could not parse pasted text: {_pe}")

                                if "_sm_paste_preview" in st.session_state:
                                    _preview_df = st.session_state["_sm_paste_preview"]
                                    st.write(f"**{len(_preview_df)} row(s) to append:**")
                                    st.dataframe(_preview_df, use_container_width=True)
                                    if st.button("Confirm — Append to file", key="_sm_paste_confirm_btn"):
                                        _updated_us = pd.concat(
                                            [_us_df_edit, _preview_df], ignore_index=True
                                        )
                                        _save_res = _sb_upload(
                                            _sb_client,
                                            f"unit-status/{_us_edit_sel}",
                                            _sm_df_to_csv_bytes(_updated_us),
                                            "text/csv",
                                        )
                                        if _save_res is True:
                                            st.session_state[_us_df_key] = _updated_us
                                            st.session_state.pop("_sm_paste_preview", None)
                                            _show_success(
                                                f"{len(_preview_df)} row(s) appended. "
                                                f"File now has {len(_updated_us)} rows."
                                            )
                                            st.rerun()
                                        else:
                                            _show_error(f"Save failed: {_save_res}")

                            # ── Edit by Date ─────────────────────────────────
                            with _tab_find:
                                if "Donation Date" not in _us_cols:
                                    st.warning("Column 'Donation Date' not found in this file.")
                                else:
                                    _date_input_str = st.text_input(
                                        "Donation date (dd.mm.yyyy)",
                                        key="_sm_date_q",
                                        placeholder="02.01.2026",
                                    )
                                    if st.button("Search by date", key="_sm_date_search_btn"):
                                        _dq_stripped = _date_input_str.strip()
                                        if not _dq_stripped:
                                            _show_warning("Enter a date to search.")
                                        else:
                                            _target_date = _parse_donation_date(_dq_stripped)
                                            if _target_date is None:
                                                _show_warning(
                                                    f"Could not parse '{_dq_stripped}'. "
                                                    "Use dd.mm.yyyy, dd-mm-yyyy or dd/mm/yyyy."
                                                )
                                            else:
                                                _date_mask = _us_df_edit["Donation Date"].apply(
                                                    lambda _v: _parse_donation_date(_v) == _target_date
                                                )
                                                _found_idx = _us_df_edit.index[_date_mask].tolist()
                                                st.session_state["_sm_found_idx"] = _found_idx
                                                st.session_state["_sm_found_key"] = _us_df_key
                                                st.session_state["_sm_found_date"] = _target_date.strftime("%d.%m.%Y")
                                                if not _found_idx:
                                                    _show_info(
                                                        f"No rows found for "
                                                        f"{_target_date.strftime('%d.%m.%Y')}."
                                                    )

                                    _found = st.session_state.get("_sm_found_idx", [])
                                    _found_key = st.session_state.get("_sm_found_key", "")
                                    _found_date_label = st.session_state.get("_sm_found_date", "")
                                    # Clear if user switched to a different file
                                    if _found_key != _us_df_key:
                                        _found = []

                                    if _found:
                                        # Filter to rows still present in the DataFrame
                                        _valid_found = [i for i in _found if i in _us_df_edit.index]
                                        st.markdown(
                                            f"**{len(_valid_found)} row(s)** for **{_found_date_label}** — "
                                            "edit directly in the table then click Save."
                                        )
                                        _edit_subset = _us_df_edit.loc[_valid_found].copy()
                                        _edited_subset = st.data_editor(
                                            _edit_subset,
                                            use_container_width=True,
                                            hide_index=False,
                                            key=f"_sm_date_editor_{_found_key}_{_found_date_label}",
                                            num_rows="fixed",
                                        )
                                        if st.button("Save Changes", key="_sm_date_save_btn"):
                                            for _col in _us_cols:
                                                if _col in _edited_subset.columns:
                                                    _us_df_edit.loc[_valid_found, _col] = (
                                                        _edited_subset[_col].values
                                                    )
                                            _save_res = _sb_upload(
                                                _sb_client,
                                                f"unit-status/{_us_edit_sel}",
                                                _sm_df_to_csv_bytes(_us_df_edit),
                                                "text/csv",
                                            )
                                            if _save_res is True:
                                                st.session_state[_us_df_key] = _us_df_edit
                                                _show_success(
                                                    f"Saved {len(_valid_found)} row(s) for "
                                                    f"{_found_date_label}."
                                                )
                                                st.rerun()
                                            else:
                                                _show_error(f"Save failed: {_save_res}")

                            # ── Edit by Control # (Unit Status) ─────────────
                            with _tab_ctrl_us:
                                _us_ctrl_col = "Donation #"
                                if _us_ctrl_col not in _us_cols:
                                    st.warning(f"Column '{_us_ctrl_col}' not found in this file.")
                                else:
                                    _us_ctrl_q = st.text_input(
                                        "Control number (Donation # suffix, e.g. 002035)",
                                        key="_sm_us_ctrl_q",
                                        placeholder="002035",
                                    )
                                    if st.button("Search", key="_sm_us_ctrl_search_btn"):
                                        if not _us_ctrl_q.strip():
                                            _show_warning("Enter a control number.")
                                        else:
                                            _us_ctrl_mask = (
                                                _us_df_edit[_us_ctrl_col]
                                                .astype(str).str.strip()
                                                .str.endswith(_us_ctrl_q.strip())
                                            )
                                            _us_ctrl_found = _us_df_edit.index[_us_ctrl_mask].tolist()
                                            st.session_state["_sm_us_ctrl_found"] = _us_ctrl_found
                                            st.session_state["_sm_us_ctrl_key"] = _us_df_key
                                            if not _us_ctrl_found:
                                                _show_info(
                                                    f"No rows where {_us_ctrl_col} ends with "
                                                    f"'{_us_ctrl_q.strip()}'."
                                                )
                                    _us_ctrl_found = st.session_state.get("_sm_us_ctrl_found", [])
                                    if st.session_state.get("_sm_us_ctrl_key", "") != _us_df_key:
                                        _us_ctrl_found = []
                                    if _us_ctrl_found:
                                        _us_ctrl_valid = [i for i in _us_ctrl_found if i in _us_df_edit.index]
                                        st.markdown(f"**{len(_us_ctrl_valid)} row(s) found**")
                                        _us_ctrl_edited = st.data_editor(
                                            _us_df_edit.loc[_us_ctrl_valid].copy(),
                                            use_container_width=True,
                                            hide_index=False,
                                            key=f"_sm_us_ctrl_editor_{_us_df_key}",
                                            num_rows="fixed",
                                        )
                                        if st.button("Save Changes", key="_sm_us_ctrl_save_btn"):
                                            for _col in _us_cols:
                                                if _col in _us_ctrl_edited.columns:
                                                    _us_df_edit.loc[_us_ctrl_valid, _col] = (
                                                        _us_ctrl_edited[_col].values
                                                    )
                                            _save_res = _sb_upload(
                                                _sb_client,
                                                f"unit-status/{_us_edit_sel}",
                                                _sm_df_to_csv_bytes(_us_df_edit),
                                                "text/csv",
                                            )
                                            if _save_res is True:
                                                st.session_state[_us_df_key] = _us_df_edit
                                                _show_success(f"Saved {len(_us_ctrl_valid)} row(s).")
                                                st.rerun()
                                            else:
                                                _show_error(f"Save failed: {_save_res}")

                # ── Shipment file row editing ────────────────────────────────
                if _sm_folder == "shipment" and _sm_files:
                    with st.expander("Edit Rows in Shipment File"):
                        _gs_edit_sel = st.selectbox(
                            "Select file to edit",
                            _sm_files,
                            key="_sm_gs_edit_sel",
                        )
                        _gs_df_key = f"_sm_gs_df_{_gs_edit_sel}"

                        if _gs_df_key not in st.session_state:
                            _gs_raw = _sb_download(_sb_client, f"shipment/{_gs_edit_sel}")
                            if _gs_raw:
                                _loaded_gs = _sm_load_df(_gs_raw, _gs_edit_sel, sheet_strategy="latest")
                                if _loaded_gs is not None:
                                    st.session_state[_gs_df_key] = _loaded_gs
                                else:
                                    _show_error("Could not parse the selected file.")
                            else:
                                _show_error("Could not download the selected file from storage.")

                        if _gs_df_key in st.session_state:
                            _gs_df_edit: pd.DataFrame = st.session_state[_gs_df_key]
                            _gs_cols = list(_gs_df_edit.columns)
                            _show_caption(f"{len(_gs_df_edit)} rows · {len(_gs_cols)} columns")

                            with st.expander("Preview last 10 rows", expanded=False):
                                st.dataframe(
                                    _gs_df_edit.tail(10),
                                    use_container_width=True,
                                    hide_index=False,
                                )

                            _gs_tab_pallet, _gs_tab_ctrl = st.tabs(
                                ["Edit by Pallet", "Edit by Control #"]
                            )

                            # ── Edit by Pallet ────────────────────────────────
                            with _gs_tab_pallet:
                                if "Comments" not in _gs_cols:
                                    st.warning("Column 'Comments' not found — cannot search by pallet.")
                                else:
                                    _gs_pallet_no = st.number_input(
                                        "Pallet number", min_value=1, step=1,
                                        value=1, key="_sm_gs_pallet_no", format="%d",
                                    )
                                    if st.button("Search pallet", key="_sm_gs_pallet_search_btn"):
                                        _p = int(_gs_pallet_no)
                                        _sop_pat = re.compile(
                                            rf"^\s*START\s+OF\s+PALLET\s+{_p}\s*$", re.IGNORECASE
                                        )
                                        _eop_pat = re.compile(
                                            rf"^\s*END\s+OF\s+PALLET\s+{_p}\s*$", re.IGNORECASE
                                        )
                                        _gs_comments = _gs_df_edit["Comments"].fillna("").astype(str)
                                        _sop_rows = _gs_df_edit.index[_gs_comments.str.match(_sop_pat)].tolist()
                                        _eop_rows = _gs_df_edit.index[_gs_comments.str.match(_eop_pat)].tolist()
                                        if not _sop_rows or not _eop_rows:
                                            _show_info(f"Pallet {_p} markers not found in Comments.")
                                            st.session_state["_sm_gs_pallet_idx"] = []
                                        else:
                                            _p_start = min(_sop_rows[0], _eop_rows[0])
                                            _p_end = max(_sop_rows[0], _eop_rows[0])
                                            _p_idx = [
                                                i for i in range(_p_start, _p_end + 1)
                                                if i in _gs_df_edit.index
                                            ]
                                            st.session_state["_sm_gs_pallet_idx"] = _p_idx
                                            st.session_state["_sm_gs_pallet_key"] = _gs_df_key
                                            st.session_state["_sm_gs_pallet_num"] = _p
                                            if not _p_idx:
                                                _show_info(f"No rows found for Pallet {_p}.")

                                    _gs_pallet_idx = st.session_state.get("_sm_gs_pallet_idx", [])
                                    if st.session_state.get("_sm_gs_pallet_key", "") != _gs_df_key:
                                        _gs_pallet_idx = []

                                    if _gs_pallet_idx:
                                        _gs_p_num = st.session_state.get("_sm_gs_pallet_num", "")
                                        st.markdown(
                                            f"**{len(_gs_pallet_idx)} row(s)** for "
                                            f"**Pallet {_gs_p_num}** — edit then click Save."
                                        )
                                        _gs_pallet_edited = st.data_editor(
                                            _gs_df_edit.loc[_gs_pallet_idx].copy(),
                                            use_container_width=True,
                                            hide_index=False,
                                            key=f"_sm_gs_pallet_editor_{_gs_df_key}_{_gs_p_num}",
                                            num_rows="fixed",
                                        )
                                        if st.button("Save Changes", key="_sm_gs_pallet_save_btn"):
                                            for _col in _gs_cols:
                                                if _col in _gs_pallet_edited.columns:
                                                    _gs_df_edit.loc[_gs_pallet_idx, _col] = (
                                                        _gs_pallet_edited[_col].values
                                                    )
                                            _save_res = _sb_upload(
                                                _sb_client,
                                                f"shipment/{_gs_edit_sel}",
                                                _sm_df_to_csv_bytes(_gs_df_edit),
                                                "text/csv",
                                            )
                                            if _save_res is True:
                                                st.session_state[_gs_df_key] = _gs_df_edit
                                                _show_success(
                                                    f"Saved {len(_gs_pallet_idx)} row(s) "
                                                    f"for Pallet {_gs_p_num}."
                                                )
                                                st.rerun()
                                            else:
                                                _show_error(f"Save failed: {_save_res}")

                            # ── Edit by Control # (Shipment) ──────────────────
                            with _gs_tab_ctrl:
                                _gs_ctrl_col = "Sample ID"
                                if _gs_ctrl_col not in _gs_cols:
                                    st.warning(f"Column '{_gs_ctrl_col}' not found in this file.")
                                else:
                                    _gs_ctrl_q = st.text_input(
                                        "Control number (Sample ID suffix, e.g. 002035)",
                                        key="_sm_gs_ctrl_q",
                                        placeholder="002035",
                                    )
                                    if st.button("Search", key="_sm_gs_ctrl_search_btn"):
                                        if not _gs_ctrl_q.strip():
                                            _show_warning("Enter a control number.")
                                        else:
                                            _gs_ctrl_mask = (
                                                _gs_df_edit[_gs_ctrl_col]
                                                .astype(str).str.strip()
                                                .str.endswith(_gs_ctrl_q.strip())
                                            )
                                            _gs_ctrl_found = _gs_df_edit.index[_gs_ctrl_mask].tolist()
                                            st.session_state["_sm_gs_ctrl_found"] = _gs_ctrl_found
                                            st.session_state["_sm_gs_ctrl_key"] = _gs_df_key
                                            if not _gs_ctrl_found:
                                                _show_info(
                                                    f"No rows where {_gs_ctrl_col} ends with "
                                                    f"'{_gs_ctrl_q.strip()}'."
                                                )
                                    _gs_ctrl_found = st.session_state.get("_sm_gs_ctrl_found", [])
                                    if st.session_state.get("_sm_gs_ctrl_key", "") != _gs_df_key:
                                        _gs_ctrl_found = []
                                    if _gs_ctrl_found:
                                        _gs_ctrl_valid = [i for i in _gs_ctrl_found if i in _gs_df_edit.index]
                                        st.markdown(f"**{len(_gs_ctrl_valid)} row(s) found**")
                                        _gs_ctrl_edited = st.data_editor(
                                            _gs_df_edit.loc[_gs_ctrl_valid].copy(),
                                            use_container_width=True,
                                            hide_index=False,
                                            key=f"_sm_gs_ctrl_editor_{_gs_df_key}",
                                            num_rows="fixed",
                                        )
                                        if st.button("Save Changes", key="_sm_gs_ctrl_save_btn"):
                                            for _col in _gs_cols:
                                                if _col in _gs_ctrl_edited.columns:
                                                    _gs_df_edit.loc[_gs_ctrl_valid, _col] = (
                                                        _gs_ctrl_edited[_col].values
                                                    )
                                            _save_res = _sb_upload(
                                                _sb_client,
                                                f"shipment/{_gs_edit_sel}",
                                                _sm_df_to_csv_bytes(_gs_df_edit),
                                                "text/csv",
                                            )
                                            if _save_res is True:
                                                st.session_state[_gs_df_key] = _gs_df_edit
                                                _show_success(f"Saved {len(_gs_ctrl_valid)} row(s).")
                                                st.rerun()
                                            else:
                                                _show_error(f"Save failed: {_save_res}")

                st.markdown("---")

    # -----------------------------------------------------------------
    # Donation Processing Dashboard (ported from the Pavlo HTML tool)
    # -----------------------------------------------------------------
    if nav_section == "Donation Processing":
        _subheader("Donation Processing Dashboard")

        _dp_c1, _dp_c2, _dp_c3, _dp_c4 = st.columns(4)
        with _dp_c1:
            _dp_vmt_raw = st.text_area(
                "VMT-NAT Barcodes",
                key="dp_vmt",
                height=150,
                placeholder="Paste or scan VMT-NAT barcodes here...\n(e.g. =C07032602987900)",
            )
        with _dp_c2:
            _dp_ser_raw = st.text_area(
                "SER Barcodes",
                key="dp_ser",
                height=150,
                placeholder="Paste or scan SER barcodes here...",
            )
        with _dp_c3:
            _dp_absc_raw = st.text_area(
                "AbSc Barcodes",
                key="dp_absc",
                height=150,
                placeholder="Paste or scan AbSc barcodes here...",
            )
        with _dp_c4:
            _dp_pcblut_raw = st.text_area(
                "Paste from PC-Blut",
                key="dp_pcblut",
                height=150,
                placeholder="Paste full PC-Blut pipe-separated rows here...",
            )

        _dp_rc1, _dp_rc2 = st.columns([1, 1])
        with _dp_rc1:
            _dp_role_label = st.radio(
                "Role",
                ["Processing Supervisor", "Processing Staff"],
                horizontal=True,
                key="dp_role",
            )
        _dp_role = (
            "supervisor" if _dp_role_label == "Processing Supervisor" else "staff"
        )
        with _dp_rc2:
            with st.expander("Freeze Tracker Settings"):
                _dp_tc1, _dp_tc2, _dp_tc3 = st.columns(3)
                _dp_max = _dp_tc1.number_input(
                    "Max Units", min_value=1, value=12, step=1, key="dp_trk_max"
                )
                _dp_win = _dp_tc2.number_input(
                    "Window (min)", min_value=1, value=30, step=1, key="dp_trk_win"
                )
                _dp_off = _dp_tc3.number_input(
                    "Offset (hrs)", value=-1, step=1, key="dp_trk_offset"
                )

        _dp_vmt = dp_parse_barcodes(_dp_vmt_raw)
        _dp_ser = dp_parse_barcodes(_dp_ser_raw)
        _dp_absc = dp_parse_barcodes(_dp_absc_raw)

        if not _dp_pcblut_raw.strip():
            _show_info(
                "Paste PC-Blut data above to generate the Excel output, "
                "freeze-tracker validation and compensation report."
            )
        else:
            _dp_res = dp_process_pc_blut(
                _dp_pcblut_raw, _dp_vmt, _dp_ser, _dp_absc, _dp_role
            )
            _dp_s5 = _dp_res["start5_values"]

            _dp_o1, _dp_o2 = st.columns([1.5, 1])
            with _dp_o1:
                st.markdown("**Excel Output** (tab-separated — paste into Excel)")
                if _dp_res["excel_rows"]:
                    st.code("\n".join(_dp_res["excel_rows"]), language=None)
                else:
                    _show_warning(
                        "No PC-Blut rows parsed. Check that the correct Role is "
                        "selected — Supervisor and Staff exports have different "
                        "column layouts."
                    )
            with _dp_o2:
                st.markdown("**Missing from PC-Blut**")
                if _dp_res["missing"]:
                    st.code("\n".join(_dp_res["missing"]), language=None)
                else:
                    st.success("All clear. All VMT-NAT numbers found in PC-Blut.")

                st.markdown("**START 5 Values**")
                if _dp_s5:
                    st.code(
                        "\n".join(_dp_s5)
                        + f"\n\n--- Total Units Extracted: {len(_dp_s5)} ---",
                        language=None,
                    )
                else:
                    _show_caption("No START 5 values extracted.")

            _subheader("Freeze Tracker Validation")
            _dp_trk = dp_validate_freeze_tracker(
                _dp_s5, int(_dp_max), int(_dp_win), int(_dp_off)
            )
            if _dp_trk["status"] == "ok":
                st.success("✔ All loading cycles are valid.")
            elif _dp_trk["status"] == "error":
                st.error("❌ Validation failed — see errors below.")
            else:
                st.info(_dp_trk["message"])
            for _dp_err in _dp_trk["cycle_errors"] + _dp_trk["format_errors"]:
                st.error(_dp_err)
            if _dp_trk["skipped"] and _dp_trk["rows"]:
                _show_caption(f"Skipped {_dp_trk['skipped']} zero-cycle line(s).")
            if _dp_trk["rows"]:
                _dp_tk1, _dp_tk2 = st.columns([1, 2])
                with _dp_tk1:
                    st.dataframe(
                        pd.DataFrame(
                            _dp_trk["rows"], columns=["Timestamp", "Units"]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.markdown(
                        f"**Total tracked units: {_dp_trk['total']}**"
                    )

            _subheader("Compensation Details")
            st.code(_dp_res["comp_text"], language=None)

            # ----------------------------------------------------------------
            # Commit the generated unit status rows to the CSV database, then
            # print Visual Inspection labels from the updated data.
            #
            # Deliberately placed *below* the freeze-tracker and compensation
            # output so every validation result is on screen before anyone
            # writes to the record.
            # ----------------------------------------------------------------
            _subheader("Commit to Unit Status Database")

            _dp_records = usdb.parse_dp_rows(_dp_res["excel_rows"])
            _dp_sig = hashlib.sha256(
                "\n".join(_dp_res["excel_rows"]).encode("utf-8")
            ).hexdigest()[:16]
            _dp_committed = st.session_state.get("dp_us_commit")

            # --- already committed this exact batch: show result + VI labels --
            if _dp_committed and _dp_committed.get("sig") == _dp_sig:
                _show_success(
                    f"Appended **{_dp_committed['rows']} row(s)** to "
                    f"**{_dp_committed['file']}** "
                    f"({_dp_committed['dates']})."
                )
                if _dp_committed.get("archive"):
                    _show_caption(
                        f"Snapshot before write: `{_dp_committed['archive']}`"
                    )
                else:
                    _show_warning(
                        "The pre-write snapshot could not be saved, so this "
                        "append is not reversible from storage."
                    )
                if st.button(
                    "Undo lock / commit a different batch",
                    key="dp_us_commit_reset",
                    help="Clears the committed flag for this screen only. It "
                         "does not roll back the database.",
                ):
                    st.session_state.pop("dp_us_commit", None)
                    st.rerun()

                st.markdown("---")
                if us_df is not None:
                    render_vi_labels(us_df, key_ns="dp")
                else:
                    _show_warning(
                        "Unit status data is not loaded in this session, so "
                        "labels cannot be generated. Select the database file "
                        "in the sidebar."
                    )

            # --- not yet committed: run the checks -------------------------
            elif not _dp_records:
                _show_warning("No rows to commit.")
            elif _sb_client is None:
                _show_error(
                    "Storage is not connected, so the unit status database "
                    "cannot be updated. "
                    + (_sb_error or "Check SUPABASE_URL / SUPABASE_KEY.")
                )
            else:
                _dp_batch_dates = sorted(
                    {
                        d
                        for d in (
                            _parse_donation_date(r["date"]) for r in _dp_records
                        )
                        if d is not None
                    }
                )
                if not _dp_batch_dates:
                    _show_error(
                        "None of the generated rows carry a readable donation "
                        "date, so the target year cannot be determined."
                    )
                else:
                    _dp_year = _dp_batch_dates[-1].year
                    _dp_names = _sb_list_files(_sb_client, _US_FOLDER)
                    _dp_pick = usdb.select_db_file(_dp_names, _dp_year)

                    # One file per year is the rule; anything else is a cleanup
                    # task for the user, not something to guess at.
                    for _yr, _files in sorted(_dp_pick.duplicates.items()):
                        _show_error(
                            f"{len(_files)} files found for {_yr} — there must "
                            f"be exactly one. Delete the extras in Storage "
                            f"Manager: {', '.join(_files)}"
                        )
                    if _dp_pick.non_conforming:
                        _show_warning(
                            "Ignoring file(s) that do not follow the "
                            "`Unit Status(UNIT STATUS <year>) ....csv` naming "
                            f"convention: {', '.join(_dp_pick.non_conforming)}"
                        )

                    _dp_target = _dp_pick.chosen
                    if _dp_target is None and _dp_year not in _dp_pick.duplicates:
                        _show_error(
                            f"No unit status file for {_dp_year} in "
                            f"`{_US_FOLDER}/`. Upload one named "
                            f"`Unit Status(UNIT STATUS {_dp_year}) ....csv` "
                            f"first."
                        )

                    if _dp_target:
                        _dp_raw = _sb_download(
                            _sb_client, f"{_US_FOLDER}/{_dp_target}"
                        )
                        if _dp_raw is None:
                            _show_error(f"Could not download {_dp_target}.")
                        else:
                            try:
                                _dp_shape = usdb.read_csv_shape(_dp_raw)
                            except ValueError as _e:
                                _dp_shape = None
                                _show_error(str(_e))

                            if _dp_shape is not None:
                                _show_caption(
                                    f"Target: **{_dp_target}** · "
                                    f"{len(_dp_shape.header)} columns"
                                )

                                # ---- column mapping (auto, with override) ----
                                _dp_map = usdb.resolve_headers(_dp_shape.header)
                                _dp_unres = usdb.unresolved_fields(_dp_map)
                                if _dp_unres:
                                    _show_warning(
                                        "Could not match "
                                        f"{len(_dp_unres)} field(s) to a column "
                                        "automatically — pick them below."
                                    )
                                with st.expander(
                                    "Column mapping"
                                    + (" — action needed" if _dp_unres else ""),
                                    expanded=bool(_dp_unres),
                                ):
                                    _dp_opts = ["— not written —"] + list(
                                        _dp_shape.header
                                    )
                                    for _f in usdb.FIELDS:
                                        _auto = _dp_map.get(_f)
                                        _idx = (
                                            _dp_opts.index(_auto)
                                            if _auto in _dp_opts
                                            else 0
                                        )
                                        _sel = st.selectbox(
                                            usdb.FIELD_LABELS[_f],
                                            _dp_opts,
                                            index=_idx,
                                            key=f"dp_map_{_f}_{_dp_target}",
                                        )
                                        _dp_map[_f] = (
                                            None
                                            if _sel == "— not written —"
                                            else _sel
                                        )

                                if not _dp_map.get("donation_num") or not _dp_map.get("date"):
                                    _show_error(
                                        "Donation # and Donation Date must both "
                                        "be mapped before committing."
                                    )
                                else:
                                    _dp_index = usdb.build_existing_index(
                                        _dp_shape, _dp_map, _parse_donation_date
                                    )
                                    _dp_pf = usdb.preflight(
                                        _dp_records,
                                        _dp_index["ids"],
                                        _dp_index["dates"],
                                        _dp_year,
                                        _parse_donation_date,
                                    )

                                    _dp_m1, _dp_m2, _dp_m3 = st.columns(3)
                                    _dp_m1.metric(
                                        "Rows to append", len(_dp_records)
                                    )
                                    _dp_m2.metric(
                                        "Rows in database", _dp_index["rows"]
                                    )
                                    _dp_m3.metric(
                                        "Latest date on file",
                                        _dp_pf.latest_existing.strftime("%d.%m.%Y")
                                        if _dp_pf.latest_existing
                                        else "—",
                                    )

                                    for _b in _dp_pf.blockers:
                                        _show_error(_b)
                                    for _w in _dp_pf.warnings:
                                        _show_warning(_w)

                                    _dp_ack = True
                                    if _dp_pf.requires_gap_ack:
                                        _dp_ack = st.checkbox(
                                            "I confirm there were no donations "
                                            "on the missing date(s) above "
                                            "(centre closed / no collection).",
                                            key=f"dp_gap_ack_{_dp_sig}",
                                        )

                                    if st.session_state.get("unit_status") is not None:
                                        _show_warning(
                                            "A locally uploaded unit status file "
                                            "is active in the sidebar and will "
                                            "take priority over the database "
                                            "after the append. Clear the "
                                            "uploader to see the updated data."
                                        )

                                    if st.button(
                                        f"Append {len(_dp_records)} row(s) to {_dp_target}",
                                        key=f"dp_us_commit_btn_{_dp_sig}",
                                        disabled=not (_dp_pf.ok and _dp_ack),
                                        type="primary",
                                    ):
                                        # 1. snapshot before the destructive upsert
                                        _dp_arch = usdb.archive_path(
                                            _US_FOLDER, _dp_target
                                        )
                                        _dp_arch_ok = (
                                            _sb_upload(
                                                _sb_client, _dp_arch, _dp_raw
                                            )
                                            is True
                                        )
                                        # 2. append and write back
                                        try:
                                            _dp_new = usdb.append_records(
                                                _dp_raw, _dp_records, _dp_map
                                            )
                                        except Exception as _e:
                                            _dp_new = None
                                            st.exception(_e)

                                        if _dp_new is not None:
                                            _dp_up = _sb_upload(
                                                _sb_client,
                                                f"{_US_FOLDER}/{_dp_target}",
                                                _dp_new,
                                                mime="text/csv",
                                            )
                                            if _dp_up is True:
                                                # 3. make the whole app see the
                                                #    updated database at once
                                                st.session_state["_sb_ls_unit_status"] = (
                                                    _sb_list_files(
                                                        _sb_client, _US_FOLDER
                                                    )
                                                )
                                                st.session_state["_sb_ln_unit_status"] = _dp_target
                                                st.session_state["_sb_ld_unit_status"] = _dp_new
                                                st.session_state["unit_status_sb_pick"] = _dp_target
                                                st.session_state["dp_us_commit"] = {
                                                    "sig": _dp_sig,
                                                    "file": _dp_target,
                                                    "rows": len(_dp_records),
                                                    "archive": _dp_arch
                                                    if _dp_arch_ok
                                                    else "",
                                                    "dates": " – ".join(
                                                        d.strftime("%d.%m.%Y")
                                                        for d in (
                                                            _dp_batch_dates[0],
                                                            _dp_batch_dates[-1],
                                                        )
                                                    )
                                                    if len(_dp_batch_dates) > 1
                                                    else _dp_batch_dates[0].strftime(
                                                        "%d.%m.%Y"
                                                    ),
                                                }
                                                st.rerun()
                                            else:
                                                _show_error(
                                                    f"Upload failed: {_dp_up}. "
                                                    f"The database was not "
                                                    f"changed."
                                                )

        _subheader("Rack Visualizations")
        st.markdown(_DP_RACK_CSS, unsafe_allow_html=True)
        _dp_r1, _dp_r2, _dp_r3 = st.columns(3)
        with _dp_r1:
            st.markdown(
                dp_build_rack_html("VMT-NAT", _dp_vmt, 12, 6),
                unsafe_allow_html=True,
            )
        with _dp_r2:
            st.markdown(
                dp_build_rack_html("SER", _dp_ser, 6, 6),
                unsafe_allow_html=True,
            )
        with _dp_r3:
            st.markdown(
                dp_build_rack_html("AbSc", _dp_absc, 6, 6),
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
