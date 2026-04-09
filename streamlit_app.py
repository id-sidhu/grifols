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

import datetime
import hashlib
import io
import json
import re
import secrets as _secrets_mod
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Feature flags
# Set a section to False to hide it from the navigation bar.
# Hidden sections remain accessible by adding ?unlock=<your_key> to the URL.
# ---------------------------------------------------------------------------
FEATURES: Dict[str, bool] = {
    "Pallet Report": True,
    "Manual racks/Unit Status Check": True,
    "Pre-built Rack Browser": True,
    "Visual Inspection Labels": True,
    "QC Report PDF Extractor": False,
}
# UNLOCK_KEY is loaded at runtime from Streamlit secrets (see .streamlit/secrets.toml
# locally, or the Streamlit Cloud "Secrets" panel in production).
# It is intentionally NOT hardcoded here so it is never visible in the public repo.


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
  .rack-wrap {{
    padding: 12px 12px 14px 12px;
    border: 1px solid rgba(0,0,0,0.10);
    border-radius: 12px;
    background: rgba(255,255,255,0.70);
    backdrop-filter: blur(4px);
    width: fit-content;
    max-width: 100%;
    overflow-x: auto;
  }}

  .rack-title {{
    font-weight: 700;
    font-size: 16px;
    margin: 0 0 8px 0;
  }}

  .rack-legend {{
    display: flex;
    gap: 14px;
    align-items: center;
    font-size: 12px;
    opacity: 0.85;
    margin-bottom: 10px;
    flex-wrap: wrap;
  }}

  .rack-date-range {{
    margin-left: auto;
    font-size: 12px;
    opacity: 0.75;
    white-space: nowrap;
    font-style: italic;
  }}

  .legend-item {{
    display: inline-flex;
    gap: 12px;
    align-items: center;
  }}

  .swatch {{
    width: 12px;
    height: 12px;
    border-radius: 4px;
    border: 1px solid rgba(0,0,0,0.15);
    display: inline-block;
  }}
  .swatch.present {{ background: #e6f4ea; }}
  .swatch.not-manifest {{ background: #ffd966; }}
  .swatch.samples-collected {{ background: #ffb3b3; }}
  .swatch.blank {{ background: #f3f4f6; }}
  .swatch.pallet-1 {{ background: #8ecae6; }}
  .swatch.pallet-2 {{ background: #c77dff; }}
  .swatch.pallet-3 {{ background: #76d9a3; }}
  .swatch.pallet-4 {{ background: #ff9f40; }}
  .swatch.pallet-5 {{ background: #ff6b9d; }}
  .swatch.pallet-6 {{ background: #ffdd44; }}

  /* TRUE 18-column grid (no spacer columns) */
  .rack-grid {{
    --cell-w: 40px;
    --gap: 6px;

    display: grid;
    grid-auto-rows: 34px;
    gap: var(--gap);
    grid-template-columns: repeat(18, var(--cell-w));
  }}

  .rack-cell {{
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.12);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.5px;
    user-select: none;
    transition: transform 0.06s ease;
    position: relative;
  }}

  .rack-cell.present {{
    background: #e6f4ea;
    color: rgba(0,0,0,0.72);
  }}

  .rack-cell.not-manifest {{
    background: #ffd966;
    color: rgba(0,0,0,0.80);
  }}

  .rack-cell.samples-collected {{
    background: #ffb3b3;
    color: rgba(0,0,0,0.80);
  }}

  .rack-cell.pallet-1 {{ background: #8ecae6; color: rgba(0,0,0,0.80); }}
  .rack-cell.pallet-2 {{ background: #c77dff; color: rgba(0,0,0,0.80); }}
  .rack-cell.pallet-3 {{ background: #76d9a3; color: rgba(0,0,0,0.80); }}
  .rack-cell.pallet-4 {{ background: #ff9f40; color: rgba(0,0,0,0.80); }}
  .rack-cell.pallet-5 {{ background: #ff6b9d; color: rgba(0,0,0,0.80); }}
  .rack-cell.pallet-6 {{ background: #ffdd44; color: rgba(0,0,0,0.80); }}

  .rack-cell.blank {{
    background: #f3f4f6;
    color: rgba(0,0,0,0.28);
    font-weight: 600;
  }}

  /* Solid strikethrough line for already-packed cells */
  .rack-cell.packed::after {{
    content: '';
    position: absolute;
    left: 4px;
    right: 4px;
    top: 50%;
    height: 2px;
    background: rgba(0, 0, 0, 0.65);
    transform: translateY(-50%);
    pointer-events: none;
    border-radius: 1px;
  }}

  /* Legend swatch for packed */
  .swatch.packed-swatch {{
    background: #d0d0d0;
    position: relative;
  }}
  .swatch.packed-swatch::after {{
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    top: 50%;
    height: 2px;
    background: rgba(0, 0, 0, 0.65);
    transform: translateY(-50%);
    border-radius: 1px;
  }}

  /* Visible separators after col 6 and col 12 (right edge of those cells) */
  .rack-grid > .rack-cell:nth-child(18n + 6),
  .rack-grid > .rack-cell:nth-child(18n + 12) {{
    box-shadow: inset -2px 0 0 blue;
  }}

  /* Optional: also mark the left edge of col 7 and col 13 to make a "double line" */
  .rack-grid > .rack-cell:nth-child(18n + 7),
  .rack-grid > .rack-cell:nth-child(18n + 13) {{
    box-shadow: inset 2px 0 0 blue;
  }}

  /* Hover: use filter instead of box-shadow so we don't overwrite separator shadows */
  .rack-cell:hover {{
    transform: translateY(-1px);
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.18));
  }}
</style>
"""
    return html



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


def main() -> None:
    """Run the Streamlit application.

    This function builds the Streamlit UI, handles user input, performs data
    processing using the helper functions defined above, and displays the
    resulting reports and data tables.  It covers both the pallet packing
    report and the unit status analysis.
    """
    st.set_page_config(page_title="Grifols Combined Reports", layout="wide")
    st.title("Grifols Combined Reports")
    st.write(
        "Upload your **Grifols shipment CSV** and optionally a **unit status CSV**. "
        "Then select the pallet number you wish to inspect and choose whether to "
        "display detailed lists of sample IDs.  If a unit status file is uploaded, "
        "you can also analyse donations on a specific prefix and control number(s)."
    )

    # File uploader for the shipment file.  This is required for the pallet report.
    shipment_file = st.file_uploader(
        label="Upload Grifols shipment file", type=["csv", "xlsx", "xls"], key="shipment"
    )
    # File uploader for the unit status file.  This is optional and can be
    # used to demonstrate the cleaning helper and the unit status analysis.
    unit_status_file = st.file_uploader(
        label="Upload unit status file (optional)", type=["csv", "xlsx", "xls"], key="unit_status"
    )

    def _read_upload(uploaded_file, dtype=None) -> Optional[pd.DataFrame]:
        """Read an uploaded file as a DataFrame, supporting CSV and Excel."""
        if uploaded_file is None:
            return None
        name = uploaded_file.name.lower()
        try:
            if name.endswith((".xlsx", ".xls")):
                kwargs = {"dtype": dtype} if dtype else {}
                return pd.read_excel(uploaded_file, **kwargs)
            else:
                kwargs = {"dtype": dtype} if dtype else {}
                return pd.read_csv(uploaded_file, **kwargs)
        except Exception as e:
            st.error(f"Failed to read {uploaded_file.name}: {e}")
            return None

    # Load the DataFrames
    gs_df = _read_upload(shipment_file)
    us_df = _read_upload(unit_status_file, dtype=str)

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
                    st.error(
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
                    st.error(
                        f"🚨 **DUPLICATE DONATION #s DETECTED in unit status file** — {len(_us_dupes)} duplicate(s):\n\n"
                        + ", ".join(str(x) for x in sorted(_us_dupes)),
                        icon="🚨",
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
        "Pre-built Rack Browser",
        "Visual Inspection Labels",
        "QC Report PDF Extractor",
    ]
    _visible_sections = (
        _all_sections if _is_unlocked
        else [s for s in _all_sections if FEATURES.get(s, True)]
    )

    # Sidebar: navigation + section-specific controls
    with st.sidebar:
        st.header("Navigation")
        if _is_unlocked:
            st.caption("🔓 All sections unlocked")
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
        st.subheader("Shipment Data Preview")
        st.write(
            "Below is a preview of the shipment DataFrame (first 5 rows). "
            "Ensure the columns include at least 'Sample ID', 'Comments' and 'Samples Packed?'."
        )
        st.dataframe(gs_df.head())
        # Optionally show a preview of the unit status file if uploaded
        if us_df is not None:
            st.subheader("Unit Status Data Preview")
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
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.exception(e)
        # Display the last generated pallet report if available
        if "pallet_report_text" in st.session_state:
            last_no = st.session_state.get("pallet_no_last", pallet_no)
            st.subheader(f"Pallet Report (Pallet {last_no})")

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
        st.info("Please upload a Grifols shipment file to begin the pallet report.")

    # If unit status CSV is loaded, provide inputs and allow control number checks
    if nav_section == "Manual racks/Unit Status Check" and us_df is not None:
        st.subheader("Manual racks/Unit Status Check")

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
                st.error("Please enter the control number(s).")
            else:
                try:
                    # Clean the unit status DataFrame once per check
                    cleaned_us_df = clean_unit_status(us_df)
                    # Determine if range or single value
                    if "," in suffix_input:
                        # Range mode: expect exactly two values
                        parts = [p.strip() for p in suffix_input.split(",") if p.strip()]
                        if len(parts) != 2:
                            st.error("Please enter exactly two control numbers separated by a comma.")
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
                                st.success(f"IDs in not_in_manifest between {start_id} and {end_id} (excluding removed):")
                                ids_df = pd.DataFrame({"Missing IDs": ids_between_sorted})
                                st.dataframe(ids_df)
                            else:
                                st.info("No IDs in not_in_manifest found within the specified range after excluding removed IDs.")
                            # Show rejected units whose samples were collected, within the range
                            sc_in_range = sorted(
                                [iid for iid in samples_collected_set if start_num <= extract_num(iid) <= end_num],
                                key=extract_num,
                            )
                            if sc_in_range:
                                st.warning(f"{len(sc_in_range)} rejected unit(s) in this range have samples collected and are NOT removed:")
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
                                st.error(f"Failed to generate rack visualisation: {e}")
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
                            st.warning(
                                f"{control_id} is **Rejected** but samples were collected."
                            )
                        elif control_id in to_remove_set:
                            st.success(f"{control_id} is classified as No Bleed, Sample Only, or Rejected.")
                        else:
                            st.error(f"{control_id} is not classified as No Bleed, Sample Only, or Rejected.")
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

    if nav_section == "Pre-built Rack Browser" and us_df is not None:
        st.subheader("Pre-built Rack Browser")
        st.write(
            "Automatically builds sequential racks of 216 samples from the unit "
            "status file. Search for any unit ID to jump straight to its rack. "
            "Enable **Edit positions** to rearrange samples or pull from the next rack."
        )

        pb_col1, pb_col2 = st.columns([3, 1])
        with pb_col1:
            pb_prefix = st.text_input(
                "Donation prefix", value="F26-", key="pb_prefix_input", max_chars=20
            ).strip()
        with pb_col2:
            st.write("")  # spacer to align button with input
            build_racks_btn = st.button("Build Racks", key="btn_build_racks")

        if build_racks_btn:
            try:
                cleaned_us_pb = clean_unit_status(us_df)
                all_pb_ids = (
                    cleaned_us_pb.get("Donation #", pd.Series(dtype=str))
                    .dropna().astype(str).str.strip()
                )

                def _pb_num(x: str) -> int:
                    m = re.match(rf"^{re.escape(pb_prefix)}(\d+)$", x, re.IGNORECASE)
                    return int(m.group(1)) if m else int(1e18)

                # Apply the same removal filter used by the range-based rack
                # so that SO, rejected and no-bleed IDs are excluded.
                try:
                    to_remove_pb, _ = process_unit_status_all(cleaned_us_pb, pb_prefix)
                except ValueError:
                    to_remove_pb = set()
                filtered_pb_ids: List[str] = sorted(
                    [
                        iid for iid in all_pb_ids
                        if iid.upper().startswith(pb_prefix.upper())
                        and iid not in to_remove_pb
                    ],
                    key=_pb_num,
                )
                # Group into racks of 216; pad the last rack with empty strings
                pb_racks: Dict[int, List[str]] = {}
                total_ids_pb = max(1, len(filtered_pb_ids))
                for i, chunk_start in enumerate(range(0, total_ids_pb, 216)):
                    chunk = filtered_pb_ids[chunk_start: chunk_start + 216]
                    pb_racks[i] = chunk + [""] * (216 - len(chunk))
                # Build reverse lookup: sample ID → rack index
                pb_id_to_rack: Dict[str, int] = {
                    sid: ridx
                    for ridx, rids in pb_racks.items()
                    for sid in rids if sid
                }
                st.session_state["pb_racks"] = pb_racks
                st.session_state["pb_id_to_rack"] = pb_id_to_rack
                st.session_state["pb_current_rack"] = 0
                st.session_state["pb_prefix"] = pb_prefix
                # Build date map: sample ID → datetime.date (from Donation Date column)
                _pb_date_map: Dict[str, datetime.date] = {}
                if (
                    "Donation Date" in cleaned_us_pb.columns
                    and "Donation #" in cleaned_us_pb.columns
                ):
                    for _, _dr in cleaned_us_pb.iterrows():
                        _sid = str(_dr.get("Donation #", "")).strip()
                        _d = _parse_donation_date(_dr.get("Donation Date"))
                        if _sid and _d is not None:
                            _pb_date_map[_sid] = _d
                st.session_state["pb_date_map"] = _pb_date_map
                total_filled_pb = sum(1 for rids in pb_racks.values() for s in rids if s)
                st.success(f"Built {len(pb_racks)} rack(s) with {total_filled_pb} samples.")
            except Exception as e:
                st.error(f"Failed to build racks: {e}")

        if "pb_racks" in st.session_state:
            pb_racks = st.session_state["pb_racks"]
            current_pb_idx = st.session_state.get("pb_current_rack", 0)

            # Search box
            pb_search = st.text_input(
                "Search by unit ID to jump to its rack",
                key="pb_search",
                placeholder="e.g. 002035 or F26-002035",
            ).strip()
            if pb_search:
                pb_prefix_val = st.session_state.get("pb_prefix", "F26-")
                if pb_search.upper().startswith(pb_prefix_val.upper()):
                    search_full_pb = pb_search
                elif pb_search.isdigit():
                    search_full_pb = f"{pb_prefix_val}{pb_search.zfill(6)}"
                else:
                    search_full_pb = pb_search
                found_rack_idx = st.session_state["pb_id_to_rack"].get(search_full_pb)
                if found_rack_idx is not None:
                    st.session_state["pb_current_rack"] = found_rack_idx
                    current_pb_idx = found_rack_idx
                    st.success(f"{search_full_pb} → Rack {found_rack_idx + 1}")
                else:
                    st.warning(f"{search_full_pb} not found in any pre-built rack.")

            # Filter by pallet — quickly find racks containing a specific pallet
            if gs_df is not None:
                _pallet_map_filter = build_pallet_map(gs_df)
                if _pallet_map_filter:
                    _available_pallets = sorted(set(_pallet_map_filter.values()))
                    _pallet_options = ["—"] + [f"Pallet {p}" for p in _available_pallets]
                    _selected_pallet = st.selectbox(
                        "Filter racks by pallet",
                        options=_pallet_options,
                        key="pb_pallet_filter",
                    )
                    if _selected_pallet != "—":
                        _pnum = int(_selected_pallet.split()[-1])
                        _pallet_sids = {
                            sid for sid, p in _pallet_map_filter.items() if p == _pnum
                        }
                        _matching_racks: Dict[int, int] = {}
                        _total_pallet_samples = 0
                        for ridx, rids in pb_racks.items():
                            cnt = sum(1 for sid in rids if sid and sid in _pallet_sids)
                            if cnt > 0:
                                _matching_racks[ridx] = cnt
                                _total_pallet_samples += cnt
                        if _matching_racks:
                            st.info(
                                f"**{_total_pallet_samples}** sample(s) from Pallet {_pnum} "
                                f"found across **{len(_matching_racks)}** rack(s)"
                            )
                            # Jump buttons in rows of up to 6
                            _rack_items = list(_matching_racks.items())
                            for _row_start in range(0, len(_rack_items), 6):
                                _row_chunk = _rack_items[_row_start : _row_start + 6]
                                _jump_cols = st.columns(len(_row_chunk))
                                for _ci, (ridx, cnt) in enumerate(_row_chunk):
                                    with _jump_cols[_ci]:
                                        if st.button(
                                            f"Rack {ridx + 1} ({cnt})",
                                            key=f"pb_pjump_{_pnum}_{ridx}",
                                        ):
                                            current_pb_idx = ridx
                                            st.session_state["pb_current_rack"] = ridx
                        else:
                            st.warning(
                                f"No samples from Pallet {_pnum} found in any rack."
                            )

            # Navigation row — avoid st.rerun() so toggle states survive navigation
            nav_c1, nav_c2, nav_c3 = st.columns([1, 5, 1])
            with nav_c1:
                prev_clicked = st.button(
                    "◀ Prev", key="pb_prev", disabled=(current_pb_idx == 0)
                )
            with nav_c2:
                _nav_info = st.empty()
            with nav_c3:
                next_clicked = st.button(
                    "Next ▶", key="pb_next",
                    disabled=(current_pb_idx >= len(pb_racks) - 1),
                )

            # Update index in-place; subsequent code uses the updated value
            if prev_clicked and current_pb_idx > 0:
                current_pb_idx -= 1
                st.session_state["pb_current_rack"] = current_pb_idx
            elif next_clicked and current_pb_idx < len(pb_racks) - 1:
                current_pb_idx += 1
                st.session_state["pb_current_rack"] = current_pb_idx

            filled_count = sum(1 for s in pb_racks[current_pb_idx] if s)
            _nav_info.markdown(
                f"**Rack {current_pb_idx + 1} of {len(pb_racks)}** "
                f"— {filled_count} / 216 positions filled"
            )

            rack_ids_current = list(pb_racks[current_pb_idx])

            # Compute date range string for rack legend
            _date_map_pb = st.session_state.get("pb_date_map", {})
            _pb_date_range_str: Optional[str] = None
            if _date_map_pb:
                _rack_dates = [
                    _date_map_pb[sid]
                    for sid in rack_ids_current
                    if sid and sid in _date_map_pb
                ]
                if _rack_dates:
                    _dmin = min(_rack_dates)
                    _dmax = max(_rack_dates)
                    _pb_date_range_str = (
                        _dmin.strftime("%d.%m.%Y")
                        if _dmin == _dmax
                        else f"{_dmin.strftime('%d.%m.%Y')} – {_dmax.strftime('%d.%m.%Y')}"
                    )

            # Edit mode toggle
            pb_edit_mode = st.checkbox("Edit positions", key="pb_edit_mode", value=False)

            if pb_edit_mode:
                st.caption(
                    "Edit Sample IDs directly in the table. Clear a cell to leave "
                    "that position empty. Use **Apply Changes** to save edits, or "
                    "**Re-pack from next rack** to pull samples forward and fill gaps."
                )
                edit_df_pb = pd.DataFrame({
                    "Position": list(range(1, 217)),
                    "Sample ID": rack_ids_current,
                })
                edited_df_pb = st.data_editor(
                    edit_df_pb,
                    use_container_width=True,
                    hide_index=True,
                    key=f"pb_editor_{current_pb_idx}",
                    column_config={
                        "Position": st.column_config.NumberColumn(
                            "Pos", disabled=True, width="small"
                        ),
                        "Sample ID": st.column_config.TextColumn(
                            "Sample ID", width="medium"
                        ),
                    },
                    num_rows="fixed",
                )

                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if st.button("Apply Changes", key="pb_apply"):
                        new_ids_pb = [
                            str(v).strip() if pd.notna(v) and str(v).strip() else ""
                            for v in edited_df_pb["Sample ID"]
                        ]
                        new_ids_pb = (new_ids_pb + [""] * 216)[:216]
                        st.session_state["pb_racks"][current_pb_idx] = new_ids_pb
                        st.session_state["pb_id_to_rack"] = {
                            sid: ridx
                            for ridx, rids in st.session_state["pb_racks"].items()
                            for sid in rids if sid
                        }
                        st.success("Changes saved.")
                        st.rerun()
                with btn_c2:
                    has_next_rack = current_pb_idx + 1 < len(pb_racks)
                    if st.button(
                        "Re-pack from next rack", key="pb_repack",
                        disabled=not has_next_rack
                    ):
                        cur_ids_edit = [
                            str(v).strip() if pd.notna(v) and str(v).strip() else ""
                            for v in edited_df_pb["Sample ID"]
                        ]
                        nxt_ids_edit = list(pb_racks[current_pb_idx + 1])
                        combined_pb = (
                            [s for s in cur_ids_edit if s]
                            + [s for s in nxt_ids_edit if s]
                        )
                        st.session_state["pb_racks"][current_pb_idx] = (
                            combined_pb[:216] + [""] * max(0, 216 - len(combined_pb))
                        )[:216]
                        st.session_state["pb_racks"][current_pb_idx + 1] = (
                            combined_pb[216:] + [""] * max(0, 216 - len(combined_pb[216:]))
                        )[:216]
                        st.session_state["pb_id_to_rack"] = {
                            sid: ridx
                            for ridx, rids in st.session_state["pb_racks"].items()
                            for sid in rids if sid
                        }
                        st.success("Re-packed successfully.")
                        st.rerun()

                # Trim rack size — limit this rack to N samples, overflow → next rack
                st.markdown("---")
                st.markdown("**Set rack size**")
                st.caption(
                    "Limit this rack to a specific number of samples. "
                    "Any samples beyond that count are moved to the start of the next rack."
                )
                _cur_filled = sum(
                    1 for v in edited_df_pb["Sample ID"]
                    if pd.notna(v) and str(v).strip()
                )
                trim_c1, trim_c2 = st.columns([3, 1])
                with trim_c1:
                    trim_size = st.number_input(
                        "Max samples in this rack",
                        min_value=1,
                        max_value=216,
                        value=min(_cur_filled, 216),
                        step=1,
                        key="pb_trim_size",
                    )
                with trim_c2:
                    st.write("")
                    _has_next_for_trim = current_pb_idx + 1 < len(pb_racks)
                    apply_trim = st.button(
                        "Apply trim",
                        key="pb_apply_trim",
                        disabled=not _has_next_for_trim,
                        help="Requires a next rack to receive the overflow samples.",
                    )
                if apply_trim:
                    cur_ids_trim = [
                        str(v).strip() if pd.notna(v) and str(v).strip() else ""
                        for v in edited_df_pb["Sample ID"]
                    ]
                    filled_trim = [s for s in cur_ids_trim if s]
                    kept_trim = filled_trim[:trim_size]
                    overflow_trim = filled_trim[trim_size:]
                    st.session_state["pb_racks"][current_pb_idx] = (
                        kept_trim + [""] * (216 - len(kept_trim))
                    )[:216]
                    if overflow_trim:
                        nxt_trim = list(st.session_state["pb_racks"][current_pb_idx + 1])
                        nxt_filled_trim = [s for s in nxt_trim if s]
                        merged_trim = overflow_trim + nxt_filled_trim
                        st.session_state["pb_racks"][current_pb_idx + 1] = (
                            merged_trim + [""] * max(0, 216 - len(merged_trim))
                        )[:216]
                    st.session_state["pb_id_to_rack"] = {
                        sid: ridx
                        for ridx, rids in st.session_state["pb_racks"].items()
                        for sid in rids if sid
                    }
                    st.success(
                        f"Rack trimmed to {len(kept_trim)} samples. "
                        f"{len(overflow_trim)} sample(s) moved to Rack {current_pb_idx + 2}."
                    )
                    st.rerun()

                # Visualization uses the live editor state
                display_ids_pb = [
                    str(v).strip() if pd.notna(v) and str(v).strip() else ""
                    for v in edited_df_pb["Sample ID"]
                ]
            else:
                display_ids_pb = rack_ids_current

            # Build visualization (reuse pallet/packed info from shipment file if loaded)
            not_manifest_set_pb = set(st.session_state.get("not_in_manifest", []))
            pallet_map_pb = build_pallet_map(gs_df) if gs_df is not None else {}
            packed_set_pb: Set[str] = set()
            if gs_df is not None and "Samples Packed?" in gs_df.columns and "Sample ID" in gs_df.columns:
                _pbpm = gs_df["Samples Packed?"].fillna("").astype(str).str.strip().ne("")
                packed_set_pb = set(
                    gs_df.loc[_pbpm, "Sample ID"].dropna().astype(str).str.strip()
                )

            # Hide/strikethrough toggles (mirrors the range-based rack controls)
            pb_hide_packed = st.toggle(
                "Hide packed samples", value=False, key="pb_hide_packed"
            )
            if pb_hide_packed:
                display_ids_pb = [
                    "" if sid in packed_set_pb else sid for sid in display_ids_pb
                ]
                packed_set_pb_display: Set[str] = set()
            else:
                pb_show_strikethrough = st.toggle(
                    "Show strikethrough on packed samples",
                    value=True,
                    key="pb_show_strikethrough",
                )
                packed_set_pb_display = packed_set_pb if pb_show_strikethrough else set()

            rack_html_pb = build_rack_html(
                display_ids_pb,
                not_manifest_set_pb,
                pallet_map=pallet_map_pb,
                packed_set=packed_set_pb_display,
                digits_to_show=3,
                fill_value="–",
                title=f"Rack {current_pb_idx + 1} of {len(pb_racks)} (Pre-built)",
                date_range_str=_pb_date_range_str,
            )
            st.markdown(rack_html_pb, unsafe_allow_html=True)

    if nav_section == "Visual Inspection Labels" and us_df is not None:
        st.subheader("Visual Inspection Labels")
        st.write(
            "Groups Quarantine units into batches of N and generates a printable "
            "PDF.  Each page has two different labels (top and bottom half). "
            "Rejected and SO units are included in the printed range but do **not** "
            "count toward the group size."
        )

        vi_c1, vi_c2, vi_c3 = st.columns([2, 3, 1])
        with vi_c1:
            vi_prefix = st.text_input(
                "Donation prefix", value="F26-", key="vi_prefix", max_chars=20
            ).strip()
        with vi_c3:
            vi_group_size = st.number_input(
                "Group size", min_value=1, max_value=216, value=12, step=1,
                key="vi_group_size",
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
                st.info(
                    f"Auto-detected start: **{_vi_auto_start}** "
                    f"— last complete group covered donations up to {_last_updated}"
                )
            with _reset_col:
                st.write("")
                if st.button("Reset", key="vi_reset_state", help="Clear saved state for this prefix"):
                    _vi_gist_state.pop(vi_prefix, None)
                    _vi_gist_save(_vi_gist_state)
                    st.session_state["vi_gist_state"] = _vi_gist_state
                    st.rerun()
        elif not _vi_gist_configured:
            st.caption(
                "No saved state found for this prefix. "
                "After generating a PDF the start position will be saved automatically."
            )

        with vi_c2:
            vi_start_id = st.text_input(
                "Start from donation ID",
                value=_vi_effective_start,
                key="vi_start_id",
                placeholder="e.g. F26-012401",
            ).strip()

        if st.button("Generate Visual Inspection Labels PDF", key="btn_vi_labels"):
            if not vi_start_id:
                st.error("Please enter a start donation ID.")
            else:
                try:
                    vi_groups = build_vi_label_groups(
                        us_df, vi_prefix, vi_start_id, group_size=int(vi_group_size)
                    )
                    if not vi_groups:
                        st.warning("No groups found from the specified start ID.")
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
                                st.success(
                                    f"Generated {len(vi_groups)} label(s). "
                                    f"Next session will auto-start from the continuation point."
                                )
                            else:
                                st.warning(
                                    f"Generated {len(vi_groups)} label(s) but could not save state "
                                    f"(check GITHUB_TOKEN and GIST_ID in Streamlit secrets)."
                                )
                        else:
                            st.success(f"Generated {len(vi_groups)} label(s).")
                        st.download_button(
                            label=f"⬇ Download PDF ({len(vi_groups)} label pages)",
                            data=vi_pdf,
                            file_name="vi_labels.pdf",
                            mime="application/pdf",
                            key="vi_pdf_dl",
                        )
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
                        st.table(pd.DataFrame(_prev))
                except ImportError as _e:
                    st.error(str(_e))
                except ValueError as _e:
                    st.error(str(_e))
                except Exception as _e:
                    st.exception(_e)

    if nav_section == "QC Report PDF Extractor":
        st.subheader("QC Report PDF Extractor")
        st.write(
            "Upload one or more Grifols QC Report PDFs to extract **Unit ID** and "
            "**Don. date** for each donation on the report.  Results from all files "
            "are combined and deduplicated."
        )

        qc_pdf_files = st.file_uploader(
            "Upload QC Report PDF(s)", type=["pdf"], key="qc_pdf",
            accept_multiple_files=True,
        )

        if qc_pdf_files:
            debug_mode = st.checkbox("Show debug info", value=False, key="qc_debug")

            if st.button("Extract Data", key="btn_extract_qc_pdf"):
                try:
                    import pdfplumber as _plumber
                    all_frames: List[pd.DataFrame] = []
                    for qc_pdf_file in qc_pdf_files:
                        qc_pdf_file.seek(0)

                        if debug_mode:
                            st.markdown(f"#### Debug: {qc_pdf_file.name}")
                            # ---- raw diagnosis ----
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
                            st.warning(f"No records found in **{qc_pdf_file.name}**.")
                        else:
                            all_frames.append(_df)

                    if not all_frames:
                        st.warning(
                            "No records were found in any of the uploaded PDFs. "
                            "Enable **Show debug info** and click Extract Data again "
                            "to see what pdfplumber is reading from these files."
                        )
                        st.session_state.pop("qc_extracted_df", None)
                    else:
                        qc_df = pd.concat(all_frames, ignore_index=True).drop_duplicates()
                        st.session_state["qc_extracted_df"] = qc_df
                        if len(qc_pdf_files) > 1:
                            st.info(
                                f"Combined {len(qc_pdf_files)} file(s): "
                                f"{sum(len(f) for f in all_frames)} total rows → "
                                f"{len(qc_df)} after deduplication."
                            )
                except ImportError as _ie:
                    st.error(str(_ie))
                except Exception as _pe:
                    st.exception(_pe)

        if "qc_extracted_df" in st.session_state:
            qc_result = st.session_state["qc_extracted_df"]
            st.success(f"Extracted {len(qc_result)} donation record(s).")
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
            # Release comparison against unit status
            # ------------------------------------------------------------------
            st.markdown("---")
            st.subheader("Release Comparison by Date")
            if us_df is None:
                st.info(
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
                        st.error(str(_ve))
                    except Exception as _ce:
                        st.exception(_ce)

                if "qc_comparison_df" in st.session_state:
                    cmp = st.session_state["qc_comparison_df"]
                    cmp_detail = st.session_state.get("qc_comparison_detail", {})
                    has_shipment_cols = "Packed" in cmp.columns and "Still to pack" in cmp.columns

                    # Colour rows: green = all released, amber = partial, red = none
                    def _row_style(row):
                        if row["Valid units (US)"] == 0:
                            colour = "#fff3cd"  # amber – date not in US file
                        elif row["Released (QC)"] >= row["Valid units (US)"]:
                            colour = "#d4edda"  # green – fully released
                        else:
                            colour = "#fff3cd"  # amber – partially released
                        return [f"background-color: {colour}"] * len(row)

                    st.dataframe(
                        cmp.style.apply(_row_style, axis=1),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Totals
                    total_released = int(cmp["Released (QC)"].sum())
                    total_valid = int(cmp["Valid units (US)"].sum())
                    total_pending = int(cmp["Pending"].sum())

                    if has_shipment_cols:
                        total_packed = int(cmp["Packed"].sum())
                        total_still_to_pack = int(cmp["Still to pack"].sum())
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Total Released (QC)", total_released)
                        mc2.metric("Valid units (US)", total_valid)
                        mc3.metric("Already packed", total_packed)
                        mc4.metric("Still to pack", total_still_to_pack)
                    else:
                        tc1, tc2, tc3 = st.columns(3)
                        tc1.metric("Total Released", total_released)
                        tc2.metric("Total Valid (US)", total_valid)
                        tc3.metric("Still Pending", total_pending)

                    # Per-date expandable ID lists (only when shipment file loaded)
                    if has_shipment_cols and cmp_detail:
                        st.markdown("#### Per-date packing detail")
                        for _, _cmp_row in cmp.iterrows():
                            _ds = _cmp_row["Don. date"]
                            _d = cmp_detail.get(_ds, {})
                            _packed_list = _d.get("packed", [])
                            _still_list = _d.get("still_to_pack", [])
                            _label = (
                                f"{_ds} — "
                                f"{len(_packed_list)} packed, "
                                f"{len(_still_list)} still to pack"
                            )
                            with st.expander(_label):
                                _exp_c1, _exp_c2 = st.columns(2)
                                with _exp_c1:
                                    st.markdown("**Already packed**")
                                    if _packed_list:
                                        st.dataframe(
                                            pd.DataFrame({"Unit ID": _packed_list}),
                                            hide_index=True,
                                            use_container_width=True,
                                        )
                                    else:
                                        st.caption("None")
                                with _exp_c2:
                                    st.markdown("**Still to pack**")
                                    if _still_list:
                                        st.dataframe(
                                            pd.DataFrame({"Unit ID": _still_list}),
                                            hide_index=True,
                                            use_container_width=True,
                                        )
                                    else:
                                        st.caption("All packed!")

                    cmp_csv = cmp.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download comparison as CSV",
                        data=cmp_csv,
                        file_name="qc_release_comparison.csv",
                        mime="text/csv",
                        key="qc_cmp_csv_dl",
                    )

    if nav_section in ["Manual racks/Unit Status Check", "Pre-built Rack Browser", "Visual Inspection Labels"] and us_df is None:
        st.info("Please upload a unit status file to use this section.")


if __name__ == "__main__":
    main()
