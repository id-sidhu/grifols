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

2.  In the unit status check section, the user may provide **two** control
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
import re
from typing import List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import streamlit as st


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
    # Drop rows where the Status contains "ejec" (case insensitive)
    mask_ejec = us_df["Status"].astype(str).str.contains("ejec", case=False, na=False)
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
        gs_df["Samples Packed?"].astype(str).str.strip().str.fullmatch(r"(y|yes)", case=False, na=False)
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

    # Remove rows with missing/blank Donor Status
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

    nums = df["donation_num"].dropna().astype(int).sort_values()
    missing_nums: List[int] = []
    if not nums.empty:
        missing_nums = sorted(set(range(nums.min(), nums.max() + 1)) - set(nums))
    no_bleeds = {f"{prefix}{n:06d}" for n in missing_nums}

    # Normalize status
    status_raw = df["Status"].fillna("").astype(str).str.strip()

    def normalize_status(s: str) -> str:
        if s == "Quarantine":
            return "Quarantine"
        if s == "SO (16 week)":
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

    rejected = set(df.loc[df["Type"] == "rejected", "Donation #"].dropna().astype(str).tolist())
    sample_only = set(df.loc[df["Type"] == "sample_only", "Donation #"].dropna().astype(str).tolist())

    return set().union(no_bleeds, rejected, sample_only)


# -----------------------------------------------------------------------------
# Rack visualisation helper
def build_rack_html(
    valid_ids: List[str],
    not_manifest_set: Set[str],
    digits_to_show: int = 3,
    fill_value: str = "",
    title: str = "Rack Visualization (last three digits)",
) -> str:
    """
    18×12 rack using a true 18-column CSS Grid (no spacer columns, no inserted blank IDs).
    Visible separators after columns 6 and 12 are drawn INSIDE the boundary cells
    (inset shadows), so they remain clearly visible regardless of background/theme.
    """
    total_positions = 216
    ids_padded = valid_ids[:total_positions] + [""] * max(0, total_positions - len(valid_ids))

    def display_text(sample_id: str) -> str:
        if not sample_id:
            return fill_value
        return sample_id[-digits_to_show:]

    cells_html: List[str] = []
    for sample_id in ids_padded:
        is_blank = not bool(sample_id)
        is_not_manifest = (sample_id in not_manifest_set) if sample_id else False

        classes = ["rack-cell"]
        if is_blank:
            classes.append("blank")
        elif is_not_manifest:
            classes.append("not-manifest")
        else:
            classes.append("present")

        tooltip = sample_id if sample_id else "Empty"
        cells_html.append(
            f'<div class="{" ".join(classes)}" title="{tooltip}">{display_text(sample_id)}</div>'
        )

    cells_joined = "\n".join(cells_html)

    html = f"""
<div class="rack-wrap">
  <div class="rack-title">{title}</div>

  <div class="rack-legend">
    <span class="legend-item"><span class="swatch present"></span> In unit status</span>
    <span class="legend-item"><span class="swatch not-manifest"></span> Not in manifest</span>
    <span class="legend-item"><span class="swatch blank"></span> Empty</span>
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
  .swatch.blank {{ background: #f3f4f6; }}

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

  .rack-cell.blank {{
    background: #f3f4f6;
    color: rgba(0,0,0,0.28);
    font-weight: 600;
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

    # File uploader for the shipment CSV.  This is required for the pallet report.
    shipment_file = st.file_uploader(
        label="Upload grifols_shipment.csv", type=["csv"], key="shipment"
    )
    # File uploader for the unit status CSV.  This is optional and can be
    # used to demonstrate the cleaning helper and the unit status analysis.
    unit_status_file = st.file_uploader(
        label="Upload unit_status.csv (optional)", type=["csv"], key="unit_status"
    )

    # Load the DataFrames only once per session
    if shipment_file:
        try:
            gs_df = pd.read_csv(shipment_file)
        except Exception as e:
            st.error(f"Failed to read shipment CSV: {e}")
            gs_df = None
    else:
        gs_df = None
    if unit_status_file:
        try:
            us_df = pd.read_csv(unit_status_file, dtype=str)
        except Exception as e:
            st.error(f"Failed to read unit status CSV: {e}")
            us_df = None
    else:
        us_df = None

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

    # Sidebar for pallet report input controls
    with st.sidebar:
        st.header("Pallet Report Inputs")
        pallet_no = st.number_input(
            "Pallet number", min_value=1, step=1, value=1, format="%d"
        )
        verbose = st.checkbox("Show full F25/F26 lists", value=False)

    # Display shipment DataFrame preview and generate pallet report
    if gs_df is not None:
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
    else:
        st.info("Please upload a grifols_shipment.csv file to begin the pallet report.")

    # If unit status CSV is loaded, provide inputs and allow control number checks
    if us_df is not None:
        st.markdown("---")
        st.subheader("Unit Status Check")

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
                            to_remove_set = process_unit_status_all(cleaned_us_df, prefix_input)
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

                            # ------------------------------------------------------------------
                            # Additional functionality: always show Rack visualization
                            try:
                                # Recompute cleaned unit status and removal set
                                cleaned_us_full = clean_unit_status(us_df)
                                to_remove_set_full = process_unit_status_all(cleaned_us_full, prefix_input)
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
                                # Construct the rack HTML and display it.  If fewer than 216 IDs
                                # are present, the remainder of the rack will be filled with '-'.
                                rack_html = build_rack_html(
                                    valid_ids_full,
                                    not_manifest_set_full,
                                    digits_to_show=3,
                                    fill_value="-",
                                )
                                rack_html = build_rack_html(
                                    valid_ids_full,
                                    not_manifest_set_full,
                                    digits_to_show=3,
                                    fill_value="–",
                                )
                                st.markdown(rack_html, unsafe_allow_html=True)

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
                        to_remove_set = process_unit_status_all(cleaned_us_df, prefix_input)
                        if control_id in to_remove_set:
                            st.success(f"{control_id} is classified as No Bleed, Sample Only, or Rejected.")
                        else:
                            st.error(f"{control_id} is classified as No Bleed, Sample Only, or Rejected.”)
                except Exception as e:
                    st.exception(e)


if __name__ == "__main__":
    main()

