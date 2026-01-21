"""
A Streamlit application built from an existing console-based script that processes
Grifols shipment and unit status CSV files.  The original script used
command‑line prompts to ask for a pallet number and verbosity setting and
printed a report to stdout.  This Streamlit version preserves all of that
functionality in a web interface without losing any of the original logic.

Usage
-----
To run this app locally, execute the following command from a terminal in
the directory containing this file:

```
streamlit run grifols_streamlit.py
```

The application expects you to upload a `grifols_shipment.csv` file (and
optionally a `unit_status.csv` file if you want to use the cleaning helpers).
You will then be prompted for the pallet number you are interested in and
whether to see a verbose listing of all F25 and F26 sample IDs.

Functional equivalence
----------------------
This app reproduces the core logic of the original Python script provided by
the user:

* Cleaning rows from a unit status DataFrame by filtering out specific
  status patterns.
* Removing already packed samples from the shipment DataFrame.
* Locating the start and end markers for a given pallet within the
  `Comments` column using case‑insensitive regular expressions.
* Extracting the sub‑DataFrame corresponding to the pallet between those
  markers.
* Generating a report that counts F25 and F26 sample IDs, shows the total
  number of samples, the first and last sample IDs in the pallet, and
  optionally displays tables of all F25 and F26 IDs.

No functionality has been removed; instead, the interactive prompts have
been replaced with Streamlit widgets and the console print statements have
been converted into Streamlit UI elements.
"""

import re
from typing import List, Tuple, Optional

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

    The `gs_df` DataFrame is expected to have a column named ``"Samples Packed?"``
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


def main() -> None:
    """Run the Streamlit application.

    This function builds the Streamlit UI, handles user input, performs data
    processing using the helper functions defined above, and displays the
    resulting report and data tables.  It is executed when the script is run
    with ``streamlit run``.
    """
    st.set_page_config(page_title="Grifols Pallet Report", layout="wide")
    st.title("Grifols Pallet Packing Report")
    st.write(
        "Upload your **Grifols shipment CSV** and optionally a **unit status CSV**. "
        "Then select the pallet number you wish to inspect and choose whether to "
        "display detailed lists of sample IDs."
    )

    # File uploader for the shipment CSV.  This is required.
    shipment_file = st.file_uploader(
        label="Upload grifols_shipment.csv", type=["csv"], key="shipment"
    )
    # File uploader for the unit status CSV.  This is optional and can be
    # used to demonstrate the cleaning helper; it's not needed for the core
    # pallet report functionality.
    unit_status_file = st.file_uploader(
        label="Upload unit_status.csv (optional)", type=["csv"], key="unit_status"
    )

    # Hold the loaded DataFrames in session state so that the app does not
    # reload them on every widget interaction.
    if shipment_file:
        try:
            gs_df = pd.read_csv(shipment_file)
        except Exception as e:
            st.error(f"Failed to read shipment CSV: {e}")
            return
    else:
        gs_df = None
    if unit_status_file:
        try:
            us_df = pd.read_csv(unit_status_file)
        except Exception as e:
            st.error(f"Failed to read unit status CSV: {e}")
            return
    else:
        us_df = None

    # Sidebar for input controls
    with st.sidebar:
        st.header("Inputs")
        pallet_no = st.number_input(
            "Pallet number", min_value=1, step=1, value=1, format="%d"
        )
        verbose = st.checkbox("Show full F25/F26 lists", value=False)

    # If a shipment DataFrame has been loaded, allow the user to run the report.
    if gs_df is not None:
        st.subheader("Loaded Data Preview")
        st.write(
            "Below is a preview of the shipment DataFrame (first 5 rows). "
            "Ensure the columns include at least 'Sample ID', 'Comments' and 'Samples Packed?'."
        )
        st.dataframe(gs_df.head())
        # Optionally show a preview of the unit status file if uploaded
        if us_df is not None:
            st.subheader("Unit Status Data Preview")
            st.write("First 5 rows of the unit status DataFrame after cleaning:")
            cleaned_us_df = clean_unit_status(us_df)
            st.dataframe(cleaned_us_df.head())
        # Process the shipment DataFrame when the user clicks the button
        if st.button("Generate Pallet Report"):
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
                # Display the report
                st.subheader("Pallet Report")
                st.text(report_text)
                # Show tables if verbose
                if verbose:
                    st.markdown("### F25 Sample IDs")
                    if f25_ids:
                        f25_df = pd.DataFrame({"Sample ID": f25_ids}) # "Row#": range(1, len(f25_ids) + 1), 
                        st.dataframe(f25_df)
                    else:
                        st.write("No F25 IDs found.")
                    st.markdown("### F26 Sample IDs")
                    if f26_ids:
                        f26_df = pd.DataFrame({"Sample ID": f26_ids}) # "Row#": range(1, len(f26_ids) + 1), 
                        st.dataframe(f26_df)
                    else:
                        st.write("No F26 IDs found.")
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.exception(e)
    else:
        st.info("Please upload a grifols_shipment.csv file to begin.")


if __name__ == "__main__":
    main()
