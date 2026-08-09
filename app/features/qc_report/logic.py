"""Parsing QC report PDFs and comparing them against release/packing data."""

import re
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import streamlit as st

from app.shared.dates import _parse_donation_date


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
