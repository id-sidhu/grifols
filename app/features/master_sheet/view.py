"""Master Sheet section: fills the daily inventory table."""

import datetime
from typing import List, Optional, Set

import pandas as pd
import streamlit as st

from app.features.master_sheet.logic import parse_master_sheet
from app.features.qc_report.logic import parse_qc_report_pdf
from app.shared.dates import _parse_donation_date
from app.shared.excel import read_upload
from app.shared.storage import _sb_file_widget, _sb_upload
from app.shared.ui import (
    _show_caption,
    _show_error,
    _show_info,
    _show_success,
    _show_warning,
    _subheader,
)


def render(gs_df, us_df, sb_client) -> None:
    """Render the Master Sheet section."""
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
            sb_client,
            save_mime="text/csv",
        )
    with _ms_c2:
        _us_2025_file = _sb_file_widget(
            "2025 Unit Status (for Donor ID columns e & f)",
            "unit-status-2025",
            "ms_us_2025_file",
            ["csv", "xlsx", "xls"],
            sb_client,
        )

    _ms_qc_pdfs = _sb_file_widget(
        "QC Report PDF(s) — required for column (c). "
        "Alternatively, extract them in the QC Report PDF Extractor section first.",
        "qc-reports",
        "ms_qc_pdf",
        ["pdf"],
        sb_client,
        accept_multiple=True,
        save_mime="application/pdf",
    )

    # Read 2025 unit status
    _us_df_2025: Optional[pd.DataFrame] = read_upload(_us_2025_file, dtype=str, sheet_strategy="unit_status")
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
                            if sb_client:
                                if st.button(
                                    "Save updated Master Sheet to storage",
                                    key="ms_sb_save_updated",
                                    help="Overwrites master-sheet/master_sheet.csv in Supabase",
                                ):
                                    _ms_save_result = _sb_upload(
                                        sb_client,
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
