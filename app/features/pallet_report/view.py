"""Pallet Report section: shipment preview, pallet report, box numbers."""

import datetime
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from app.features.pallet_report.logic import (
    clean_grifols_shipment,
    generate_report_text,
    get_pallet_between_markers,
    remove_packed,
)
from app.shared.dates import _parse_donation_date
from app.shared.ui import _show_caption, _show_error, _show_info, _subheader, _show_success
from app.shared.unit_status import clean_unit_status


def sidebar_inputs() -> Tuple[int, bool]:
    """Draw this section's sidebar controls and return their values."""
    st.header("Pallet Report Inputs")
    pallet_no = st.number_input(
        "Pallet number", min_value=1, step=1, value=1, format="%d"
    )
    verbose = st.checkbox("Show full F25/F26 lists", value=False)
    return int(pallet_no), verbose


def render(gs_df, us_df, pallet_no: int, verbose: bool) -> None:
    """Render the Pallet Report section."""
    if gs_df is not None:
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
    else:
        _show_info("Please upload a Grifols shipment file to begin the pallet report.")

    # ── Box Number Generator — always visible in Pallet Report ───────────────
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
