"""Composition root: page setup, file loading, navigation and dispatch.

Each section lives in its own package under :mod:`app.features`; this
module only wires shared state (the two uploaded DataFrames and the
Supabase client) into whichever section the user selected.
"""

import io

import numpy as np
import pandas as pd
import streamlit as st

from app.config import visible_sections
from app.features import donation_processing, manual_racks, master_sheet
from app.features import pallet_report, qc_report, storage_manager, vi_labels
from app.shared.excel import read_upload
from app.shared.storage import (
    _get_supabase_client,
    _get_supabase_error,
    _sb_file_widget,
)
from app.shared.ui import (
    VERBOSE_DEFAULT,
    _show_caption,
    _show_error,
    _show_info,
    render_page_chrome,
)
from app.shared.unit_status import clean_unit_status


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="Grifols", layout="wide", page_icon=None, initial_sidebar_state="expanded")
    render_page_chrome()

    # _get_supabase_client is @st.cache_resource — returns the same instance every call.
    sb_client = _get_supabase_client()
    sb_error = _get_supabase_error() if sb_client is None else ""

    shipment_file = _sb_file_widget(
        "Upload Grifols shipment file",
        "shipment",
        "shipment",
        ["csv", "xlsx", "xls"],
        sb_client,
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
        sb_client,
    )

    # Load the DataFrames
    gs_df = read_upload(shipment_file, sheet_strategy="latest", explicit_sheet=_gs_explicit_sheet)
    us_df = read_upload(unit_status_file, dtype=str, sheet_strategy="unit_status")

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

    _visible_sections = visible_sections(_is_unlocked)

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
        if sb_client:
            _show_caption("Storage: connected")
        elif sb_error:
            _show_caption(f"Storage: {sb_error}")
        st.markdown("---")
        nav_section = st.radio(
            "Go to section",
            _visible_sections,
            key="nav_section",
        )
        st.markdown("---")
        if nav_section == "Pallet Report":
            pallet_no, verbose = pallet_report.sidebar_inputs()
        else:
            pallet_no = st.session_state.get("pallet_no_last", 1)
            verbose = False

    # ── Dispatch ────────────────────────────────────────────────────────
    if nav_section == "Pallet Report":
        pallet_report.render(gs_df, us_df, pallet_no, verbose)
    elif nav_section == "Manual racks/Unit Status Check":
        manual_racks.render(gs_df, us_df)
    elif nav_section == "Visual Inspection Labels":
        if us_df is None:
            _show_info("Please upload a unit status file to use this section.")
        else:
            vi_labels.render(us_df, key_ns="nav")
    elif nav_section == "QC Report PDF Extractor":
        qc_report.render(gs_df, us_df, sb_client)
    elif nav_section == "Master Sheet ":
        master_sheet.render(gs_df, us_df, sb_client)
    elif nav_section == "Storage Manager":
        storage_manager.render(sb_client, sb_error)
    elif nav_section == "Donation Processing":
        donation_processing.render(us_df, sb_client, sb_error)
