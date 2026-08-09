"""Manual racks / Unit Status Check section."""

import re
from typing import Dict, List, Set

import pandas as pd
import streamlit as st

from app.features.manual_racks.logic import process_unit_status_all
from app.features.manual_racks.rack import (
    _show_rack_fullscreen_dialog,
    build_rack_html,
)
from app.features.pallet_report.logic import build_pallet_map
from app.shared.dates import _parse_donation_date
from app.shared.ui import _show_error, _show_info, _show_success, _show_warning, _subheader
from app.shared.unit_status import clean_unit_status


def render(gs_df, us_df) -> None:
    """Render the Manual racks / Unit Status Check section."""
    if us_df is None:
        _show_info("Please upload a unit status file to use this section.")
        return

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
