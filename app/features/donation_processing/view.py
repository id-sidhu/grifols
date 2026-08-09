"""Donation Processing dashboard section."""

import hashlib

import pandas as pd
import streamlit as st

from app.features.donation_processing import unit_status_db as usdb
from app.features.donation_processing.logic import (
    dp_parse_barcodes,
    dp_process_pc_blut,
    dp_validate_freeze_tracker,
)
from app.features.donation_processing.rack import _DP_RACK_CSS, dp_build_rack_html
from app.features.vi_labels.view import render_vi_labels
from app.shared.dates import _parse_donation_date
from app.shared.storage import (
    _US_FOLDER,
    _sb_download,
    _sb_list_files,
    _sb_upload,
)
from app.shared.ui import (
    _show_caption,
    _show_error,
    _show_info,
    _show_success,
    _show_warning,
    _subheader,
)


def render(us_df, sb_client, sb_error: str) -> None:
    """Render the Donation Processing dashboard."""
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
        elif sb_client is None:
            _show_error(
                "Storage is not connected, so the unit status database "
                "cannot be updated. "
                + (sb_error or "Check SUPABASE_URL / SUPABASE_KEY.")
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
                _dp_names = _sb_list_files(sb_client, _US_FOLDER)
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
                        sb_client, f"{_US_FOLDER}/{_dp_target}"
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

                                # The sidebar drives us_df, which the label
                                # step below reads.  Anything overriding the
                                # commit target there would build labels from
                                # the wrong data, so say so up front.
                                _dp_side = st.session_state.get(
                                    "unit_status_sb_pick"
                                )
                                if st.session_state.get("unit_status") is not None:
                                    _show_warning(
                                        "A locally uploaded unit status file "
                                        "is active in the sidebar and takes "
                                        "priority over storage. Clear the "
                                        "uploader so the labels are built "
                                        "from the updated database."
                                    )
                                elif (
                                    _dp_side
                                    and _dp_side != _dp_target
                                    and _dp_side in _dp_names
                                ):
                                    _show_warning(
                                        f"The sidebar is loading "
                                        f"**{_dp_side}**, not the commit "
                                        f"target **{_dp_target}**. Switch it "
                                        f"after committing so the labels use "
                                        f"the updated database."
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
                                            sb_client, _dp_arch, _dp_raw
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
                                            sb_client,
                                            f"{_US_FOLDER}/{_dp_target}",
                                            _dp_new,
                                            mime="text/csv",
                                        )
                                        if _dp_up is True:
                                            # 3. make the whole app see the
                                            #    updated database at once
                                            st.session_state["_sb_ls_unit_status"] = (
                                                _sb_list_files(
                                                    sb_client, _US_FOLDER
                                                )
                                            )
                                            st.session_state["_sb_ln_unit_status"] = _dp_target
                                            st.session_state["_sb_ld_unit_status"] = _dp_new
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
