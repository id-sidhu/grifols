"""QC Report PDF Extractor section."""

import io
import re
from typing import Dict, List

import pandas as pd
import streamlit as st

from app.features.qc_report.cache import (
    _sb_delete_qc_cache,
    _sb_get_qc_cache,
    _sb_save_qc_cache,
)
from app.features.qc_report.logic import (
    build_qc_packing_comparison,
    build_qc_release_comparison,
    parse_qc_report_pdf,
)
from app.shared.storage import _sb_download, _sb_list_files, _sb_upload
from app.shared.ui import (
    _show_caption,
    _show_error,
    _show_info,
    _show_success,
    _show_warning,
    _subheader,
)


def render(gs_df, us_df, sb_client) -> None:
    """Render the QC Report PDF Extractor section."""
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

    if sb_client:
        # refresh listing once per session (or after a save)
        if _qc_ls_key not in st.session_state:
            st.session_state[_qc_ls_key] = _sb_list_files(sb_client, "qc-reports")
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
                        sb_client, f"qc-reports/{_qf.name}",
                        _qf.read(), "application/pdf",
                    )
                    _qf.seek(0)
                    (_qc_saved if _res is True else _qc_failed).append(_qf.name)
                if _qc_saved:
                    _show_success(f"Saved: {', '.join(_qc_saved)}")
                    st.session_state[_qc_ls_key] = _sb_list_files(sb_client, "qc-reports")
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
                        _raw = _sb_download(sb_client, f"qc-reports/{_qfn}")
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
        if sb_client:
            _cache_key = "_qc_cache_status_" + "_".join(
                sorted(f.name for f in qc_pdf_files)
            )
            if _cache_key not in st.session_state:
                st.session_state[_cache_key] = {
                    f.name: _sb_get_qc_cache(sb_client, f.name) is not None
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
                            _sb_delete_qc_cache(sb_client, _fname)
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
                    if sb_client:
                        _cached_df = _sb_get_qc_cache(sb_client, qc_pdf_file.name)
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
                        if sb_client:
                            _sb_save_qc_cache(sb_client, qc_pdf_file.name, _df)
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
