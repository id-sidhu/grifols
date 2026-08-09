"""Storage Manager section: browse, upload and delete Supabase objects."""

import re
from typing import Optional

import pandas as pd
import streamlit as st

from app.shared.dates import _parse_donation_date
from app.shared.excel import _read_excel_smart
from app.shared.storage import (
    _get_supabase_client,
    _sb_delete,
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


def render(sb_client, sb_error: str) -> None:
    """Render the Storage Manager section."""
    _subheader("Storage Manager")

    if sb_client is None:
        st.error(
            "Supabase is not connected. "
            + (f"**Reason:** {sb_error}" if sb_error else "Check SUPABASE_URL and SUPABASE_KEY in secrets.")
        )
        if st.button("Retry connection", key="_sm_retry_conn"):
            _get_supabase_client.clear()
            st.rerun()
    else:
        _sm_folders = {
            "Unit Status (2026)": ("unit-status", ["csv", "xlsx", "xls"]),
            "Grifols Shipment": ("shipment", ["csv", "xlsx", "xls"]),
            "Master Sheet": ("master-sheet", ["csv"]),
            "QC Reports": ("qc-reports", ["pdf"]),
        }

        def _sm_file_mime(fname: str) -> str:
            if fname.endswith(".xlsx"):
                return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if fname.endswith(".xls"):
                return "application/vnd.ms-excel"
            return "text/csv"

        def _sm_load_df(raw: bytes, fname: str, sheet_strategy: str = "score") -> Optional[pd.DataFrame]:
            import io as _io_sm
            _f = _io_sm.BytesIO(raw)
            try:
                if fname.endswith((".xlsx", ".xls")):
                    return _read_excel_smart(_f, sheet_strategy=sheet_strategy)
                for _enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                    try:
                        _f.seek(0)
                        return pd.read_csv(_f, dtype=str, encoding=_enc).fillna("")
                    except UnicodeDecodeError:
                        continue
            except Exception:
                pass
            return None

        def _sm_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
            return df.to_csv(index=False).encode("utf-8")

        for _sm_label, (_sm_folder, _sm_types) in _sm_folders.items():
            st.markdown(f"### {_sm_label}")
            _sm_ls_key = f"_sm_ls_{_sm_folder}"
            if _sm_ls_key not in st.session_state:
                st.session_state[_sm_ls_key] = _sb_list_files(sb_client, _sm_folder)
            _sm_files = st.session_state[_sm_ls_key]

            if _sm_files:
                for _sm_fname in _sm_files:
                    _sm_col1, _sm_col2, _sm_col3 = st.columns([4, 1, 1])
                    _sm_col1.write(_sm_fname)

                    # Replace button
                    _sm_replace_file = _sm_col2.file_uploader(
                        "Replace",
                        type=_sm_types,
                        key=f"_sm_replace_{_sm_folder}_{_sm_fname}",
                        label_visibility="collapsed",
                    )
                    if _sm_replace_file is not None:
                        _sm_replace_file.seek(0)
                        _sm_res = _sb_upload(
                            sb_client,
                            f"{_sm_folder}/{_sm_fname}",
                            _sm_replace_file.read(),
                            _sm_file_mime(_sm_replace_file.name),
                        )
                        if _sm_res is True:
                            _show_success(f"Replaced **{_sm_fname}** successfully.")
                            st.session_state[_sm_ls_key] = _sb_list_files(sb_client, _sm_folder)
                            for _k in list(st.session_state.keys()):
                                if _sm_folder.replace("-", "_") in _k and "_sb_ln_" in _k:
                                    st.session_state.pop(_k, None)
                            st.rerun()
                        else:
                            _show_error(f"Replace failed: {_sm_res}")

                    # Delete button
                    if _sm_col3.button("Delete", key=f"_sm_del_{_sm_folder}_{_sm_fname}"):
                        _sm_del_res = _sb_delete(sb_client, f"{_sm_folder}/{_sm_fname}")
                        if _sm_del_res is True:
                            _show_success(f"Deleted **{_sm_fname}**.")
                            st.session_state[_sm_ls_key] = _sb_list_files(sb_client, _sm_folder)
                            for _k in list(st.session_state.keys()):
                                if _sm_folder.replace("-", "_") in _k and "_sb_ln_" in _k:
                                    st.session_state.pop(_k, None)
                            st.rerun()
                        else:
                            _show_error(f"Delete failed: {_sm_del_res}")
            else:
                _show_caption(f"No files stored in `{_sm_folder}/` yet.")

            # Upload new file to this folder
            with st.expander(f"Upload new file to {_sm_label}"):
                _sm_new_file = st.file_uploader(
                    "Choose file", type=_sm_types, key=f"_sm_new_{_sm_folder}"
                )
                if _sm_new_file is not None:
                    if st.button("Save to storage", key=f"_sm_new_save_{_sm_folder}"):
                        _sm_new_file.seek(0)
                        _sm_new_res = _sb_upload(
                            sb_client,
                            f"{_sm_folder}/{_sm_new_file.name}",
                            _sm_new_file.read(),
                            _sm_file_mime(_sm_new_file.name),
                        )
                        if _sm_new_res is True:
                            _show_success(f"Saved **{_sm_new_file.name}** to storage.")
                            st.session_state[_sm_ls_key] = _sb_list_files(sb_client, _sm_folder)
                            st.rerun()
                        else:
                            _show_error(f"Upload failed: {_sm_new_res}")

            # ── Unit Status row editing ──────────────────────────────────
            if _sm_folder == "unit-status" and _sm_files:
                with st.expander("Edit Rows in Unit Status File"):
                    _us_edit_sel = st.selectbox(
                        "Select file to edit",
                        _sm_files,
                        key="_sm_us_edit_sel",
                    )
                    _us_df_key = f"_sm_us_df_{_us_edit_sel}"

                    # Auto-load when selection changes or not yet loaded
                    if _us_df_key not in st.session_state:
                        _us_raw = _sb_download(sb_client, f"unit-status/{_us_edit_sel}")
                        if _us_raw:
                            _loaded = _sm_load_df(_us_raw, _us_edit_sel, sheet_strategy="unit_status")
                            if _loaded is not None:
                                st.session_state[_us_df_key] = _loaded
                            else:
                                _show_error("Could not parse the selected file.")
                        else:
                            _show_error("Could not download the selected file from storage.")

                    if _us_df_key in st.session_state:
                        _us_df_edit: pd.DataFrame = st.session_state[_us_df_key]
                        _us_cols = list(_us_df_edit.columns)
                        _show_caption(f"{len(_us_df_edit)} rows · {len(_us_cols)} columns")

                        with st.expander("Preview last 10 rows", expanded=False):
                            st.dataframe(
                                _us_df_edit.tail(10),
                                use_container_width=True,
                                hide_index=False,
                            )

                        _tab_add, _tab_find, _tab_ctrl_us = st.tabs(
                            ["Add Rows", "Edit by Date", "Edit by Control #"]
                        )

                        # ── Add Rows (paste) ─────────────────────────────
                        with _tab_add:
                            _show_caption(
                                f"Paste rows copied from Excel. Expected column order: "
                                f"`{'  |  '.join(_us_cols)}`"
                            )
                            _paste_text = st.text_area(
                                "Paste rows here (tab-separated, no header)",
                                height=200,
                                key="_sm_paste_rows",
                                placeholder="F26-000001\t10057852\tQ\t02.01.2026\t1\tQuarantine\t\t\tTO BE SHIPPED ON 05.01.2026",
                            )
                            if st.button("Preview & Append", key="_sm_paste_preview_btn"):
                                if not _paste_text.strip():
                                    _show_warning("Nothing pasted.")
                                else:
                                    import io as _io_paste
                                    try:
                                        _pasted_df = pd.read_csv(
                                            _io_paste.StringIO(_paste_text.strip()),
                                            sep="\t",
                                            header=None,
                                            dtype=str,
                                        ).fillna("")
                                        _n_pasted_cols = _pasted_df.shape[1]
                                        _n_expected = len(_us_cols)
                                        if _n_pasted_cols != _n_expected:
                                            _show_warning(
                                                f"Pasted data has {_n_pasted_cols} columns "
                                                f"but file has {_n_expected}. "
                                                "Extra columns will be dropped; missing ones filled blank."
                                            )
                                        # Align columns
                                        _pasted_df.columns = (
                                            _us_cols[:_n_pasted_cols]
                                            + [f"_extra_{i}" for i in range(max(0, _n_pasted_cols - _n_expected))]
                                        )
                                        for _mc in _us_cols:
                                            if _mc not in _pasted_df.columns:
                                                _pasted_df[_mc] = ""
                                        _pasted_df = _pasted_df[_us_cols]
                                        st.session_state["_sm_paste_preview"] = _pasted_df
                                    except Exception as _pe:
                                        _show_error(f"Could not parse pasted text: {_pe}")

                            if "_sm_paste_preview" in st.session_state:
                                _preview_df = st.session_state["_sm_paste_preview"]
                                st.write(f"**{len(_preview_df)} row(s) to append:**")
                                st.dataframe(_preview_df, use_container_width=True)
                                if st.button("Confirm — Append to file", key="_sm_paste_confirm_btn"):
                                    _updated_us = pd.concat(
                                        [_us_df_edit, _preview_df], ignore_index=True
                                    )
                                    _save_res = _sb_upload(
                                        sb_client,
                                        f"unit-status/{_us_edit_sel}",
                                        _sm_df_to_csv_bytes(_updated_us),
                                        "text/csv",
                                    )
                                    if _save_res is True:
                                        st.session_state[_us_df_key] = _updated_us
                                        st.session_state.pop("_sm_paste_preview", None)
                                        _show_success(
                                            f"{len(_preview_df)} row(s) appended. "
                                            f"File now has {len(_updated_us)} rows."
                                        )
                                        st.rerun()
                                    else:
                                        _show_error(f"Save failed: {_save_res}")

                        # ── Edit by Date ─────────────────────────────────
                        with _tab_find:
                            if "Donation Date" not in _us_cols:
                                st.warning("Column 'Donation Date' not found in this file.")
                            else:
                                _date_input_str = st.text_input(
                                    "Donation date (dd.mm.yyyy)",
                                    key="_sm_date_q",
                                    placeholder="02.01.2026",
                                )
                                if st.button("Search by date", key="_sm_date_search_btn"):
                                    _dq_stripped = _date_input_str.strip()
                                    if not _dq_stripped:
                                        _show_warning("Enter a date to search.")
                                    else:
                                        _target_date = _parse_donation_date(_dq_stripped)
                                        if _target_date is None:
                                            _show_warning(
                                                f"Could not parse '{_dq_stripped}'. "
                                                "Use dd.mm.yyyy, dd-mm-yyyy or dd/mm/yyyy."
                                            )
                                        else:
                                            _date_mask = _us_df_edit["Donation Date"].apply(
                                                lambda _v: _parse_donation_date(_v) == _target_date
                                            )
                                            _found_idx = _us_df_edit.index[_date_mask].tolist()
                                            st.session_state["_sm_found_idx"] = _found_idx
                                            st.session_state["_sm_found_key"] = _us_df_key
                                            st.session_state["_sm_found_date"] = _target_date.strftime("%d.%m.%Y")
                                            if not _found_idx:
                                                _show_info(
                                                    f"No rows found for "
                                                    f"{_target_date.strftime('%d.%m.%Y')}."
                                                )

                                _found = st.session_state.get("_sm_found_idx", [])
                                _found_key = st.session_state.get("_sm_found_key", "")
                                _found_date_label = st.session_state.get("_sm_found_date", "")
                                # Clear if user switched to a different file
                                if _found_key != _us_df_key:
                                    _found = []

                                if _found:
                                    # Filter to rows still present in the DataFrame
                                    _valid_found = [i for i in _found if i in _us_df_edit.index]
                                    st.markdown(
                                        f"**{len(_valid_found)} row(s)** for **{_found_date_label}** — "
                                        "edit directly in the table then click Save."
                                    )
                                    _edit_subset = _us_df_edit.loc[_valid_found].copy()
                                    _edited_subset = st.data_editor(
                                        _edit_subset,
                                        use_container_width=True,
                                        hide_index=False,
                                        key=f"_sm_date_editor_{_found_key}_{_found_date_label}",
                                        num_rows="fixed",
                                    )
                                    if st.button("Save Changes", key="_sm_date_save_btn"):
                                        for _col in _us_cols:
                                            if _col in _edited_subset.columns:
                                                _us_df_edit.loc[_valid_found, _col] = (
                                                    _edited_subset[_col].values
                                                )
                                        _save_res = _sb_upload(
                                            sb_client,
                                            f"unit-status/{_us_edit_sel}",
                                            _sm_df_to_csv_bytes(_us_df_edit),
                                            "text/csv",
                                        )
                                        if _save_res is True:
                                            st.session_state[_us_df_key] = _us_df_edit
                                            _show_success(
                                                f"Saved {len(_valid_found)} row(s) for "
                                                f"{_found_date_label}."
                                            )
                                            st.rerun()
                                        else:
                                            _show_error(f"Save failed: {_save_res}")

                        # ── Edit by Control # (Unit Status) ─────────────
                        with _tab_ctrl_us:
                            _us_ctrl_col = "Donation #"
                            if _us_ctrl_col not in _us_cols:
                                st.warning(f"Column '{_us_ctrl_col}' not found in this file.")
                            else:
                                _us_ctrl_q = st.text_input(
                                    "Control number (Donation # suffix, e.g. 002035)",
                                    key="_sm_us_ctrl_q",
                                    placeholder="002035",
                                )
                                if st.button("Search", key="_sm_us_ctrl_search_btn"):
                                    if not _us_ctrl_q.strip():
                                        _show_warning("Enter a control number.")
                                    else:
                                        _us_ctrl_mask = (
                                            _us_df_edit[_us_ctrl_col]
                                            .astype(str).str.strip()
                                            .str.endswith(_us_ctrl_q.strip())
                                        )
                                        _us_ctrl_found = _us_df_edit.index[_us_ctrl_mask].tolist()
                                        st.session_state["_sm_us_ctrl_found"] = _us_ctrl_found
                                        st.session_state["_sm_us_ctrl_key"] = _us_df_key
                                        if not _us_ctrl_found:
                                            _show_info(
                                                f"No rows where {_us_ctrl_col} ends with "
                                                f"'{_us_ctrl_q.strip()}'."
                                            )
                                _us_ctrl_found = st.session_state.get("_sm_us_ctrl_found", [])
                                if st.session_state.get("_sm_us_ctrl_key", "") != _us_df_key:
                                    _us_ctrl_found = []
                                if _us_ctrl_found:
                                    _us_ctrl_valid = [i for i in _us_ctrl_found if i in _us_df_edit.index]
                                    st.markdown(f"**{len(_us_ctrl_valid)} row(s) found**")
                                    _us_ctrl_edited = st.data_editor(
                                        _us_df_edit.loc[_us_ctrl_valid].copy(),
                                        use_container_width=True,
                                        hide_index=False,
                                        key=f"_sm_us_ctrl_editor_{_us_df_key}",
                                        num_rows="fixed",
                                    )
                                    if st.button("Save Changes", key="_sm_us_ctrl_save_btn"):
                                        for _col in _us_cols:
                                            if _col in _us_ctrl_edited.columns:
                                                _us_df_edit.loc[_us_ctrl_valid, _col] = (
                                                    _us_ctrl_edited[_col].values
                                                )
                                        _save_res = _sb_upload(
                                            sb_client,
                                            f"unit-status/{_us_edit_sel}",
                                            _sm_df_to_csv_bytes(_us_df_edit),
                                            "text/csv",
                                        )
                                        if _save_res is True:
                                            st.session_state[_us_df_key] = _us_df_edit
                                            _show_success(f"Saved {len(_us_ctrl_valid)} row(s).")
                                            st.rerun()
                                        else:
                                            _show_error(f"Save failed: {_save_res}")

            # ── Shipment file row editing ────────────────────────────────
            if _sm_folder == "shipment" and _sm_files:
                with st.expander("Edit Rows in Shipment File"):
                    _gs_edit_sel = st.selectbox(
                        "Select file to edit",
                        _sm_files,
                        key="_sm_gs_edit_sel",
                    )
                    _gs_df_key = f"_sm_gs_df_{_gs_edit_sel}"

                    if _gs_df_key not in st.session_state:
                        _gs_raw = _sb_download(sb_client, f"shipment/{_gs_edit_sel}")
                        if _gs_raw:
                            _loaded_gs = _sm_load_df(_gs_raw, _gs_edit_sel, sheet_strategy="latest")
                            if _loaded_gs is not None:
                                st.session_state[_gs_df_key] = _loaded_gs
                            else:
                                _show_error("Could not parse the selected file.")
                        else:
                            _show_error("Could not download the selected file from storage.")

                    if _gs_df_key in st.session_state:
                        _gs_df_edit: pd.DataFrame = st.session_state[_gs_df_key]
                        _gs_cols = list(_gs_df_edit.columns)
                        _show_caption(f"{len(_gs_df_edit)} rows · {len(_gs_cols)} columns")

                        with st.expander("Preview last 10 rows", expanded=False):
                            st.dataframe(
                                _gs_df_edit.tail(10),
                                use_container_width=True,
                                hide_index=False,
                            )

                        _gs_tab_pallet, _gs_tab_ctrl = st.tabs(
                            ["Edit by Pallet", "Edit by Control #"]
                        )

                        # ── Edit by Pallet ────────────────────────────────
                        with _gs_tab_pallet:
                            if "Comments" not in _gs_cols:
                                st.warning("Column 'Comments' not found — cannot search by pallet.")
                            else:
                                _gs_pallet_no = st.number_input(
                                    "Pallet number", min_value=1, step=1,
                                    value=1, key="_sm_gs_pallet_no", format="%d",
                                )
                                if st.button("Search pallet", key="_sm_gs_pallet_search_btn"):
                                    _p = int(_gs_pallet_no)
                                    _sop_pat = re.compile(
                                        rf"^\s*START\s+OF\s+PALLET\s+{_p}\s*$", re.IGNORECASE
                                    )
                                    _eop_pat = re.compile(
                                        rf"^\s*END\s+OF\s+PALLET\s+{_p}\s*$", re.IGNORECASE
                                    )
                                    _gs_comments = _gs_df_edit["Comments"].fillna("").astype(str)
                                    _sop_rows = _gs_df_edit.index[_gs_comments.str.match(_sop_pat)].tolist()
                                    _eop_rows = _gs_df_edit.index[_gs_comments.str.match(_eop_pat)].tolist()
                                    if not _sop_rows or not _eop_rows:
                                        _show_info(f"Pallet {_p} markers not found in Comments.")
                                        st.session_state["_sm_gs_pallet_idx"] = []
                                    else:
                                        _p_start = min(_sop_rows[0], _eop_rows[0])
                                        _p_end = max(_sop_rows[0], _eop_rows[0])
                                        _p_idx = [
                                            i for i in range(_p_start, _p_end + 1)
                                            if i in _gs_df_edit.index
                                        ]
                                        st.session_state["_sm_gs_pallet_idx"] = _p_idx
                                        st.session_state["_sm_gs_pallet_key"] = _gs_df_key
                                        st.session_state["_sm_gs_pallet_num"] = _p
                                        if not _p_idx:
                                            _show_info(f"No rows found for Pallet {_p}.")

                                _gs_pallet_idx = st.session_state.get("_sm_gs_pallet_idx", [])
                                if st.session_state.get("_sm_gs_pallet_key", "") != _gs_df_key:
                                    _gs_pallet_idx = []

                                if _gs_pallet_idx:
                                    _gs_p_num = st.session_state.get("_sm_gs_pallet_num", "")
                                    st.markdown(
                                        f"**{len(_gs_pallet_idx)} row(s)** for "
                                        f"**Pallet {_gs_p_num}** — edit then click Save."
                                    )
                                    _gs_pallet_edited = st.data_editor(
                                        _gs_df_edit.loc[_gs_pallet_idx].copy(),
                                        use_container_width=True,
                                        hide_index=False,
                                        key=f"_sm_gs_pallet_editor_{_gs_df_key}_{_gs_p_num}",
                                        num_rows="fixed",
                                    )
                                    if st.button("Save Changes", key="_sm_gs_pallet_save_btn"):
                                        for _col in _gs_cols:
                                            if _col in _gs_pallet_edited.columns:
                                                _gs_df_edit.loc[_gs_pallet_idx, _col] = (
                                                    _gs_pallet_edited[_col].values
                                                )
                                        _save_res = _sb_upload(
                                            sb_client,
                                            f"shipment/{_gs_edit_sel}",
                                            _sm_df_to_csv_bytes(_gs_df_edit),
                                            "text/csv",
                                        )
                                        if _save_res is True:
                                            st.session_state[_gs_df_key] = _gs_df_edit
                                            _show_success(
                                                f"Saved {len(_gs_pallet_idx)} row(s) "
                                                f"for Pallet {_gs_p_num}."
                                            )
                                            st.rerun()
                                        else:
                                            _show_error(f"Save failed: {_save_res}")

                        # ── Edit by Control # (Shipment) ──────────────────
                        with _gs_tab_ctrl:
                            _gs_ctrl_col = "Sample ID"
                            if _gs_ctrl_col not in _gs_cols:
                                st.warning(f"Column '{_gs_ctrl_col}' not found in this file.")
                            else:
                                _gs_ctrl_q = st.text_input(
                                    "Control number (Sample ID suffix, e.g. 002035)",
                                    key="_sm_gs_ctrl_q",
                                    placeholder="002035",
                                )
                                if st.button("Search", key="_sm_gs_ctrl_search_btn"):
                                    if not _gs_ctrl_q.strip():
                                        _show_warning("Enter a control number.")
                                    else:
                                        _gs_ctrl_mask = (
                                            _gs_df_edit[_gs_ctrl_col]
                                            .astype(str).str.strip()
                                            .str.endswith(_gs_ctrl_q.strip())
                                        )
                                        _gs_ctrl_found = _gs_df_edit.index[_gs_ctrl_mask].tolist()
                                        st.session_state["_sm_gs_ctrl_found"] = _gs_ctrl_found
                                        st.session_state["_sm_gs_ctrl_key"] = _gs_df_key
                                        if not _gs_ctrl_found:
                                            _show_info(
                                                f"No rows where {_gs_ctrl_col} ends with "
                                                f"'{_gs_ctrl_q.strip()}'."
                                            )
                                _gs_ctrl_found = st.session_state.get("_sm_gs_ctrl_found", [])
                                if st.session_state.get("_sm_gs_ctrl_key", "") != _gs_df_key:
                                    _gs_ctrl_found = []
                                if _gs_ctrl_found:
                                    _gs_ctrl_valid = [i for i in _gs_ctrl_found if i in _gs_df_edit.index]
                                    st.markdown(f"**{len(_gs_ctrl_valid)} row(s) found**")
                                    _gs_ctrl_edited = st.data_editor(
                                        _gs_df_edit.loc[_gs_ctrl_valid].copy(),
                                        use_container_width=True,
                                        hide_index=False,
                                        key=f"_sm_gs_ctrl_editor_{_gs_df_key}",
                                        num_rows="fixed",
                                    )
                                    if st.button("Save Changes", key="_sm_gs_ctrl_save_btn"):
                                        for _col in _gs_cols:
                                            if _col in _gs_ctrl_edited.columns:
                                                _gs_df_edit.loc[_gs_ctrl_valid, _col] = (
                                                    _gs_ctrl_edited[_col].values
                                                )
                                        _save_res = _sb_upload(
                                            sb_client,
                                            f"shipment/{_gs_edit_sel}",
                                            _sm_df_to_csv_bytes(_gs_df_edit),
                                            "text/csv",
                                        )
                                        if _save_res is True:
                                            st.session_state[_gs_df_key] = _gs_df_edit
                                            _show_success(f"Saved {len(_gs_ctrl_valid)} row(s).")
                                            st.rerun()
                                        else:
                                            _show_error(f"Save failed: {_save_res}")

            st.markdown("---")
