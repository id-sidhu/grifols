"""Supabase Storage access: the client, raw object helpers, and the
upload-or-pick-from-storage file widget used by the main uploaders.
"""

from typing import Dict, List, Optional

import streamlit as st

from app.shared.ui import _show_caption, _show_error, _show_info, _show_success, _show_warning


# ---------------------------------------------------------------------------
# Supabase storage helpers
# All files live inside one bucket whose name is set here or overridden via
# the SUPABASE_BUCKET secret.
# ---------------------------------------------------------------------------
_SUPABASE_BUCKET = "grifols"

# Storage folder holding the canonical Unit Status CSV database (one file per
# year, named "Unit Status(UNIT STATUS <year>) ...csv").
_US_FOLDER = "unit-status"


@st.cache_resource(show_spinner=False)
def _get_supabase_client():
    """Return a Supabase client if SUPABASE_URL / SUPABASE_KEY are in secrets.

    Decorated with @st.cache_resource so the connection is created once and
    reused across all reruns (avoids repeated TCP handshakes).
    """
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        if not url or not key:
            return None
        _bucket_override = st.secrets.get("SUPABASE_BUCKET", "")
        global _SUPABASE_BUCKET
        if _bucket_override:
            _SUPABASE_BUCKET = _bucket_override
        return create_client(url, key)
    except Exception:
        return None


def _get_supabase_error() -> str:
    """Return a human-readable reason why Supabase failed to connect, or empty string."""
    try:
        from supabase import create_client  # noqa: F401
    except ImportError:
        return "supabase package not installed (run: pip install supabase)"
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    if not url:
        return "SUPABASE_URL missing from secrets.toml"
    if not key:
        return "SUPABASE_KEY missing from secrets.toml"
    try:
        from supabase import create_client
        create_client(url, key)
        return ""
    except Exception as e:
        return str(e)


def _sb_list_files(client, folder: str) -> List[str]:
    """Return sorted file names inside a Supabase storage folder."""
    try:
        items = client.storage.from_(_SUPABASE_BUCKET).list(folder) or []
        return sorted(
            item["name"] for item in items
            if item.get("name") and not item["name"].startswith(".")
        )
    except Exception:
        return []


def _sb_download(client, path: str) -> Optional[bytes]:
    """Download raw bytes from Supabase storage. Returns None on failure."""
    try:
        return client.storage.from_(_SUPABASE_BUCKET).download(path)
    except Exception:
        return None


def _sb_upload(client, path: str, data: bytes, mime: str = "text/csv"):
    """Upsert a file to Supabase storage. Returns True on success, error string on failure."""
    try:
        client.storage.from_(_SUPABASE_BUCKET).upload(
            path, data, file_options={"content-type": mime, "upsert": "true"},
        )
        return True
    except Exception as e:
        return str(e)


def _sb_delete(client, path: str):
    """Delete a file from Supabase storage. Returns True on success, error string on failure."""
    try:
        client.storage.from_(_SUPABASE_BUCKET).remove([path])
        return True
    except Exception as e:
        return str(e)

def _sb_file_widget(
    label: str,
    folder: str,
    uploader_key: str,
    file_types: List[str],
    client,
    accept_multiple: bool = False,
    save_mime: str = "text/csv",
):
    """File uploader augmented with optional Supabase storage.

    Without Supabase (``client=None``) behaves identically to
    ``st.file_uploader``.  When connected:

    * An uploaded file shows a **"Save to storage"** button.
    * A selectbox / multiselect lets the user load a previously-saved file
      instead of uploading again.  Downloaded bytes are cached in session
      state so repeated reruns don't re-fetch from Supabase.
    * An uploaded file always takes priority over a storage selection.
    """
    # --- standard uploader (always shown) ---
    uploaded = st.file_uploader(
        label, type=file_types, key=uploader_key,
        accept_multiple_files=accept_multiple,
    )

    if client is None:
        return uploaded

    # ---- keys used for session-state caching ----
    _ls_key = f"_sb_ls_{uploader_key}"   # cached file listing
    _ln_key = f"_sb_ln_{uploader_key}"   # last loaded name(s)
    _ld_key = f"_sb_ld_{uploader_key}"   # loaded bytes (or dict)

    # Populate the file listing once per session (invalidated after a save)
    if _ls_key not in st.session_state:
        st.session_state[_ls_key] = _sb_list_files(client, folder)
    _sb_files: List[str] = st.session_state[_ls_key]

    # ---- save button(s) for freshly uploaded files ----
    if uploaded and not accept_multiple:
        if st.button(
            f"Save '{uploaded.name}' to storage",
            key=f"{uploader_key}_sb_save",
        ):
            uploaded.seek(0)
            _raw = uploaded.read()
            uploaded.seek(0)
            _result = _sb_upload(client, f"{folder}/{uploaded.name}", _raw, mime=save_mime)
            if _result is True:
                _show_success(f"Saved to storage: **{uploaded.name}**")
                st.session_state[_ls_key] = _sb_list_files(client, folder)
            else:
                _show_error(f"Save failed: {_result}")

    elif uploaded and accept_multiple:
        if st.button(
            f"Save {len(uploaded)} file(s) to storage",
            key=f"{uploader_key}_sb_save",
        ):
            _saved, _failed = [], []
            for _uf in uploaded:
                _uf.seek(0)
                _raw = _uf.read()
                _uf.seek(0)
                if _sb_upload(client, f"{folder}/{_uf.name}", _raw, save_mime):
                    _saved.append(_uf.name)
                else:
                    _failed.append(_uf.name)
            if _saved:
                _show_success(f"Saved: {', '.join(_saved)}")
                st.session_state[_ls_key] = _sb_list_files(client, folder)
            if _failed:
                _show_warning(f"Failed to save: {', '.join(_failed)}")

    # ---- load from storage ----
    if _sb_files:
        if not accept_multiple:
            _opts = ["— select from storage —"] + _sb_files
            _sel = st.selectbox(
                "Or load from storage:",
                _opts,
                key=f"{uploader_key}_sb_pick",
            )
            if _sel and _sel != "— select from storage —":
                if st.session_state.get(_ln_key) != _sel:
                    _data = _sb_download(client, f"{folder}/{_sel}")
                    st.session_state[_ln_key] = _sel
                    st.session_state[_ld_key] = _data
        else:
            _sel_multi = st.multiselect(
                "Or load from storage:",
                _sb_files,
                key=f"{uploader_key}_sb_pick",
            )
            if _sel_multi:
                if st.session_state.get(_ln_key) != _sel_multi:
                    _multi_data: Dict[str, bytes] = {}
                    for _fn in _sel_multi:
                        _d = _sb_download(client, f"{folder}/{_fn}")
                        if _d:
                            _multi_data[_fn] = _d
                    st.session_state[_ln_key] = _sel_multi
                    st.session_state[_ld_key] = _multi_data
            else:
                st.session_state.pop(_ln_key, None)
                st.session_state.pop(_ld_key, None)
    elif not uploaded:
        _show_caption(f"No files in storage folder `{folder}/` yet — upload one above.")

    # ---- determine return value ----
    if uploaded:
        # Uploaded file wins; clear any stale storage cache so the next
        # time the uploader is empty the storage pick still works.
        st.session_state.pop(_ln_key, None)
        st.session_state.pop(_ld_key, None)
        return uploaded

    _cached_data = st.session_state.get(_ld_key)
    _cached_name = st.session_state.get(_ln_key)
    if _cached_data and _cached_name:
        import io as _io_sb
        if not accept_multiple:
            _f = _io_sb.BytesIO(_cached_data)
            _f.name = _cached_name
            _show_info(f"Using from storage: **{_cached_name}**")
            return _f
        else:
            _files_sb: List = []
            for _fn, _fd in _cached_data.items():
                _f = _io_sb.BytesIO(_fd)
                _f.name = _fn
                _files_sb.append(_f)
            if _files_sb:
                _show_info(f"Using from storage: **{', '.join(_cached_data.keys())}**")
            return _files_sb

    return [] if accept_multiple else None
