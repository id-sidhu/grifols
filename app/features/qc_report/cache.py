"""Supabase-backed cache of parsed QC PDFs, keyed by source filename."""

import io
from typing import Optional

import pandas as pd

from app.shared.storage import _sb_delete, _sb_download, _sb_upload


def _sb_get_qc_cache(client, pdf_name: str) -> Optional[pd.DataFrame]:
    """Load cached QC extraction for pdf_name from Supabase (qc-cache/ folder).

    Returns a DataFrame on hit, None on miss or error.
    """
    raw = _sb_download(client, f"qc-cache/{pdf_name}.json")
    if raw is None:
        return None
    try:
        return pd.read_json(io.BytesIO(raw), orient="records")
    except Exception:
        return None


def _sb_save_qc_cache(client, pdf_name: str, df: pd.DataFrame) -> bool:
    """Persist QC extraction result to Supabase (qc-cache/{pdf_name}.json).

    Returns True on success, False on failure.
    """
    try:
        json_bytes = df.to_json(orient="records").encode("utf-8")
        return _sb_upload(client, f"qc-cache/{pdf_name}.json", json_bytes, "application/json") is True
    except Exception:
        return False


def _sb_delete_qc_cache(client, pdf_name: str) -> bool:
    """Delete the cached extraction for pdf_name. Returns True on success."""
    return _sb_delete(client, f"qc-cache/{pdf_name}.json") is True
