"""Reading uploaded CSV/Excel files.

Workbooks from the floor carry many sheets; :func:`_read_excel_smart` picks the
right one by score, by recency, or by an explicitly requested name.
"""

import datetime
import io
from typing import Dict, Optional, Set

import pandas as pd

from app.shared.ui import _show_error


# ---------------------------------------------------------------------------
# Excel multi-sheet helper
# ---------------------------------------------------------------------------
_EXCEL_PRIORITY_COLS: Set[str] = {
    "Donor ID", "Donation Date", "Status", "Donation #",
    "Donor Status", "Redo Panel",
    "Sample ID", "Pallet", "Comments",
}


def _score_sheet_name(sheets_raw: Dict[str, pd.DataFrame]) -> str:
    """Return the sheet name whose first 10 rows contain the most priority column values."""
    _best, _bscore = None, -1
    for _sn, _sdf in sheets_raw.items():
        _vals: Set[str] = set()
        for _si in range(min(10, len(_sdf))):
            _vals.update(
                str(v).strip() for v in _sdf.iloc[_si] if pd.notna(v) and str(v).strip()
            )
        _sc = len(_vals & _EXCEL_PRIORITY_COLS)
        if _sc > _bscore:
            _bscore = _sc
            _best = _sn
    return _best or next(iter(sheets_raw))


def _read_excel_smart(
    buf, dtype=str, sheet_strategy: str = "score", explicit_sheet: str = ""
) -> pd.DataFrame:
    """Read an Excel file, selecting the best sheet and auto-detecting the header row.

    Two-pass approach:
      1. Read all sheets with ``header=None, dtype=str`` for structure analysis
         (finds the right sheet AND the real header row even when decorative
         title rows appear above the column headers).
      2. Re-read only the chosen sheet with the caller's ``dtype`` and the
         detected ``header=`` offset, preserving original column dtypes.

    explicit_sheet: when non-empty, load that sheet name exactly (case-
                    insensitive).  Falls through to sheet_strategy on miss.
    sheet_strategy:
        "score"       — pick the sheet whose first rows contain the most
                        known column names.
        "latest"      — sort sheet names alphabetically, pick the last one.
        "unit_status" — prefer "UNIT STATUS {year}", then any "UNIT STATUS *",
                        then fall back to score.
    """
    # Slurp bytes once so we can read the buffer twice
    if isinstance(buf, io.BytesIO):
        _xb = buf.getvalue()
    elif isinstance(buf, (bytes, bytearray)):
        _xb = bytes(buf)
    else:
        _xb = buf.read()
        if hasattr(buf, "seek"):
            buf.seek(0)

    # Pass 1 — all sheets, no header assumption, everything as strings
    try:
        _raw: Dict[str, pd.DataFrame] = pd.read_excel(
            io.BytesIO(_xb), sheet_name=None, header=None, dtype=str
        )
    except Exception:
        return pd.DataFrame()
    if not _raw:
        return pd.DataFrame()

    # --- pick target sheet name ---
    _chosen: Optional[str] = None
    if explicit_sheet:
        _tgt = explicit_sheet.strip().upper()
        _chosen = next((n for n in _raw if n.strip().upper() == _tgt), None)

    if _chosen is None:
        if sheet_strategy == "latest":
            _chosen = sorted(_raw)[-1]
        elif sheet_strategy == "unit_status":
            _yr = str(datetime.date.today().year)
            for _n in _raw:
                if _n.strip().upper() == f"UNIT STATUS {_yr}":
                    _chosen = _n
                    break
            if _chosen is None:
                for _n in _raw:
                    if "UNIT STATUS" in _n.strip().upper():
                        _chosen = _n
                        break
            if _chosen is None:
                _chosen = _score_sheet_name(_raw)
        else:
            _chosen = _score_sheet_name(_raw)

    # --- detect real header row (handles decorative title rows above data) ---
    _sheet_raw = _raw[_chosen]
    _hdr_row = 0
    for _ri in range(min(10, len(_sheet_raw))):
        _rv = {
            str(v).strip() for v in _sheet_raw.iloc[_ri]
            if pd.notna(v) and str(v).strip()
        }
        if len(_rv & _EXCEL_PRIORITY_COLS) >= 2:
            _hdr_row = _ri
            break

    # Pass 2 — re-read only the chosen sheet with correct dtype and header offset
    _kw = {"dtype": dtype} if dtype is not None else {}
    try:
        _df = pd.read_excel(
            io.BytesIO(_xb), sheet_name=_chosen, header=_hdr_row, **_kw
        )
        return _df.fillna("")
    except Exception:
        # Fallback: build from the raw string data already in memory
        _data = _sheet_raw.iloc[_hdr_row + 1:].copy().reset_index(drop=True)
        _data.columns = [
            str(v).strip() if (pd.notna(v) and str(v).strip()) else f"_c{j}"
            for j, v in enumerate(_sheet_raw.iloc[_hdr_row])
        ]
        return _data.fillna("")


def read_upload(
    uploaded_file, dtype=None, sheet_strategy: str = "score", explicit_sheet: str = ""
) -> Optional[pd.DataFrame]:
    """Read an uploaded file as a DataFrame, supporting CSV and Excel."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return _read_excel_smart(
                uploaded_file, dtype=dtype,
                sheet_strategy=sheet_strategy, explicit_sheet=explicit_sheet,
            )
        else:
            kwargs = {"dtype": dtype} if dtype else {}
            return pd.read_csv(uploaded_file, **kwargs)
    except Exception as e:
        _show_error(f"Failed to read {uploaded_file.name}: {e}")
        return None
