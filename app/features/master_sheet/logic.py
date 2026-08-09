"""Parsing the Master Sheet workbook into the daily inventory structure."""

from typing import Dict, List

import pandas as pd


def parse_master_sheet(ms_file) -> Dict:
    """Parse a master sheet CSV into structured data keyed by freezer ID.

    The CSV is expected to contain two stacked freezer tables, each starting
    with a row whose first cell begins with "Freezer ID:".  Within each table
    the structure is:
        row 0 – Freezer ID
        row 1 – Maximum Operation Capacity
        row 2 – Date (DD.MM.YYYY) followed by date strings in subsequent columns
        row 3 – Category of plasma units (header, skipped)
        rows 4-11 – 8 data category rows (columns a-h)
        row 12 – Total
        row 13 – Units remained to reach Maximum Operation Capacity

    Returns a dict mapping freezer_id → dict with keys:
        freezer_id, date_row_idx, cat_start_idx, dates, date_col_indices,
        row_labels, categories (label → {date_str: int})
    """
    ms_file.seek(0)
    raw_bytes = ms_file.read()
    for _enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            import io as _io
            raw = pd.read_csv(
                _io.BytesIO(raw_bytes), header=None, dtype=str, encoding=_enc
            ).fillna("")
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        raise ValueError(
            "Could not decode the master sheet CSV. "
            "Try saving it as UTF-8 in Excel (File → Save As → CSV UTF-8)."
        )
    n_rows, n_cols = raw.shape
    result: Dict = {}

    i = 0
    while i < n_rows:
        cell = str(raw.iloc[i, 0]).strip()
        if "Freezer ID:" in cell:
            freezer_id = cell.replace("Freezer ID:", "").strip()

            # Locate the Date row within the next 5 rows
            date_row_idx = None
            for j in range(i + 1, min(i + 6, n_rows)):
                if "Date" in str(raw.iloc[j, 0]) and "DD.MM" in str(raw.iloc[j, 0]):
                    date_row_idx = j
                    break

            if date_row_idx is None:
                i += 1
                continue

            # Collect non-empty date strings and their column positions
            dates: List[str] = []
            date_col_indices: List[int] = []
            for col in range(1, n_cols):
                v = str(raw.iloc[date_row_idx, col]).strip()
                if v and v.lower() != "nan":
                    dates.append(v)
                    date_col_indices.append(col)

            # Category data starts 2 rows after the date row
            # (skipping the "Category of plasma units" header row)
            cat_start_idx = date_row_idx + 2

            row_labels: List[str] = []
            categories: Dict[str, Dict[str, int]] = {}
            for k in range(8):
                ridx = cat_start_idx + k
                if ridx >= n_rows:
                    break
                label = str(raw.iloc[ridx, 0]).strip()
                row_labels.append(label)
                vals: Dict[str, int] = {}
                for date, col in zip(dates, date_col_indices):
                    v = str(raw.iloc[ridx, col]).strip()
                    try:
                        vals[date] = int(float(v)) if v and v.lower() not in ("nan", "") else 0
                    except (ValueError, TypeError):
                        vals[date] = 0
                categories[label] = vals

            result[freezer_id] = {
                "freezer_id": freezer_id,
                "freezer_start": i,
                "date_row_idx": date_row_idx,
                "cat_start_idx": cat_start_idx,
                "dates": dates,
                "date_col_indices": date_col_indices,
                "row_labels": row_labels,
                "categories": categories,
            }
            i = cat_start_idx + 8
        else:
            i += 1

    return result
