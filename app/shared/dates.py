"""Donation-date parsing shared by every section that groups data by day."""

import datetime
from typing import Optional

import pandas as pd


# -----------------------------------------------------------------------------
# Donation date parsing helper
def _parse_donation_date(raw) -> Optional[datetime.date]:
    """Parse a raw Donation Date value into a datetime.date for comparison.

    Handles:
    - pandas Timestamp / datetime objects
    - Strings in DD.MM.YYYY, YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY formats
    - Excel serial numbers (numeric, > 1)

    Returns None when the value cannot be parsed.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (pd.Timestamp, datetime.datetime)):
        try:
            return pd.Timestamp(raw).date()
        except Exception:
            pass
    if isinstance(raw, datetime.date):
        return raw
    s = str(raw).strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
                "%d.%m.%y", "%Y%m%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    # Excel serial number (float/int stored as text)
    try:
        n = float(s)
        if n > 1:
            return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(n))).date()
    except (ValueError, OverflowError, OSError):
        pass
    return None
