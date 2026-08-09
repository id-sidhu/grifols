"""Barcode parsing, freeze-tracker validation and the PC-Blut row builder."""

import datetime
import math
import re
from typing import Dict, List, Optional, Set


_DP_BARCODE_RE = re.compile(r"^=C0703(\d{2})(\d{6})00$")

# PC-Blut pipe-separated column indices per role
_DP_COLS: Dict[str, Dict[str, int]] = {
    "supervisor": {
        "product": 39, "donation_num": 47, "donor_id": 38, "raw_date": 48,
        "start5": 76, "quarantine": 42, "prod_vol": 50, "vol": 51,
    },
    "staff": {
        "product": 9, "donation_num": 17, "donor_id": 8, "raw_date": 18,
        "start5": 46, "quarantine": 12, "prod_vol": 20, "vol": 21,
    },
}


def dp_parse_barcodes(raw_text: str) -> List[str]:
    """Convert scanned barcode lines into donation IDs.

    Lines matching ``=C0703XXYYYYYY00`` become ``FXX-YYYYYY``; any other
    non-empty line is kept as-is.
    """
    out: List[str] = []
    for line in (raw_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _DP_BARCODE_RE.match(line)
        out.append(f"F{m.group(1)}-{m.group(2)}" if m else line)
    return out


def dp_iso_week(date_str: str) -> str:
    """ISO week number for a DD-MM-YYYY / DD.MM.YYYY date string."""
    parts = re.split(r"[-.]", date_str.strip()) if date_str else []
    if len(parts) != 3:
        return ""
    try:
        return str(
            datetime.date(int(parts[2]), int(parts[1]), int(parts[0])).isocalendar()[1]
        )
    except (ValueError, OverflowError):
        return ""


def _dp_parse_float(s: str) -> float:
    """Mimic JS ``parseFloat``: leading numeric prefix or NaN."""
    m = re.match(r"^[+-]?(\d+\.?\d*|\.\d+)", s.strip())
    return float(m.group(0)) if m else float("nan")


def dp_parse_time(raw: str) -> Optional[int]:
    """Parse ``H:MM`` / ``H.MM`` (single-digit minutes padded to tens, e.g.
    ``9:3`` → ``9:30``) into minutes since midnight, or None if invalid."""
    s = raw.strip().replace(".", ":", 1)
    s = re.sub(r"^(\d{1,2}):(\d)$", lambda m: f"{m.group(1)}:{m.group(2)}0", s)
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    h, mn = int(m.group(1)), int(m.group(2))
    if h > 23 or mn > 59:
        return None
    return h * 60 + mn


def dp_is_zero_cycle(raw: str) -> bool:
    """True for placeholder lines like ``0``, ``00:00``, ``0.0``."""
    return re.fullmatch(r"0+([.:][0]*)?", raw.strip()) is not None


def dp_fmt_time(total_min: int) -> str:
    return f"{(total_min // 60) % 24:02d}:{total_min % 60:02d}"


def dp_validate_freeze_tracker(
    times: List[str], max_units: int, window_min: int, offset_hrs: int
) -> Dict:
    """Validate freezer loading cycles from START 5 timestamps.

    Groups timestamps into loading cycles of at most ``max_units`` units per
    ``window_min``-minute window, enforcing a cooldown between cycles.
    ``offset_hrs`` shifts all timestamps (e.g. -1 for timezone correction).

    Returns a dict with ``status`` ('empty' | 'ok' | 'error'), ``message``,
    ``rows`` (list of (time_str, count)), ``total``, ``cycle_errors``,
    ``format_errors`` and ``skipped``.
    """
    result: Dict = {
        "status": "empty", "message": "", "rows": [], "total": 0,
        "cycle_errors": [], "format_errors": [], "skipped": 0,
    }
    if not times:
        result["message"] = "No timestamps found."
        return result

    offset = offset_hrs * 60
    count_map: Dict[int, int] = {}
    bad_lines: List[str] = []
    skipped = 0
    for line in times:
        if dp_is_zero_cycle(line):
            skipped += 1
            continue
        raw = dp_parse_time(line)
        if raw is None:
            bad_lines.append(line)
            continue
        t = ((raw + offset) % 1440 + 1440) % 1440
        count_map[t] = count_map.get(t, 0) + 1
    result["skipped"] = skipped

    sorted_entries = sorted(count_map.items())
    if not sorted_entries:
        result["message"] = (
            f"Skipped {skipped} zero-cycle lines. No data."
            if skipped else "No valid timestamps."
        )
        return result

    result["rows"] = [(dp_fmt_time(t), c) for t, c in sorted_entries]
    result["total"] = sum(c for _, c in sorted_entries)

    has_errors = bool(bad_lines)
    cycles: List[Dict] = []
    cur: Optional[Dict] = None
    running_total = 0
    cycle_window_end: Optional[int] = None
    cooldown_until: Optional[int] = None

    def _close_cycle() -> None:
        nonlocal cur
        if cur is not None:
            cycles.append(cur)
            cur = None

    def _open_cycle(t: int, too_early: bool, prev_cooldown: Optional[int]) -> None:
        nonlocal cur
        _close_cycle()
        cur = {
            "index": len(cycles) + 1, "start": t, "window_end": t + window_min,
            "too_early": too_early, "prev_cooldown": prev_cooldown,
            "over_capacity": False, "over_at": None, "over_total": None,
            "next_allowed": None,
        }

    for t, count in sorted_entries:
        too_early = cooldown_until is not None and t < cooldown_until
        if too_early:
            has_errors = True

        need_new_cycle = (
            cur is None or too_early
            or (cooldown_until is not None and t >= cooldown_until)
        )
        if need_new_cycle:
            if not too_early and cooldown_until is not None and t >= cooldown_until:
                running_total = 0
            _open_cycle(t, too_early, cooldown_until)
            if too_early:
                running_total = 0
                cooldown_until = None
            cycle_window_end = t + window_min
        else:
            if cycle_window_end is not None and t >= cycle_window_end:
                _open_cycle(t, False, None)
                cycle_window_end = t + window_min

        if running_total + count > max_units:
            has_errors = True
            cur["over_capacity"] = True
            cur["over_at"] = t
            cur["over_total"] = running_total + count
            cur["next_allowed"] = t + window_min
            cooldown_until = t + window_min
            running_total = 0
            _close_cycle()
        else:
            running_total += count
            if running_total == max_units:
                cooldown_until = t + window_min
                running_total = 0
                _close_cycle()
                cycle_window_end = None
    _close_cycle()

    for cyc in cycles:
        if not (cyc["too_early"] or cyc["over_capacity"]):
            continue
        header = (
            f"Cycle #{cyc['index']} "
            f"({dp_fmt_time(cyc['start'])} – {dp_fmt_time(cyc['window_end'] - 1)})"
        )
        if cyc["over_capacity"]:
            detail = (
                f"Capacity exceeded at {dp_fmt_time(cyc['over_at'])}: "
                f"{cyc['over_total']}/{max_units} units. "
                f"Next allowed: {dp_fmt_time(cyc['next_allowed'])}."
            )
        else:
            detail = (
                f"Started too early — cooldown active until "
                f"{dp_fmt_time(cyc['prev_cooldown'])}."
            )
        result["cycle_errors"].append(f"{header}: {detail}")

    result["format_errors"] = [
        f'Format error: "{bl}" was skipped.' for bl in bad_lines
    ]
    result["status"] = "error" if has_errors else "ok"
    result["message"] = (
        "Validation failed — see errors below."
        if has_errors else "All loading cycles are valid."
    )
    return result


def dp_process_pc_blut(
    pc_blut_raw: str,
    vmt_list: List[str],
    ser_list: List[str],
    absc_list: List[str],
    role: str,
) -> Dict:
    """Process pasted PC-Blut rows against the scanned barcode lists.

    Returns a dict with ``excel_rows`` (tab-separated), ``start5_values``,
    ``missing`` (VMT-NAT IDs absent from PC-Blut), ``comp_text``,
    ``rejected_units`` and ``incomplete_units``.
    """
    cols = _DP_COLS[role]
    min_cols = max(cols.values()) + 1
    vmt_set, ser_set, absc_set = set(vmt_list), set(ser_list), set(absc_list)

    excel_rows: List[str] = []
    start5_values: List[str] = []
    processed: Set[str] = set()
    comp_date = ""
    rejected_units: List[str] = []
    incomplete_units: List[str] = []

    for line in pc_blut_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        columns = line.split("|")
        if len(columns) < min_cols:
            continue

        donor_id = columns[cols["donor_id"]].strip()
        product = columns[cols["product"]].strip()
        quarantine_raw = columns[cols["quarantine"]].strip()
        donation_num = columns[cols["donation_num"]].strip()
        raw_date = columns[cols["raw_date"]].strip()
        prod_vol = _dp_parse_float(columns[cols["prod_vol"]])
        vol = _dp_parse_float(columns[cols["vol"]])
        start5_value = columns[cols["start5"]].strip()

        if product == "No bleed":
            continue
        processed.add(donation_num)

        if not comp_date and raw_date:
            comp_date = raw_date.replace("-", ".")

        clean_reason = quarantine_raw.replace("*", "").strip()
        is_rejected = False
        if clean_reason:
            rejected_units.append(f"{donation_num}/ {donor_id} ({clean_reason})")
            is_rejected = True

        if (
            not is_rejected
            and not math.isnan(prod_vol)
            and not math.isnan(vol)
            and prod_vol - vol >= 5
        ):
            incomplete_units.append(f"{donation_num}/ {donor_id}")

        in_vmt = donation_num in vmt_set
        in_ser = donation_num in ser_set
        in_absc = donation_num in absc_set

        if product == "Test sample":
            donor_status = "SO (16 week)"
            status = "SO (16 week)" if (in_vmt and in_ser) else "Rejected"
        elif product == "SP/PPH":
            if not in_vmt:
                donor_status = "Q"
                status = "Rejected"
            else:
                status = "Quarantine"
                if in_ser and in_absc:
                    donor_status = "1"
                elif in_ser:
                    donor_status = "16"
                else:
                    donor_status = "Q"
        else:
            donor_status = "Q"
            status = "Quarantine"

        formatted_date = raw_date.replace("-", ".")
        iso_week = dp_iso_week(raw_date)
        excel_rows.append(
            f"{donation_num}\t{donor_id}\t{donor_status}\t{formatted_date}\t"
            f"{iso_week}\t{status}\t{clean_reason}"
        )
        start5_values.append(start5_value)

    comp_text = (
        f"Hello! Please see Compensation details for {comp_date or 'Unknown Date'}:\n"
        f"Total Rejected: {len(rejected_units)}\n\n"
    )
    if rejected_units:
        comp_text += "\n".join(rejected_units) + "\n"
    comp_text += f"\nTotal Incompletes: {len(incomplete_units)}\n\n"
    if incomplete_units:
        comp_text += "\n".join(incomplete_units) + "\n"

    return {
        "excel_rows": excel_rows,
        "start5_values": start5_values,
        "missing": [num for num in vmt_list if num not in processed],
        "comp_text": comp_text.strip(),
        "rejected_units": rejected_units,
        "incomplete_units": incomplete_units,
    }
