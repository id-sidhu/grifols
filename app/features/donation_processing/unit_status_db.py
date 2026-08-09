"""Unit Status CSV "database" helpers.

The Donation Processing dashboard produces tab-separated rows that map 1:1 onto
the columns of the Unit Status CSV.  This module owns everything needed to get
those rows *into* the stored CSV safely:

* :func:`parse_db_year` / :func:`select_db_file` — enforce the one-file-per-year
  storage convention (``Unit Status(UNIT STATUS 2026) anything.csv``).
* :func:`resolve_headers` — match the seven logical fields onto the CSV's real
  column names at runtime, so nothing is hardcoded to a column order.
* :func:`build_existing_index` — pull donation IDs and dates out of the stored
  file for validation.
* :func:`preflight` — refuse unsafe appends (duplicate IDs, wrong year) and
  flag ones needing acknowledgement (missing prior day, backfill).
* :func:`append_records` — append rows **without rewriting the existing bytes**,
  preserving the original encoding, line terminator, quoting and column order.

Every function here is pure and Streamlit-free so it can be unit tested.
"""

from __future__ import annotations

import csv
import datetime
import io
import re
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Set


__all__ = [
    "FIELDS",
    "FIELD_LABELS",
    "DbFileSelection",
    "CsvShape",
    "Preflight",
    "parse_db_year",
    "select_db_file",
    "resolve_headers",
    "read_csv_shape",
    "parse_dp_rows",
    "build_existing_index",
    "preflight",
    "append_records",
    "archive_path",
]

# ---------------------------------------------------------------------------
# Logical fields, in the order dp_process_pc_blut emits them
# ---------------------------------------------------------------------------

FIELDS = (
    "donation_num",
    "donor_id",
    "donor_status",
    "date",
    "iso_week",
    "status",
    "reason",
)

FIELD_LABELS: Dict[str, str] = {
    "donation_num": "Donation #",
    "donor_id": "Donor ID",
    "donor_status": "Donor Status",
    "date": "Donation Date",
    "iso_week": "Week of year",
    "status": "Status",
    "reason": "Reason / Notes",
}


# ---------------------------------------------------------------------------
# Filename convention
# ---------------------------------------------------------------------------

# "Unit Status(UNIT STATUS 2026) [anything].csv"
_CANON_RE = re.compile(
    r"unit\s*status\s*\(\s*unit\s*status\s*(\d{4})\s*\)", re.IGNORECASE
)


def parse_db_year(filename: str) -> Optional[int]:
    """Return the year encoded in a canonical unit status filename.

    ``None`` when the name is not a ``.csv`` following the convention.

    >>> parse_db_year("Unit Status(UNIT STATUS 2026) master.csv")
    2026
    >>> parse_db_year("random_export.csv") is None
    True
    """
    if not filename.lower().endswith(".csv"):
        return None
    m = _CANON_RE.search(filename)
    return int(m.group(1)) if m else None


class DbFileSelection(NamedTuple):
    """Outcome of picking the canonical DB file for a given year.

    Attributes
    ----------
    chosen:
        The single file for ``year``, or ``None`` if absent/ambiguous.
    duplicates:
        Maps year -> files, for every year holding more than one file.  Any
        entry here violates the one-file-per-year rule and should be surfaced
        to the user as "delete the unnecessary files".
    non_conforming:
        ``.csv`` files whose names do not follow the convention.
    other_years:
        Maps year -> files for years other than ``year``.  These are the
        legitimate long-run archive and are not a problem.
    """

    chosen: Optional[str]
    duplicates: Dict[int, List[str]]
    non_conforming: List[str]
    other_years: Dict[int, List[str]]

    @property
    def is_ambiguous(self) -> bool:
        """True when the target year cannot be resolved to exactly one file."""
        return self.chosen is None


def select_db_file(names: Sequence[str], year: int) -> DbFileSelection:
    """Pick the one canonical unit status CSV for ``year``.

    Files belonging to *other* years are allowed (one per year, for long-run
    use).  More than one file for any single year is reported in
    ``duplicates``.
    """
    csvs = [n for n in names if n.lower().endswith(".csv")]

    by_year: Dict[int, List[str]] = {}
    non_conforming: List[str] = []
    for name in csvs:
        y = parse_db_year(name)
        if y is None:
            non_conforming.append(name)
        else:
            by_year.setdefault(y, []).append(name)

    for y in by_year:
        by_year[y].sort()

    duplicates = {y: files for y, files in by_year.items() if len(files) > 1}
    target = by_year.get(year, [])
    chosen = target[0] if len(target) == 1 else None
    other_years = {y: files for y, files in by_year.items() if y != year}

    return DbFileSelection(
        chosen=chosen,
        duplicates=duplicates,
        non_conforming=sorted(non_conforming),
        other_years=other_years,
    )


# ---------------------------------------------------------------------------
# Header resolution
# ---------------------------------------------------------------------------

# Aliases are matched against normalised (lowercased, whitespace-collapsed)
# headers.  Order of _RESOLVE_ORDER matters: "Donor Status" must claim its
# column before the looser "status" aliases get a chance at it.
_ALIASES: Dict[str, Sequence[str]] = {
    "donation_num": (
        "donation #", "donation#", "donation no", "donation no.",
        "donation number", "donation num", "don #", "don. #",
    ),
    "donor_id": (
        "donor id", "donor #", "donor#", "donor no", "donor no.",
        "donor number", "donorid",
    ),
    "donor_status": ("donor status", "donorstatus", "donor stat"),
    "date": ("donation date", "don. date", "don date", "donationdate", "date"),
    "iso_week": (
        "week", "iso week", "week of year", "week of the year", "week #",
        "week no", "week no.", "week number", "wk", "cw", "calendar week",
    ),
    "status": ("status", "unit status", "unitstatus"),
    "reason": (
        "reasons/ notes", "reasons/notes", "reasons / notes", "reason/ notes",
        "reasons", "reason", "notes", "comments", "comment",
    ),
}

_RESOLVE_ORDER = (
    "donation_num",
    "donor_id",
    "donor_status",
    "date",
    "iso_week",
    "status",
    "reason",
)


def _norm(value: object) -> str:
    """Normalise a header cell for comparison."""
    text = str(value or "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def resolve_headers(header: Sequence[str]) -> Dict[str, Optional[str]]:
    """Map each logical field onto an actual column name from ``header``.

    Two passes: exact alias match first, then "alias appears inside the
    header text" for messier real-world names.  Each column is claimed at
    most once.  Fields that cannot be resolved map to ``None`` so the caller
    can ask the user to pick manually.
    """
    normalised = [_norm(h) for h in header]
    mapping: Dict[str, Optional[str]] = {}
    claimed: Set[int] = set()

    # Pass 1 — exact match on an alias.
    for field in _RESOLVE_ORDER:
        aliases = _ALIASES[field]
        for idx, norm_header in enumerate(normalised):
            if idx in claimed or not norm_header:
                continue
            if norm_header in aliases:
                mapping[field] = header[idx]
                claimed.add(idx)
                break

    # Pass 2 — alias contained within the header (e.g. "Week Of Year (ISO)").
    for field in _RESOLVE_ORDER:
        if field in mapping:
            continue
        aliases = _ALIASES[field]
        for idx, norm_header in enumerate(normalised):
            if idx in claimed or len(norm_header) < 3:
                continue
            if any(alias in norm_header for alias in aliases if len(alias) >= 3):
                mapping[field] = header[idx]
                claimed.add(idx)
                break

    for field in FIELDS:
        mapping.setdefault(field, None)
    return mapping


def unresolved_fields(mapping: Dict[str, Optional[str]]) -> List[str]:
    """Logical fields with no column assigned."""
    return [f for f in FIELDS if not mapping.get(f)]


# ---------------------------------------------------------------------------
# Reading the stored CSV
# ---------------------------------------------------------------------------

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


class CsvShape(NamedTuple):
    """Everything needed to append to a CSV byte-for-byte compatibly.

    ``encoding`` is the codec to use when encoding *appended* lines.  It is
    deliberately never ``utf-8-sig``: that codec emits a BOM for whatever it
    encodes, which mid-file would corrupt the CSV.  A leading BOM in the
    original file is recorded in ``has_bom`` and left alone.
    """

    header: List[str]
    encoding: str
    newline: str
    delimiter: str
    quotechar: str
    has_trailing_newline: bool
    has_bom: bool
    text: str


_UTF8_BOM = b"\xef\xbb\xbf"


def read_csv_shape(raw: bytes) -> CsvShape:
    """Inspect raw CSV bytes: header, encoding, line terminator, dialect."""
    if not raw:
        raise ValueError("Unit status file is empty.")

    has_bom = raw.startswith(_UTF8_BOM)

    text = None
    encoding = "utf-8"
    for candidate in _ENCODINGS:
        try:
            text = raw.decode(candidate)
            # utf-8-sig only differs from utf-8 by stripping a leading BOM,
            # so appended lines are always written as plain utf-8.
            encoding = "utf-8" if candidate == "utf-8-sig" else candidate
            break
        except UnicodeDecodeError:
            continue
    if text is None:  # pragma: no cover - latin-1 never fails
        raise ValueError("Could not decode the unit status file.")

    newline = "\r\n" if "\r\n" in text else "\n"

    first_line = text.split("\n", 1)[0]
    delimiter = ","
    try:
        sniffed = csv.Sniffer().sniff(first_line, delimiters=",;\t|")
        delimiter = sniffed.delimiter
    except csv.Error:
        pass

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = [str(cell) for cell in next(reader)]
    except StopIteration:
        raise ValueError("Unit status file has no header row.")

    if header and header[0].startswith("\ufeff"):
        header[0] = header[0].lstrip("\ufeff")

    return CsvShape(
        header=header,
        encoding=encoding,
        newline=newline,
        delimiter=delimiter,
        quotechar='"',
        has_trailing_newline=text.endswith(("\n", "\r")),
        has_bom=has_bom,
        text=text,
    )


def parse_dp_rows(excel_rows: Sequence[str]) -> List[Dict[str, str]]:
    """Split ``dp_process_pc_blut``'s tab-separated rows into field dicts.

    Short rows are padded rather than dropped, so a malformed row surfaces as
    a validation blocker later instead of vanishing silently.
    """
    records: List[Dict[str, str]] = []
    for raw in excel_rows:
        parts = [p.strip() for p in str(raw).split("\t")]
        if len(parts) < len(FIELDS):
            parts += [""] * (len(FIELDS) - len(parts))
        records.append(dict(zip(FIELDS, parts[: len(FIELDS)])))
    return records


def build_existing_index(
    shape: CsvShape,
    mapping: Dict[str, Optional[str]],
    date_parser: Callable[[object], Optional[datetime.date]],
) -> Dict[str, object]:
    """Extract donation IDs and parsed dates from the stored CSV.

    Returns ``{"ids": set[str], "dates": list[date], "rows": int}``.
    """
    id_col = mapping.get("donation_num")
    date_col = mapping.get("date")

    ids: Set[str] = set()
    dates: List[datetime.date] = []
    rows = 0

    reader = csv.DictReader(
        io.StringIO(shape.text),
        delimiter=shape.delimiter,
    )
    for row in reader:
        rows += 1
        if id_col:
            value = (row.get(id_col) or "").strip()
            if value:
                ids.add(value)
        if date_col:
            parsed = date_parser((row.get(date_col) or "").strip())
            if parsed is not None:
                dates.append(parsed)

    return {"ids": ids, "dates": dates, "rows": rows}


# ---------------------------------------------------------------------------
# Preflight validation
# ---------------------------------------------------------------------------


class Preflight(NamedTuple):
    """Verdict on a proposed append.

    ``blockers`` are hard stops — data integrity problems that must not be
    overridden.  ``warnings`` are advisory.  ``requires_gap_ack`` means the
    batch skips one or more calendar days and the user must explicitly
    confirm (e.g. the centre was closed) before the write is allowed.
    """

    ok: bool
    blockers: List[str]
    warnings: List[str]
    requires_gap_ack: bool
    overlap: List[str]
    new_dates: List[datetime.date]
    latest_existing: Optional[datetime.date]
    gap_dates: List[datetime.date]


def _fmt(day: datetime.date) -> str:
    return day.strftime("%d.%m.%Y")


def _sample(items: Sequence[str], limit: int = 6) -> str:
    head = list(items[:limit])
    suffix = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return ", ".join(head) + suffix


def preflight(
    records: Sequence[Dict[str, str]],
    existing_ids: Set[str],
    existing_dates: Sequence[datetime.date],
    db_year: int,
    date_parser: Callable[[object], Optional[datetime.date]],
) -> Preflight:
    """Validate a proposed append against the stored file."""
    blockers: List[str] = []
    warnings: List[str] = []

    if not records:
        return Preflight(
            ok=False,
            blockers=["Nothing to append — no rows were parsed from PC-Blut."],
            warnings=[],
            requires_gap_ack=False,
            overlap=[],
            new_dates=[],
            latest_existing=max(existing_dates) if existing_dates else None,
            gap_dates=[],
        )

    # --- donation IDs -----------------------------------------------------
    incoming = [r.get("donation_num", "").strip() for r in records]
    blank_ids = sum(1 for i in incoming if not i)
    if blank_ids:
        blockers.append(f"{blank_ids} row(s) have a blank Donation #.")

    seen: Set[str] = set()
    dup_in_batch: List[str] = []
    for donation_id in incoming:
        if donation_id and donation_id in seen:
            dup_in_batch.append(donation_id)
        seen.add(donation_id)
    if dup_in_batch:
        blockers.append(
            f"{len(dup_in_batch)} duplicate Donation # within this batch: "
            f"{_sample(sorted(set(dup_in_batch)))}"
        )

    overlap = sorted({i for i in incoming if i} & existing_ids)
    if overlap:
        blockers.append(
            f"{len(overlap)} Donation # already exist in the database — this "
            f"batch has already been committed (fully or partly): "
            f"{_sample(overlap)}"
        )

    # --- dates ------------------------------------------------------------
    parsed = [date_parser(r.get("date", "")) for r in records]
    unparsed = sum(1 for p in parsed if p is None)
    if unparsed:
        blockers.append(
            f"{unparsed} row(s) have an unreadable Donation Date — expected "
            f"DD.MM.YYYY."
        )

    new_dates = sorted({p for p in parsed if p is not None})
    if not new_dates:
        blockers.append("No usable donation dates in this batch.")

    wrong_year = sorted({d.year for d in new_dates} - {db_year})
    if wrong_year:
        blockers.append(
            f"Batch contains dates from {', '.join(str(y) for y in wrong_year)} "
            f"but the target file is the {db_year} database. Load the correct "
            f"year's file."
        )

    if len(new_dates) > 1:
        warnings.append(
            f"Batch spans {len(new_dates)} donation dates "
            f"({_fmt(new_dates[0])} – {_fmt(new_dates[-1])})."
        )

    # --- prior-day continuity --------------------------------------------
    latest_existing = max(existing_dates) if existing_dates else None
    gap_dates: List[datetime.date] = []
    requires_gap_ack = False

    if latest_existing is None:
        warnings.append(
            "The database has no readable dates yet, so the prior-day check "
            "was skipped."
        )
    elif new_dates:
        earliest_new = new_dates[0]
        delta = (earliest_new - latest_existing).days
        if delta == 1:
            pass  # clean continuation
        elif delta == 0:
            warnings.append(
                f"{_fmt(latest_existing)} is already the latest date in the "
                f"database — this looks like a same-day top-up."
            )
        elif delta < 0:
            warnings.append(
                f"Out-of-order backfill: this batch starts "
                f"{_fmt(earliest_new)} but the database already runs to "
                f"{_fmt(latest_existing)}."
            )
        else:
            gap_dates = [
                latest_existing + datetime.timedelta(days=n)
                for n in range(1, delta)
            ]
            requires_gap_ack = True
            warnings.append(
                f"Missing prior day(s): the database ends "
                f"{_fmt(latest_existing)} and this batch starts "
                f"{_fmt(earliest_new)}. No data for "
                f"{', '.join(_fmt(d) for d in gap_dates)}."
            )

    return Preflight(
        ok=not blockers,
        blockers=blockers,
        warnings=warnings,
        requires_gap_ack=requires_gap_ack,
        overlap=overlap,
        new_dates=new_dates,
        latest_existing=latest_existing,
        gap_dates=gap_dates,
    )


# ---------------------------------------------------------------------------
# Appending
# ---------------------------------------------------------------------------


def append_records(
    raw: bytes,
    records: Sequence[Dict[str, str]],
    mapping: Dict[str, Optional[str]],
) -> bytes:
    """Append ``records`` to the CSV in ``raw``, returning the new bytes.

    The existing content is copied through untouched — only new lines are
    added.  Column order, encoding, delimiter and line terminator are taken
    from the original file, so unmapped columns are written as empty cells in
    their correct positions.
    """
    shape = read_csv_shape(raw)

    # Column name -> field, for the columns we actually fill.
    column_to_field: Dict[str, str] = {}
    for field, column in mapping.items():
        if column:
            column_to_field[column] = field

    buffer = io.StringIO()
    writer = csv.writer(
        buffer,
        delimiter=shape.delimiter,
        quotechar=shape.quotechar,
        lineterminator=shape.newline,
        quoting=csv.QUOTE_MINIMAL,
    )
    for record in records:
        writer.writerow(
            [
                record.get(column_to_field[column], "")
                if column in column_to_field
                else ""
                for column in shape.header
            ]
        )

    addition = buffer.getvalue()
    prefix = "" if shape.has_trailing_newline else shape.newline
    return raw + (prefix + addition).encode(shape.encoding)


def archive_path(folder: str, filename: str, now: Optional[datetime.datetime] = None) -> str:
    """Timestamped snapshot path, e.g. ``unit-status/_archive/x.20260725-143012.csv``.

    Taking a snapshot before every append keeps the write reversible and
    leaves an audit trail, since Supabase uploads are upsert-in-place.
    """
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")
    stem, _, ext = filename.rpartition(".")
    if not stem:
        stem, ext = filename, "csv"
    return f"{folder}/_archive/{stem}.{stamp}.{ext}"
