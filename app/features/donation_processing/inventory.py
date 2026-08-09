"""Warehouse balance matcher — "Current Status Of Warehouse Balances".

Two pasted, pipe-separated lists are cross-referenced:

* the **stock list** exported from inventory, where each row carries the item
  name, its lot number and the quantity on hand;
* the **current lot list**, naming the lots actually in use right now.

Every current lot is looked up in the stock list so the operator can see, at a
glance, which lots are running low or are missing from stock entirely.
"""

import math
from typing import Dict, List, NamedTuple

from app.features.donation_processing.logic import _dp_parse_float

#: Column positions inside a pipe-separated inventory stock row.
_STOCK_NAME_COL = 1
_STOCK_LOT_COL = 2
_STOCK_QTY_COL = 8
#: A stock row must have at least this many columns to be usable.
_STOCK_MIN_COLS = 9

#: Status values a matched lot can take.
STATUS_OK = "ok"
STATUS_LOW = "low"
STATUS_MISSING = "missing"


class StockItem(NamedTuple):
    """One lot found in the inventory stock list."""

    name: str
    qty: str


class InventoryRow(NamedTuple):
    """One row of the warehouse balance report.

    ``quantity`` keeps the quantity exactly as the export wrote it (thousands
    separators and all) so the printed report matches the source system.
    """

    item: str
    lot: str
    quantity: str
    status: str

    @property
    def is_problem(self) -> bool:
        """True when this lot needs attention (low stock or not found)."""
        return self.status in (STATUS_LOW, STATUS_MISSING)


def parse_stock_list(text: str) -> Dict[str, StockItem]:
    """Index an inventory stock export by upper-cased lot number.

    Rows with too few columns, or with a blank lot, are ignored.  A repeated
    lot keeps the last row seen, matching the browser tool.
    """
    stock: Dict[str, StockItem] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < _STOCK_MIN_COLS:
            continue
        lot = parts[_STOCK_LOT_COL].strip().upper()
        if not lot:
            continue
        stock[lot] = StockItem(
            name=parts[_STOCK_NAME_COL].strip(),
            qty=parts[_STOCK_QTY_COL].strip(),
        )
    return stock


def _qty_value(raw: str) -> float:
    """Numeric value of a quantity cell; unreadable values count as zero."""
    value = _dp_parse_float(raw.replace(",", ""))
    return 0.0 if math.isnan(value) else value


def match_lots(
    lot_text: str, stock_text: str, min_qty: float
) -> List[InventoryRow]:
    """Look every current lot up in the stock list.

    ``lot_text`` rows may be a bare lot number or a pipe-separated row whose
    first column is the lot.  A lot at or below ``min_qty`` is reported as
    low, and one absent from the stock list as missing.
    """
    stock = parse_stock_list(stock_text)
    rows: List[InventoryRow] = []

    for line in (lot_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        first = line.split("|")[0]
        lot = (first if first else line).strip().upper()
        if not lot:
            continue

        item = stock.get(lot)
        if item is None:
            rows.append(
                InventoryRow(
                    item="Unknown Item",
                    lot=lot,
                    quantity="Not found in stock",
                    status=STATUS_MISSING,
                )
            )
            continue

        low = _qty_value(item.qty) <= min_qty
        rows.append(
            InventoryRow(
                item=item.name,
                lot=lot,
                quantity=f"{item.qty} pcs.",
                status=STATUS_LOW if low else STATUS_OK,
            )
        )
    return rows
