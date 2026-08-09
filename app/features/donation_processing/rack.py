"""Small rack visualisations for the Donation Processing dashboard."""

from typing import List


_DP_RACK_CSS = """
<style>
  .dp-rack { display: inline-grid; gap: 3px; background: #f6f8fa; padding: 8px; border-radius: 8px; border: 1px solid #d0d7de; margin-bottom: 14px; }
  .dp-cell { background: #eef1f4; border: 1px solid rgba(0,0,0,0.08); width: 36px; height: 34px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #6b7280; box-sizing: border-box; border-radius: 4px; }
  .dp-filled { background: #60a5fa; border-color: #3b82f6; color: #1e3a8a; font-weight: 700; }
  .dp-rack-title { font-weight: 700; font-size: 13px; color: #0056b3; margin: 6px 0; }
</style>
"""


def dp_build_rack_html(title: str, data_list: List[str], rows: int, cols: int) -> str:
    """Render Pavlo-style rack grids (filled top-to-bottom, row-major)."""
    capacity = rows * cols
    num_racks = max(1, -(-len(data_list) // capacity))
    blocks: List[str] = []
    for r in range(num_racks):
        cells: List[str] = []
        for i in range(capacity):
            idx = r * capacity + i
            if idx < len(data_list):
                full = data_list[idx]
                parts = full.split("-")
                text = parts[1] if len(parts) > 1 and parts[1] else full
                cells.append(
                    f'<div class="dp-cell dp-filled" title="{full}">{text}</div>'
                )
            else:
                cells.append('<div class="dp-cell"></div>')
        blocks.append(
            f'<div class="dp-rack-title">{title} (Rack {r + 1} of {num_racks})</div>'
            f'<div class="dp-rack" style="grid-template-columns: repeat({cols}, 36px);">'
            + "".join(cells)
            + "</div>"
        )
    return "".join(blocks)
