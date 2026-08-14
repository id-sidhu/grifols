"""ReportLab rendering of the visual-inspection label sheets."""

import datetime
import io
from typing import Dict, List, Optional


def generate_vi_labels_pdf(
    groups: List[Dict],
    tomorrow: datetime.date,
) -> bytes:
    """Render Visual Inspection label groups to a PDF.

    Each A4 page contains **two identical copies** of the same group label
    (top half and bottom half), so one sheet can be cut in two and one copy
    kept with the unit and one with the paperwork.

    * **Complete group** label: ``DD.MM.YYYY – DD.MM.YYYY`` date range and
      ``FIRST_ID – LAST_ID`` ID range.
    * **Last partial group** label: ``LATEST_DATE – TOMORROW`` date range
      and ``FIRST_ID –`` (end ID omitted because the group is open-ended).

    Parameters
    ----------
    groups : list of dict
        As returned by :func:`build_vi_label_groups`.
    tomorrow : datetime.date
        End date used for the last partial group label.

    Returns
    -------
    bytes
        Raw PDF content suitable for ``st.download_button``.
    """
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfbase.pdfmetrics import stringWidth
    except ImportError as exc:
        raise ImportError(
            "reportlab is required to generate PDFs. "
            "Install it with:  pip install reportlab"
        ) from exc

    buf = io.BytesIO()
    page_w, page_h = A4          # ~595 × 842 pt
    margin_x = 12 * mm
    margin_y = 14 * mm
    gap = 8 * mm

    label_w = page_w - 2 * margin_x
    label_h = (page_h - 2 * margin_y - gap) / 2

    c = rl_canvas.Canvas(buf, pagesize=A4)

    def _fit_size(text: str, font: str, max_w: float, start: int = 60) -> int:
        sz = start
        while sz > 8:
            if stringWidth(text, font, sz) <= max_w:
                return sz
            sz -= 1
        return sz

    def _draw_label(
        x: float, y: float, date_str: str,
        id_left: str, id_right: Optional[str] = None,
        label_num: Optional[int] = None,
    ) -> None:
        inner_w = label_w - 10 * mm
        cx = x + label_w / 2

        # Border
        c.setStrokeColorRGB(0.55, 0.55, 0.55)
        c.setLineWidth(0.8)
        c.rect(x, y, label_w, label_h)

        # Small label number in top-right corner
        if label_num is not None:
            c.setFont("Helvetica", 14)
            c.setFillColorRGB(0.55, 0.55, 0.55)
            num_str = str(label_num)
            num_w = stringWidth(num_str, "Helvetica", 14)
            c.drawString(x + label_w - num_w - 3 * mm, y + label_h - 4 * mm, num_str)
            c.setFillColorRGB(0, 0, 0)

        # Quarantine header — bold, centered, evenly spaced above the date
        header_str = "QUARANTINE - PACKAGED PLASMA"
        header_sz = _fit_size(header_str, "Helvetica-Bold", inner_w, start=26)
        c.setFont("Helvetica-Bold", header_sz)
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(cx, y + label_h * 0.78, header_str)

        # Date range — normal font, large
        date_sz = _fit_size(date_str, "Helvetica", inner_w, start=60)
        c.setFont("Helvetica", date_sz)
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(cx, y + label_h * 0.50, date_str)

        # ID range — all normal font, same size; last 3 digits bold with 1 space before them
        sizing_str = (
            f"{id_left}    -    {id_right}" if id_right else f"{id_left}   -"
        )
        id_sz = _fit_size(sizing_str, "Helvetica", inner_w, start=28)
        space_w = stringWidth(" ", "Helvetica", id_sz)

        def _id_w(sid: str) -> float:
            pre = sid[:-3] if len(sid) > 3 else ""
            last3 = sid[-3:] if len(sid) >= 3 else sid
            return (
                stringWidth(pre, "Helvetica", id_sz)
                + space_w
                + stringWidth(last3, "Helvetica-Bold", id_sz)
            )

        gap_w = stringWidth("    ", "Helvetica", id_sz)
        dash_w = stringWidth("-", "Helvetica", id_sz)
        left_w = _id_w(id_left)
        right_w = _id_w(id_right) if id_right else 0

        total_w = left_w + gap_w + dash_w + (gap_w + right_w if id_right else 0)
        sx = cx - total_w / 2
        base_y = y + label_h * 0.22

        def _draw_id(draw_x: float, sid: str) -> float:
            pre = sid[:-3] if len(sid) > 3 else ""
            last3 = sid[-3:] if len(sid) >= 3 else sid
            if pre:
                c.setFont("Helvetica", id_sz)
                c.drawString(draw_x, base_y, pre)
                draw_x += stringWidth(pre, "Helvetica", id_sz)
            draw_x += space_w
            c.setFont("Helvetica-Bold", id_sz)
            c.drawString(draw_x, base_y, last3)
            return draw_x + stringWidth(last3, "Helvetica-Bold", id_sz)

        cur_x = _draw_id(sx, id_left)
        c.setFont("Helvetica", id_sz)
        c.drawString(cur_x + gap_w, base_y, "-")
        if id_right:
            _draw_id(cur_x + gap_w + dash_w + gap_w, id_right)

    def _label_parts(group: Dict):
        """Return (date_str, id_left, id_right) for a group."""
        if group["is_complete"]:
            d_min, d_max = group["date_min"], group["date_max"]
            date_str = (
                f"{d_min.strftime('%d.%m.%Y')} - {d_max.strftime('%d.%m.%Y')}"
                if d_min and d_max
                else (d_min or d_max).strftime("%d.%m.%Y") if (d_min or d_max) else "Date unknown"
            )
            return date_str, group["first_id"], group["last_id"]
        else:
            d_max = group["date_max"]
            date_str = (
                f"{d_max.strftime('%d.%m.%Y')} - {tomorrow.strftime('%d.%m.%Y')}"
                if d_max else f"? - {tomorrow.strftime('%d.%m.%Y')}"
            )
            return date_str, group["first_id"], None

    # Pair labels so cutting all pages in half and stacking gives sequential order:
    # top half of every page = labels 0..half-1, bottom half = labels half..n-1
    n_groups = len(groups)
    half = (n_groups + 1) // 2
    for i in range(half):
        top_ds, top_il, top_ir = _label_parts(groups[i])
        _draw_label(margin_x, margin_y + gap + label_h, top_ds, top_il, top_ir, label_num=i + 1)
        j = i + half
        if j < n_groups:
            bot_ds, bot_il, bot_ir = _label_parts(groups[j])
            _draw_label(margin_x, margin_y, bot_ds, bot_il, bot_ir, label_num=j + 1)
        c.showPage()

    c.save()
    buf.seek(0)
    return buf.read()
