"""Rack grid rendering (HTML) and the fullscreen dialog."""

from typing import Dict, List, Optional, Set

import streamlit as st


@st.cache_data(show_spinner=False)
def build_rack_html(
    valid_ids: List[str],
    not_manifest_set: Set[str],
    samples_collected_set: Optional[Set[str]] = None,
    pallet_map: Optional[Dict[str, int]] = None,
    packed_set: Optional[Set[str]] = None,
    digits_to_show: int = 3,
    fill_value: str = "",
    title: str = "Rack Visualization (last three digits)",
    date_range_str: Optional[str] = None,
) -> str:
    """
    18×12 rack using a true 18-column CSS Grid (no spacer columns, no inserted blank IDs).
    Visible separators after columns 6 and 12 are drawn INSIDE the boundary cells
    (inset shadows), so they remain clearly visible regardless of background/theme.
    Cells whose sample IDs appear in ``packed_set`` receive a diagonal hatching
    overlay so they are visually marked as already packed while retaining their
    underlying colour.
    """
    total_positions = 216
    ids_padded = valid_ids[:total_positions] + [""] * max(0, total_positions - len(valid_ids))

    def display_text(sample_id: str) -> str:
        if not sample_id:
            return fill_value
        return sample_id[-digits_to_show:]

    _samples_collected = samples_collected_set or set()
    _pallet_map = pallet_map or {}
    _packed_set = packed_set or set()
    pallets_in_rack = sorted({_pallet_map[sid] for sid in ids_padded if sid and sid in _pallet_map})
    has_sc_in_rack = any(sid in _samples_collected for sid in ids_padded if sid)
    has_not_manifest_in_rack = any(sid in not_manifest_set for sid in ids_padded if sid)
    has_packed_in_rack = any(sid in _packed_set for sid in ids_padded if sid)
    # Build legend HTML as a Python string to avoid 4-space indentation being
    # misinterpreted as a Markdown code block by Streamlit's renderer.
    _legend_parts: List[str] = []
    if pallets_in_rack:
        for _p in pallets_in_rack:
            _legend_parts.append(f'<span class="legend-item"><span class="swatch pallet-{_p}"></span> Pallet {_p}</span>')
    else:
        _legend_parts.append('<span class="legend-item"><span class="swatch present"></span> In unit status</span>')
    if has_not_manifest_in_rack:
        _legend_parts.append('<span class="legend-item"><span class="swatch not-manifest"></span> Not in manifest</span>')
    if has_sc_in_rack:
        _legend_parts.append('<span class="legend-item"><span class="swatch samples-collected"></span> Rejected (samples collected)</span>')
    if has_packed_in_rack:
        _legend_parts.append('<span class="legend-item"><span class="swatch packed-swatch"></span> Already packed</span>')
    _legend_parts.append('<span class="legend-item"><span class="swatch blank"></span> Empty</span>')
    legend_html = "".join(_legend_parts)
    cells_html: List[str] = []
    for sample_id in ids_padded:
        is_blank = not bool(sample_id)
        is_not_manifest = (sample_id in not_manifest_set) if sample_id else False
        is_samples_collected = (sample_id in _samples_collected) if sample_id else False
        is_packed = (sample_id in _packed_set) if sample_id else False
        pallet_num = _pallet_map.get(sample_id) if sample_id else None

        classes = ["rack-cell"]
        if is_blank:
            classes.append("blank")
        elif is_not_manifest:
            classes.append("not-manifest")
        elif is_samples_collected:
            classes.append("samples-collected")
        elif pallet_num is not None:
            classes.append(f"pallet-{pallet_num}")
        else:
            classes.append("present")

        # packed is an additive overlay class – applied on top of the colour class
        if is_packed:
            classes.append("packed")

        tooltip = (sample_id + " [PACKED]") if is_packed and sample_id else (sample_id if sample_id else "Empty")
        cells_html.append(
            f'<div class="{" ".join(classes)}" title="{tooltip}">{display_text(sample_id)}</div>'
        )

    cells_joined = "\n".join(cells_html)

    _date_range_html = (
        f'<span class="rack-date-range">{date_range_str}</span>'
        if date_range_str
        else ""
    )

    html = f"""
<div class="rack-wrap">
  <div class="rack-title">{title}</div>

  <div class="rack-legend">
    {legend_html}
    {_date_range_html}
  </div>

  <div class="rack-grid">
    {cells_joined}
  </div>
</div>

<style>
  /* ── Rack container ────────────────────────────────── */
  .rack-wrap {{
    padding: 14px 14px 16px 14px;
    border: 1px solid #d0d7de;
    border-radius: 12px;
    background: #f6f8fa;
    width: fit-content;
    max-width: 100%;
    overflow-x: auto;
  }}

  .rack-title {{
    font-weight: 800;
    font-size: 15px;
    margin: 0 0 10px 0;
    color: #1f2328;
    letter-spacing: -0.02em;
  }}

  /* ── Legend ─────────────────────────────────────────── */
  .rack-legend {{
    display: flex;
    gap: 16px;
    align-items: center;
    font-size: 12px;
    font-weight: 600;
    color: #424a53;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }}

  .rack-date-range {{
    margin-left: auto;
    font-size: 11px;
    font-weight: 400;
    opacity: 0.6;
    white-space: nowrap;
    font-style: italic;
  }}

  .legend-item {{
    display: inline-flex;
    gap: 6px;
    align-items: center;
  }}

  .swatch {{
    width: 14px;
    height: 14px;
    border-radius: 4px;
    border: 1px solid rgba(0,0,0,0.25);
    display: inline-block;
    flex-shrink: 0;
  }}

  /* Status swatches */
  .swatch.present           {{ background: #4ade80; }}
  .swatch.not-manifest      {{ background: #fbbf24; }}
  .swatch.samples-collected {{ background: #f87171; }}
  .swatch.blank             {{ background: #e7ecf0; border-color: rgba(0,0,0,0.12); }}

  /* Pallet swatches — one per hue family */
  .swatch.pallet-1  {{ background: #60a5fa; }}   /* sky blue   */
  .swatch.pallet-2  {{ background: #c084fc; }}   /* violet     */
  .swatch.pallet-3  {{ background: #34d399; }}   /* emerald    */
  .swatch.pallet-4  {{ background: #fb923c; }}   /* orange     */
  .swatch.pallet-5  {{ background: #f472b6; }}   /* pink       */
  .swatch.pallet-6  {{ background: #a3e635; }}   /* lime       */

  /* ── Grid ───────────────────────────────────────────── */
  .rack-grid {{
    --cell-w: 42px;
    --gap: 5px;
    display: grid;
    grid-auto-rows: 36px;
    gap: var(--gap);
    grid-template-columns: repeat(18, var(--cell-w));
  }}

  /* ── Mobile: shrink cells so all 18 columns fit on screen ── */
  @media (max-width: 900px) {{
    .rack-grid {{
      --cell-w: clamp(18px, 4.8vw, 38px);
      --gap: 3px;
      grid-auto-rows: clamp(22px, 5.5vw, 34px);
    }}
    .rack-cell {{
      font-size: clamp(8px, 2.2vw, 13px);
      border-radius: 4px;
    }}
    .rack-wrap {{
      padding: 8px 6px 10px 6px;
    }}
    .rack-title {{
      font-size: 13px;
    }}
    .rack-legend {{
      font-size: 10px;
      gap: 8px;
    }}
  }}

  /* ── Base cell ──────────────────────────────────────── */
  .rack-cell {{
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 7px;
    border: 1px solid rgba(0,0,0,0.20);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 0.4px;
    user-select: none;
    transition: transform 0.07s ease, filter 0.07s ease;
    position: relative;
  }}

  /* ── Status cells ───────────────────────────────────── */
  .rack-cell.present {{
    background: #4ade80;
    color: #14532d;
  }}

  .rack-cell.not-manifest {{
    background: #fbbf24;
    color: #78350f;
  }}

  .rack-cell.samples-collected {{
    background: #f87171;
    color: #7f1d1d;
  }}

  /* ── Pallet cells — vibrant, maximally distinct ─────── */
  .rack-cell.pallet-1 {{ background: #60a5fa; color: #1e3a8a; }}  /* sky blue   */
  .rack-cell.pallet-2 {{ background: #c084fc; color: #3b0764; }}  /* violet     */
  .rack-cell.pallet-3 {{ background: #34d399; color: #064e3b; }}  /* emerald    */
  .rack-cell.pallet-4 {{ background: #fb923c; color: #7c2d12; }}  /* orange     */
  .rack-cell.pallet-5 {{ background: #f472b6; color: #831843; }}  /* pink       */
  .rack-cell.pallet-6 {{ background: #a3e635; color: #365314; }}  /* lime       */

  /* ── Empty cell ─────────────────────────────────────── */
  .rack-cell.blank {{
    background: #eef1f4;
    color: #c6ccd2;
    border-color: rgba(0,0,0,0.06);
  }}

  /* ── Packed strikethrough ───────────────────────────── */
  .rack-cell.packed::after {{
    content: '';
    position: absolute;
    left: 4px;
    right: 4px;
    top: 50%;
    height: 2.5px;
    background: rgba(0, 0, 0, 0.55);
    transform: translateY(-50%);
    pointer-events: none;
    border-radius: 2px;
  }}

  .swatch.packed-swatch {{
    background: #9ca3af;
    position: relative;
  }}
  .swatch.packed-swatch::after {{
    content: '';
    position: absolute;
    left: 1px; right: 1px;
    top: 50%;
    height: 2px;
    background: rgba(0, 0, 0, 0.6);
    transform: translateY(-50%);
    border-radius: 1px;
  }}

  /* ── Column-group separators (after col 6 and col 12) ── */
  /* Dark inset shadow = "wider gap" illusion — visible on every cell colour  */
  .rack-grid > .rack-cell:nth-child(18n + 6),
  .rack-grid > .rack-cell:nth-child(18n + 12) {{
    box-shadow: inset -2px 0 0 rgba(0,0,0,0.70);
  }}
  .rack-grid > .rack-cell:nth-child(18n + 7),
  .rack-grid > .rack-cell:nth-child(18n + 13) {{
    box-shadow: inset 2px 0 0 rgba(0,0,0,0.70);
  }}

  /* ── Hover ──────────────────────────────────────────── */
  .rack-cell:hover {{
    transform: translateY(-2px);
    filter: brightness(1.05) drop-shadow(0 4px 10px rgba(0,0,0,0.20));
    z-index: 1;
  }}
</style>
"""
    return html


# ---------------------------------------------------------------------------
# Rack fullscreen dialog (mobile-friendly expand view)
@st.dialog("🔍 Rack — Full View", width="large")
def _show_rack_fullscreen_dialog():
    """Render the saved rack HTML inside a large modal dialog."""
    _html = st.session_state.get("_rack_fs_html", "")
    if _html:
        st.markdown(_html, unsafe_allow_html=True)
    else:
        st.info("No rack to display.")
