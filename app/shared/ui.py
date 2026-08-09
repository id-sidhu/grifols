"""Verbosity-aware Streamlit output helpers and page chrome.

Every section routes its user-facing messages through these wrappers so the
sidebar "Verbose" toggle can silence routine feedback in one place.
"""

import streamlit as st


#: Default verbosity. Users can toggle this live from the sidebar.
VERBOSE_DEFAULT: bool = True


def _verbose() -> bool:
    """Return current verbosity setting (True = show feedback messages)."""
    return st.session_state.get("_verbose", VERBOSE_DEFAULT)


def _show_success(msg: str, **kwargs) -> None:
    if _verbose():
        st.success(msg, **kwargs)


def _show_error(msg: str, **kwargs) -> None:
    if _verbose():
        st.error(msg, **kwargs)


def _show_warning(msg: str, **kwargs) -> None:
    if _verbose():
        st.warning(msg, **kwargs)


def _show_info(msg: str, **kwargs) -> None:
    if _verbose():
        st.info(msg, **kwargs)


def _show_caption(msg: str, **kwargs) -> None:
    if _verbose():
        st.caption(msg, **kwargs)


def _subheader(title: str) -> None:
    st.markdown(
        f'<div class="gf-section-header"><span>{title}</span></div>',
        unsafe_allow_html=True,
    )


def render_page_chrome() -> None:
    """Inject the app stylesheet and draw the title bar."""
    st.markdown("""
<style>
:root {
    --accent:    #3b82f6;
    --accent-bg: rgba(59,130,246,0.10);
    --accent-hi: #1d4ed8;
    --bg-card:   #f6f8fa;
    --border:    #d0d7de;
    --text:      #1f2328;
    --text-dim:  #57606a;
}

/* ── Layout ──────────────────────────────────────────── */
.main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1300px; }

/* ── Hide Streamlit chrome ───────────────────────────── */
#MainMenu, footer { visibility: hidden; }
header { visibility: visible; }

/* ── Custom title bar ────────────────────────────────── */
.gf-title-bar {
    display: flex; align-items: center; gap: 12px;
    padding: 0.4rem 0 0.9rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}
.gf-title-accent { width: 4px; height: 28px; background: var(--accent); border-radius: 2px; flex-shrink: 0; }
.gf-title-text { font-size: 1.3rem; font-weight: 800; color: #1f2328; letter-spacing: -0.03em; line-height: 1; }
.gf-title-sub { font-size: 0.7rem; color: var(--text-dim); margin-top: 3px; letter-spacing: 0.02em; }

/* ── Section headers ─────────────────────────────────── */
.gf-section-header {
    margin: 0.25rem 0 1rem;
    padding: 0.45rem 0.75rem;
    background: var(--bg-card);
    border-left: 3px solid var(--accent);
    border-radius: 0 5px 5px 0;
}
.gf-section-header span { font-size: 1rem; font-weight: 700; color: #1f2328; letter-spacing: -0.01em; }
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="gf-title-bar">
  <div class="gf-title-accent"></div>
  <div>
    <div class="gf-title-text">Grifols</div>
    <div class="gf-title-sub">Winnipeg — Operations</div>
  </div>
</div>
""", unsafe_allow_html=True)
