"""Visual Inspection Labels section."""

import base64
import datetime
import re
import secrets as _secrets_mod

import pandas as pd
import streamlit as st

from app.features.vi_labels.gist import (
    _vi_find_next_start,
    _vi_gist_load,
    _vi_gist_save,
    _vi_hash_id,
)
from app.features.vi_labels.logic import build_vi_label_groups
from app.features.vi_labels.pdf import generate_vi_labels_pdf
from app.shared.dates import _parse_donation_date
from app.shared.ui import (
    _show_caption,
    _show_error,
    _show_info,
    _show_success,
    _show_warning,
    _subheader,
)


def render_vi_labels(us_df: pd.DataFrame, key_ns: str = "nav") -> None:
    """Render the Visual Inspection Labels UI for ``us_df``.

    Called from two places: the "Visual Inspection Labels" nav section, and
    the Donation Processing section immediately after a successful append to
    the unit status database.  ``key_ns`` namespaces every widget key and the
    session-state result slot so the two instances never collide.

    The Gist continuation state is deliberately *not* namespaced -- it is
    keyed on donation prefix, so labels resume from the right ID whichever
    entry point was used.
    """
    _vi_result_key = f"{key_ns}_vi_pdf_result"
    _subheader("Visual Inspection Labels")
    st.write(
        "Groups Quarantine units into batches of N and generates a printable "
        "PDF.  Each page has two different labels (top and bottom half). "
        "Rejected and SO units are included in the printed range but do **not** "
        "count toward the group size."
    )

    vi_c1, vi_c2, vi_c3 = st.columns([2, 3, 1])
    with vi_c1:
        vi_prefix = st.text_input(
            "Donation prefix", value="F26-", key=f"{key_ns}_vi_prefix", max_chars=20
        ).strip()
    with vi_c3:
        vi_group_size = st.number_input(
            "Group size", min_value=1, max_value=216, value=12, step=1,
            key=f"{key_ns}_vi_group_size",
        )

    # Load Gist state once per session (cached in session_state)
    if "vi_gist_state" not in st.session_state:
        st.session_state["vi_gist_state"] = _vi_gist_load()
    _vi_gist_state = st.session_state["vi_gist_state"]
    _vi_prefix_state = _vi_gist_state.get(vi_prefix, {})
    _vi_gist_configured = bool(
        _vi_prefix_state.get("salt") and _vi_prefix_state.get("last_complete_hash")
    )

    # Auto-detect start: find the ID after the last printed complete group
    _vi_auto_start = ""
    if _vi_gist_configured:
        _vi_auto_start = _vi_find_next_start(
            us_df, vi_prefix,
            _vi_prefix_state["salt"],
            _vi_prefix_state["last_complete_hash"],
        ) or ""

    # Compute default start ID: first Quarantine unit from the latest date
    _vi_default_start = ""
    try:
        _vi_dn = us_df.get("Donation #", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
        _vi_pm = _vi_dn.str.upper().str.startswith(vi_prefix.upper())
        _vi_qs = us_df.get("Status", pd.Series(dtype=str)).fillna("").astype(str).str.strip().str.lower() == "quarantine"
        _vi_filt = us_df.loc[_vi_pm & _vi_qs].copy()
        if "Donation Date" in _vi_filt.columns and not _vi_filt.empty:
            _vi_filt["_pd"] = _vi_filt["Donation Date"].map(_parse_donation_date)
            _vi_max_d = _vi_filt["_pd"].dropna().max()
            if _vi_max_d is not None:
                _vi_on_max = _vi_filt[_vi_filt["_pd"] == _vi_max_d]
                def _vi_sort_num(x):
                    m = re.match(rf"^{re.escape(vi_prefix)}(\d+)$", str(x), re.IGNORECASE)
                    return int(m.group(1)) if m else int(1e18)
                _vi_sorted = _vi_on_max.sort_values(
                    by="Donation #", key=lambda col: col.map(_vi_sort_num)
                )
                if not _vi_sorted.empty:
                    _vi_default_start = str(_vi_sorted.iloc[0]["Donation #"]).strip()
    except Exception:
        pass

    # Auto-detected takes priority over date-based default
    _vi_effective_start = _vi_auto_start or _vi_default_start

    # Show auto-detect info / reset button
    if _vi_auto_start:
        _last_updated = _vi_prefix_state.get("last_updated", "unknown date")
        _info_col, _reset_col = st.columns([5, 1])
        with _info_col:
            _show_info(
                f"Auto-detected start: **{_vi_auto_start}** "
                f"— last complete group covered donations up to {_last_updated}"
            )
        with _reset_col:
            st.write("")
            if st.button("Reset", key=f"{key_ns}_vi_reset_state", help="Clear saved state for this prefix"):
                _vi_gist_state.pop(vi_prefix, None)
                _vi_gist_save(_vi_gist_state)
                st.session_state["vi_gist_state"] = _vi_gist_state
                st.rerun()
    elif not _vi_gist_configured:
        _show_caption(
            "No saved state found for this prefix. "
            "After generating a PDF the start position will be saved automatically."
        )

    with vi_c2:
        vi_start_id = st.text_input(
            "Start from donation ID",
            value=_vi_effective_start,
            key=f"{key_ns}_vi_start_id",
            placeholder="e.g. F26-012401",
        ).strip()

    if st.button("Generate Visual Inspection Labels PDF", key=f"{key_ns}_btn_vi_labels"):
        st.session_state.pop(_vi_result_key, None)
        if not vi_start_id:
            _show_error("Please enter a start donation ID.")
        else:
            try:
                vi_groups = build_vi_label_groups(
                    us_df, vi_prefix, vi_start_id, group_size=int(vi_group_size)
                )
                if not vi_groups:
                    _show_warning("No groups found from the specified start ID.")
                else:
                    # End date = last donation date in the file + 1 day
                    _all_dates = (
                        us_df.get("Donation Date", pd.Series(dtype=str))
                        .map(_parse_donation_date)
                        .dropna()
                    )
                    _max_date = _all_dates.max() if not _all_dates.empty else None
                    _tomorrow = (
                        _max_date + datetime.timedelta(days=1)
                        if _max_date
                        else datetime.date.today()
                    )
                    vi_pdf = generate_vi_labels_pdf(vi_groups, tomorrow=_tomorrow)

                    # Save state: hash of last complete group's last ID
                    _vi_last_complete = next(
                        (g for g in reversed(vi_groups) if g["is_complete"]), None
                    )
                    if _vi_last_complete:
                        _vi_state = st.session_state.get("vi_gist_state", {})
                        _vi_ps = _vi_state.get(vi_prefix, {})
                        _vi_salt = _vi_ps.get("salt") or _secrets_mod.token_hex(16)
                        _vi_lc_date = _vi_last_complete.get("date_max")
                        _vi_state[vi_prefix] = {
                            "salt": _vi_salt,
                            "last_complete_hash": _vi_hash_id(_vi_salt, _vi_last_complete["last_id"]),
                            "last_updated": (
                                _vi_lc_date.strftime("%d.%m.%Y")
                                if _vi_lc_date
                                else datetime.date.today().strftime("%d.%m.%Y")
                            ),
                        }
                        _saved_ok = _vi_gist_save(_vi_state)
                        st.session_state["vi_gist_state"] = _vi_state
                        if _saved_ok:
                            _vi_msg = (
                                f"Generated {len(vi_groups)} label(s). "
                                f"Next session will auto-start from the continuation point."
                            )
                            _vi_msg_kind = "success"
                        else:
                            _vi_msg = (
                                f"Generated {len(vi_groups)} label(s) but could not save state "
                                f"(check GITHUB_TOKEN and GIST_ID in Streamlit secrets)."
                            )
                            _vi_msg_kind = "warning"
                    else:
                        _vi_msg = f"Generated {len(vi_groups)} label(s)."
                        _vi_msg_kind = "success"
                    # Preview table
                    _prev = []
                    for _g in vi_groups:
                        _dm, _dx = _g["date_min"], _g["date_max"]
                        if _g["is_complete"]:
                            _dr = (
                                f"{_dm.strftime('%d.%m.%Y')} – {_dx.strftime('%d.%m.%Y')}"
                                if _dm and _dx else "—"
                            )
                            _ir = f"{_g['first_id']} – {_g['last_id']}"
                        else:
                            _dr = (
                                f"{_dx.strftime('%d.%m.%Y')} – {_tomorrow.strftime('%d.%m.%Y')}"
                                if _dx else f"? – {_tomorrow.strftime('%d.%m.%Y')}"
                            )
                            _ir = f"{_g['first_id']} –"
                        _prev.append({
                            "Date range": _dr,
                            "ID range": _ir,
                            "Quarantine": _g["valid_count"],
                            "Total rows": len(_g["rows"]),
                            "Complete": "✓" if _g["is_complete"] else "(partial)",
                        })
                    # Keep result in session state so the Download / Print
                    # buttons and preview survive Streamlit reruns
                    st.session_state[_vi_result_key] = {
                        "pdf": vi_pdf,
                        "n_groups": len(vi_groups),
                        "msg": _vi_msg,
                        "msg_kind": _vi_msg_kind,
                        "preview": _prev,
                    }
            except ImportError as _e:
                _show_error(str(_e))
            except ValueError as _e:
                _show_error(str(_e))
            except Exception as _e:
                st.exception(_e)

    _vi_res = st.session_state.get(_vi_result_key)
    if _vi_res:
        if _vi_res["msg_kind"] == "warning":
            _show_warning(_vi_res["msg"])
        else:
            _show_success(_vi_res["msg"])
        _dl_col, _pr_col = st.columns([1, 1])
        with _dl_col:
            st.download_button(
                label=f"⬇ Download PDF ({_vi_res['n_groups']} label pages)",
                data=_vi_res["pdf"],
                file_name="vi_labels.pdf",
                mime="application/pdf",
                key=f"{key_ns}_vi_pdf_dl",
            )
        with _pr_col:
            _vi_b64 = base64.b64encode(_vi_res["pdf"]).decode()
            _vi_print_html = """
<button onclick="printVI()" style="
  background:#1e6fbf;color:#fff;border:none;padding:0.45rem 1rem;
  border-radius:0.375rem;cursor:pointer;font-size:0.875rem;
  font-family:sans-serif;width:100%;margin-top:4px;">
  &#128438;&nbsp;Print PDF
</button>
<script>
var _viUrl=null;
function _viBlobUrl(){
  if(_viUrl)return _viUrl;
  var bin=atob("__VI_B64__");
  var arr=new Uint8Array(bin.length);
  for(var i=0;i<bin.length;i++)arr[i]=bin.charCodeAt(i);
  _viUrl=URL.createObjectURL(new Blob([arr],{type:"application/pdf"}));
  return _viUrl;
}
function printVI(){
  var url=_viBlobUrl();
  // Firefox cannot print a PDF from a hidden iframe - open a tab instead
  if(navigator.userAgent.toLowerCase().indexOf("firefox")>-1){
    window.open(url,"_blank");
    return;
  }
  var old=document.getElementById("__VI_FRAME__");
  if(old)old.parentNode.removeChild(old);
  var f=document.createElement("iframe");
  f.id="__VI_FRAME__";
  f.style.cssText="position:fixed;right:0;bottom:0;width:0;height:0;border:0;";
  f.onload=function(){
    setTimeout(function(){
      try{f.contentWindow.focus();f.contentWindow.print();}
      catch(e){window.open(url,"_blank");}
    },200);
  };
  f.src=url;
  document.body.appendChild(f);
}
</script>
""".replace("__VI_B64__", _vi_b64).replace(
                "__VI_FRAME__", f"vi-print-frame-{key_ns}"
            )
            st.components.v1.html(_vi_print_html, height=48)
        st.table(pd.DataFrame(_vi_res["preview"]))
