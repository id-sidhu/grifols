"""Cross-session VI progress, persisted to a GitHub gist.

Donation IDs are stored only as salted hashes so the gist never holds
identifiable numbers.
"""

import hashlib
import json
import re
from typing import Optional

import pandas as pd
import streamlit as st


def _vi_gist_load() -> dict:
    """Read VI label state JSON from a GitHub Gist.

    Requires ``GITHUB_TOKEN`` and ``GIST_ID`` to be set in Streamlit secrets.
    Returns an empty dict if the Gist is unreachable or secrets are missing.
    The Gist file is named ``vi_state.json``.
    """
    import requests as _req
    try:
        token = st.secrets["GITHUB_TOKEN"]
        gist_id = st.secrets["GIST_ID"]
    except (KeyError, Exception):
        return {}
    try:
        resp = _req.get(
            f"https://api.github.com/gists/{gist_id}",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        content = resp.json().get("files", {}).get("vi_state.json", {}).get("content", "{}")
        return json.loads(content)
    except Exception:
        return {}


def _vi_gist_save(state: dict) -> bool:
    """Write VI label state JSON back to the GitHub Gist.

    Returns ``True`` on success, ``False`` otherwise.
    """
    import requests as _req
    try:
        token = st.secrets["GITHUB_TOKEN"]
        gist_id = st.secrets["GIST_ID"]
    except (KeyError, Exception):
        return False
    try:
        resp = _req.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json={"files": {"vi_state.json": {"content": json.dumps(state, indent=2)}}},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _vi_hash_id(salt: str, donation_id: str) -> str:
    """Return the SHA-256 hex digest of ``salt + donation_id``."""
    return hashlib.sha256(f"{salt}{donation_id}".encode()).hexdigest()


def _vi_find_next_start(
    us_df: pd.DataFrame, prefix: str, salt: str, target_hash: str
) -> Optional[str]:
    """Return the donation ID that comes immediately after the one matching ``target_hash``.

    Hashes every candidate ID (those starting with ``prefix``) using ``salt``
    and compares against ``target_hash``.  Returns ``None`` if the hash is not
    found or the matching ID is the last in the sorted list.
    """
    dn_col = us_df.get("Donation #", pd.Series(dtype=str)).fillna("").astype(str).str.strip()

    def _num(x: str) -> int:
        m = re.match(rf"^{re.escape(prefix)}(\d+)$", x, re.IGNORECASE)
        return int(m.group(1)) if m else int(1e18)

    candidates = sorted(
        {x for x in dn_col if x.upper().startswith(prefix.upper()) and _num(x) < int(1e18)},
        key=_num,
    )
    for idx, cid in enumerate(candidates):
        if _vi_hash_id(salt, cid) == target_hash:
            return candidates[idx + 1] if idx + 1 < len(candidates) else None
    return None
