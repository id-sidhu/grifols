"""Section registry and feature flags.

Set a section to ``False`` in :data:`FEATURES` to hide it from the navigation
bar.  Hidden sections stay reachable by adding ``?unlock=<UNLOCK_KEY>`` to the
URL, where ``UNLOCK_KEY`` comes from Streamlit secrets.
"""

from typing import Dict, List


#: Every section, in navigation order.
SECTIONS: List[str] = [
    "Pallet Report",
    "Manual racks/Unit Status Check",
    "Visual Inspection Labels",
    "QC Report PDF Extractor",
    "Master Sheet ",
    "Storage Manager",
    "Donation Processing",
]

#: Which sections are visible without the unlock key.
FEATURES: Dict[str, bool] = {
    "Pallet Report": True,
    "Manual racks/Unit Status Check": True,
    "Visual Inspection Labels": True,
    "QC Report PDF Extractor": True,
    "Master Sheet ": False,
    "Storage Manager": True,
    "Donation Processing": True,
}


def visible_sections(unlocked: bool) -> List[str]:
    """Sections to show in the navigation bar."""
    if unlocked:
        return list(SECTIONS)
    return [s for s in SECTIONS if FEATURES.get(s, True)]
