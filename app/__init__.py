"""Grifols Winnipeg operations app.

A Streamlit application combining the pallet packing report with several
supporting tools.  Users upload a Grifols shipment file and, optionally, a
unit status file; every section then works from those two DataFrames.

Sections
--------
Pallet Report
    Select a pallet number to get its start/end markers, the samples still to
    pack, F25/F26 splits and a per-donation-date breakdown.  Also generates
    box numbers for pasting into Excel.
Manual racks/Unit Status Check
    Check one control number, or a comma-separated pair to list every ID in
    that range that is present in the unit status file but missing from the
    shipment manifest (``not_in_manifest``), excluding those classified for
    removal (no bleed, sample only, rejected).  Renders the rack grid.
Visual Inspection Labels
    Groups donations into label batches and renders printable PDF sheets,
    tracking progress across sessions in a GitHub gist.
QC Report PDF Extractor
    Extracts Unit ID and donation date from QC report PDFs and compares them
    against release and packing data.
Master Sheet
    Fills the daily inventory table from the uploaded files.
Storage Manager
    Browses, uploads and deletes files in Supabase storage.
Donation Processing
    Parses PC-Blut barcodes, validates the freeze tracker, and appends the
    resulting rows to the Unit Status CSV database.

Layout
------
``app.shared``
    Helpers used by more than one section: UI feedback wrappers, date
    parsing, CSV/Excel reading and Supabase storage access.
``app.features.<name>``
    One package per section.  Each exposes ``render(...)`` and keeps its own
    logic modules next to it.
``app.main``
    Composition root: loads the uploaded files once and dispatches to the
    selected section.

Run locally with::

    streamlit run streamlit_app.py
"""
