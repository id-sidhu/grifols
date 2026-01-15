"""
Streamlit application for Grifols shipment data processing.

This app replicates the functionality of the CLI and HTML versions of
the Grifols data processing pipeline. It reads a raw sample shipment
text file and a Grifols shipment CSV, cleans and processes the data in
memory, and displays the resulting DataFrames in a series of tabs.

Key features:

  * File upload widgets for the raw text and the shipment CSV.
  * Data processing logic identical to the original CLI functions.
  * Interactive tabs showing each intermediate and final DataFrame.
  * Totals: The number of rows for each DataFrame is displayed below
    the table. For the final summary table, counts per category are
    shown separately.
  * All processing occurs in memory; no files are written to disk.

To run the app, install Streamlit (``pip install streamlit``) and
execute ``streamlit run streamlit_app.py``. The app will open in a
web browser. This script does not contain any data of its own; you
must provide the raw sample shipment text and the Grifols shipment
CSV via the upload widgets. If you need to adapt the paths or
functionality, the code has been structured to separate concerns.

This version of the app has been updated so that sample IDs from
rows where the "Samples Packed?" (or equivalent) column is marked
``yes`` are *not* dropped entirely. In the original implementation
these rows were removed from the QC DataFrame before further
processing, which caused the corresponding sample IDs to end up in
the ``to_be_removed`` DataFrame. In practice, a ``yes`` value in
``Samples Packed?`` indicates that the sample has already been
collected/packed and therefore should not be flagged for removal. To
accommodate this, the app now collects those sample IDs and excludes
them from the removal list. Optionally, the collected sample IDs are
displayed in their own tab for reference.
"""

import io
import pandas as pd
import streamlit as st


def clean_data(data: list[str]) -> list[str]:
    """Replace pipes with commas, collapse whitespace and strip lines.

    Parameters
    ----------
    data: list[str]
        A list of raw lines from the sample shipment text file.

    Returns
    -------
    list[str]
        A list of cleaned lines suitable for CSV parsing.
    """
    cleaned: list[str] = []
    for line in data:
        # Replace pipe characters with commas
        line = line.replace('|', ',')
        # Collapse consecutive whitespace to single spaces
        clean_text = ' '.join(line.split()).strip()
        # Remove any trailing comma
        if clean_text.endswith(','):
            clean_text = clean_text[:-1]
        cleaned.append(clean_text)
    return cleaned


def raw_df_from_cleaned(cleaned_lines: list[str]):
    """Construct DataFrames from cleaned raw shipment data.

    Parameters
    ----------
    cleaned_lines: list[str]
        Cleaned lines of the raw sample data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the following DataFrames: product_df,
        samples_df, rejected_units_df, no_bleeds_df, sample_only_df.
    """
    csv_data = '\n'.join(cleaned_lines)
    df = pd.read_csv(io.StringIO(csv_data))
    # Extract the columns of interest
    product_df = df[['Product']].copy()
    samples_df = df[['Sample ID']].copy()
    rejected_units = df[df['Quarantine obs.'] != '* '][['Product', 'Sample ID']].copy()
    no_bleeds = df[df['Product'] == 'No bleed '][['Product', 'Sample ID']].copy()
    sample_only = df[df['Product'] == 'Test sample '][['Product', 'Sample ID']].copy()
    return product_df, samples_df, rejected_units, no_bleeds, sample_only


def grifols_shipment_from_df(df: pd.DataFrame):
    """Process a Grifols shipment DataFrame to extract samples and dates.

    This function reads the QC CSV, validates required columns,
    identifies rows where the ``Samples Packed?`` column indicates that
    the sample has already been collected (value ``yes``), and
    returns both the full list of sample IDs and those with ``yes``.

    Parameters
    ----------
    df : pandas.DataFrame
        The uploaded QC CSV as a DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing: samples_df (all sample IDs), date_df
        (parsed donation dates), and packed_df (sample IDs where
        ``Samples Packed?`` is ``yes``).
    """
    # Ensure required columns exist. The column may sometimes be named
    # slightly differently (e.g. missing a question mark), so we
    # normalise the lookup.
    required_cols = {'Sample ID', 'Donation date'}
    packed_col_candidates = [col for col in df.columns if col.lower().startswith('samples') and 'packed' in col.lower()]
    if packed_col_candidates:
        packed_col = packed_col_candidates[0]
        required_cols.add(packed_col)
    else:
        packed_col = None

    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        raise ValueError(
            f"QC CSV must contain the following columns: {', '.join(missing)}."
        )

    # Remove rows with missing Sample ID
    df = df[df['Sample ID'].notna()].copy()

    # Identify rows where Samples Packed? is yes (case-insensitive).
    if packed_col is not None:
        packed_mask = (
            df[packed_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq('yes')
        )
        packed_df = df.loc[packed_mask, ['Sample ID']].copy()
    else:
        # If there is no packed column, return an empty DataFrame
        packed_df = pd.DataFrame(columns=['Sample ID'])

    # For purposes of comparison to the raw data we *keep* all rows,
    # including those with Samples Packed == 'yes'. This ensures that
    # their sample IDs will not appear in the removal list.

    # Parse dates. Use dayfirst=True to interpret dates like '22/01/2025'.
    df['Donation date'] = pd.to_datetime(
        df['Donation date'], dayfirst=True, errors='coerce'
    )

    # Construct output DataFrames
    samples_df = df[['Sample ID']].copy()
    date_df = df[['Donation date']].copy()

    return samples_df, date_df, packed_df


def final_df(
    original_data: pd.DataFrame,
    qc_data: pd.DataFrame,
    no_bleed: pd.DataFrame,
    rejected: pd.DataFrame,
    sample_only: pd.DataFrame,
    packed_yes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute samples to remove and the final summary table.

    Parameters
    ----------
    original_data : pandas.DataFrame
        The DataFrame of sample IDs extracted from the raw text file.
    qc_data : pandas.DataFrame
        The DataFrame of sample IDs extracted from the QC CSV. This
        should include all sample IDs, including those where the
        ``Samples Packed?`` column is ``yes``.
    no_bleed : pandas.DataFrame
        Rows from the raw data where Product is ``No bleed ``.
    rejected : pandas.DataFrame
        Rows from the raw data where the quarantine observation is not
        ``* ``.
    sample_only : pandas.DataFrame
        Rows from the raw data where Product is ``Test sample ``.
    packed_yes : pandas.DataFrame, optional
        A DataFrame of sample IDs where ``Samples Packed?`` equals
        ``yes`` in the QC CSV. These sample IDs will be excluded
        from the removal list.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the DataFrame of samples to be removed and
        the final summary DataFrame.
    """
    # Normalise and strip whitespace for reliable comparison
    original_ids = original_data['Sample ID'].astype('string').str.strip()
    qc_ids = qc_data['Sample ID'].astype('string').str.strip()

    # Samples present in the raw data but *not* in the QC data are
    # candidates for removal.
    to_be_removed = original_data.loc[~original_ids.isin(qc_ids)].copy()
    to_be_removed['Sample ID'] = to_be_removed['Sample ID'].astype('string').str.strip()

    # Build the final summary table containing the special categories
    final_output = pd.DataFrame({
        'No bleeds': no_bleed['Sample ID'].astype('string').str.strip().reset_index(drop=True),
        'Rejected Units': rejected['Sample ID'].astype('string').str.strip().reset_index(drop=True),
        'Sample Only': sample_only['Sample ID'].astype('string').str.strip().reset_index(drop=True)
    })

    # Collect IDs from the special categories so they can be excluded from
    # the removal list.
    final_ids = final_output.stack().dropna().unique()

    # Include packed sample IDs (where Samples Packed == yes) in the
    # exclusion set. These are valid samples that have already been
    # collected and therefore should not be slated for removal.
    if packed_yes is not None and not packed_yes.empty:
        packed_ids = packed_yes['Sample ID'].astype('string').str.strip().unique()
        # Combine with existing final_ids and deduplicate
        final_ids = pd.unique(pd.concat([
            pd.Series(final_ids, dtype='string'),
            pd.Series(packed_ids, dtype='string')
        ]))

    # Exclude any sample IDs that appear in the special categories or
    # among the packed (already collected) samples from the removal list.
    to_be_removed = to_be_removed[~to_be_removed['Sample ID'].isin(final_ids)]

    return to_be_removed[['Sample ID']], final_output.fillna('')


def display_dataframe_with_count(df: pd.DataFrame, label: str) -> None:
    """Helper to display a DataFrame with its row count in Streamlit.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to display.
    label : str
        A descriptive label used as the subheader and in the caption.
    """
    st.subheader(label)
    st.dataframe(df)
    st.caption(f"Total rows: {len(df)}")


def display_final_counts(df: pd.DataFrame) -> None:
    """Display counts per column in the final summary DataFrame.

    Each column corresponds to one of the special categories. The
    counts are shown in a small table.
    """
    counts = df.count()
    counts_df = pd.DataFrame({'Category': counts.index, 'Count': counts.values})
    st.table(counts_df)


def main() -> None:
    """Run the Streamlit application."""
    st.title("Grifols Shipment Data Processing")
    st.write(
        "Upload the raw sample shipment text file and the Grifols shipment "
        "CSV. This app processes the data and displays the results."
    )

    raw_file = st.file_uploader("Raw sample shipment text file", type=['txt'])
    qc_file = st.file_uploader("Grifols shipment CSV", type=['csv'])

    if raw_file and qc_file:
        process = st.button("Run Processing")
        if process:
            try:
                # Read raw text lines
                raw_lines = raw_file.getvalue().decode('utf-8').splitlines()
                cleaned = clean_data(raw_lines)
                product_df, samples_df, rejected_df, no_bleeds_df, sample_only_df = raw_df_from_cleaned(cleaned)

                # Read QC CSV into DataFrame
                qc_df = pd.read_csv(qc_file)
                samples_to_be_packed, dates_df, packed_yes_df = grifols_shipment_from_df(qc_df)

                # Compute final removal and output. We pass the packed_yes_df so
                # that its sample IDs are excluded from the removal list.
                removed_samples, final_out = final_df(
                    samples_df,
                    samples_to_be_packed,
                    no_bleeds_df,
                    rejected_df,
                    sample_only_df,
                    packed_yes=packed_yes_df
                )

                # Prepare tab labels and associated DataFrames. We include
                # the packed_yes_df as its own tab so that users can see
                # which sample IDs were marked as already collected/packed.
                tab_labels = [
                    ("Product", product_df),
                    ("Samples", samples_df),
                    ("Rejected Units", rejected_df),
                    ("No Bleeds", no_bleeds_df),
                    ("Sample Only", sample_only_df),
                    ("Samples to be Packed", samples_to_be_packed),
                    ("Donation Date", dates_df),
                    ("Packed Samples", packed_yes_df),
                    ("Removed Samples", removed_samples),
                    ("Final Output", final_out)
                ]
                tabs = st.tabs([label for label, _ in tab_labels])
                for idx, (label, df) in enumerate(tab_labels):
                    with tabs[idx]:
                        st.dataframe(df)
                        st.caption(f"Total rows: {len(df)}")
                        if label == "Final Output":
                            # Show counts per category in final output
                            st.write("Counts per category:")
                            display_final_counts(df)
            except Exception as e:
                st.error(f"An error occurred while processing the files: {e}")

    else:
        st.info("Please upload both files to begin.")


if __name__ == "__main__":
    main()
