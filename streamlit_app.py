"""
Streamlit application for Grifols shipment data processing.

This app replicates the functionality of the CLI and HTML versions of
the Grifols data processing pipeline. It reads a raw sample shipment
text file and a Grifols shipment CSV, cleans and processes the data
in memory, and displays the resulting DataFrames in a series of tabs.

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
"""

import io
import pandas as pd
import streamlit as st


def clean_data(data: list[str]) -> list[str]:
    """Replace pipes with commas, collapse whitespace and strip lines."""
    cleaned = []
    for line in data:
        line = line.replace('|', ',')
        # Collapse consecutive whitespace to single spaces using split/join
        clean_text = ' '.join(line.split()).strip()
        # Remove any trailing comma
        if clean_text.endswith(','):
            clean_text = clean_text[:-1]
        cleaned.append(clean_text)
    return cleaned


def raw_df_from_cleaned(cleaned_lines: list[str]):
    """Construct DataFrames from cleaned raw shipment data."""
    csv_data = '\n'.join(cleaned_lines)
    df = pd.read_csv(io.StringIO(csv_data))
    product_df = df[['Product']].copy()
    samples_df = df[['Sample ID']].copy()
    rejected_units = df[df['Quarantine obs.'] != '* '][['Product', 'Sample ID']].copy()
    no_bleeds = df[df['Product'] == 'No bleed '][['Product', 'Sample ID']].copy()
    sample_only = df[df['Product'] == 'Test sample '][['Product', 'Sample ID']].copy()
    return product_df, samples_df, rejected_units, no_bleeds, sample_only


def grifols_shipment_from_df(df: pd.DataFrame):
    """Process a Grifols shipment DataFrame to extract samples and dates."""

    # Ensure required columns exist
    required_cols = {'Sample ID', 'Donation date', 'Samples Packed?'}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            "QC CSV must contain 'Sample ID', 'Donation date', and 'Samples Packed' columns."
        )

    # Remove rows with missing Sample ID
    df = df[df['Sample ID'].notna()].copy()

    # Remove rows where Samples Packed == 'yes'
    df = df[
        ~df['Samples Packed?']
        .astype(str)
        .str.strip()
        .str.lower()
        .eq('yes')
    ]

    # Parse dates
    df['Donation date'] = pd.to_datetime(
        df['Donation date'], dayfirst=True, errors='coerce'
    )

    # Final outputs
    samples_df = df[['Sample ID']].copy()
    date_df = df[['Donation date']].copy()

    return samples_df, date_df


def final_df(original_data, qc_data, no_bleed, rejected, sample_only):
    """Compute samples to remove and the final summary table."""

    # Normalize IDs
    original_ids = original_data['Sample ID'].astype('string').str.strip()
    qc_ids = qc_data['Sample ID'].astype('string').str.strip()

    # Initial removal candidates
    to_be_removed = original_data.loc[~original_ids.isin(qc_ids)].copy()
    to_be_removed['Sample ID'] = to_be_removed['Sample ID'].astype('string').str.strip()

    # Build final summary table (DISPLAY ONLY)
    final_output = pd.DataFrame({
        'To_be_removed': to_be_removed['Sample ID'].reset_index(drop=True),
        'No bleeds': no_bleed['Sample ID'].astype('string').str.strip().reset_index(drop=True),
        'Rejected Units': rejected['Sample ID'].astype('string').str.strip().reset_index(drop=True),
        'Sample Only': sample_only['Sample ID'].astype('string').str.strip().reset_index(drop=True)
    }).fillna('')

    # IDs that must NOT be removed
    protected_ids = pd.concat([
        no_bleed['Sample ID'],
        rejected['Sample ID'],
        sample_only['Sample ID']
    ]).astype('string').str.strip().unique()

    # Final removal list
    to_be_removed = to_be_removed[
        ~to_be_removed['Sample ID'].isin(protected_ids)
    ]

    return to_be_removed[['Sample ID']], final_output



def display_dataframe_with_count(df: pd.DataFrame, label: str):
    """Helper to display a DataFrame with its row count."""
    st.subheader(label)
    st.dataframe(df)
    st.caption(f"Total rows: {len(df)}")


def display_final_counts(df: pd.DataFrame):
    """Display counts per column in the final summary DataFrame."""
    counts = (df != '').sum()
    counts_df = pd.DataFrame({
        'Category': counts.index,
        'Count': counts.values
    })
    st.table(counts_df)


def main():
    st.title("Grifols Shipment Data Processing")
    st.write("Upload the raw sample shipment text file and the Grifols shipment CSV. This app processes the data and displays the results.")

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
                samples_to_be_packed, dates_df = grifols_shipment_from_df(qc_df)

                # Compute final removal and output
                removed_samples, final_out = final_df(
                    samples_df,
                    samples_to_be_packed,
                    no_bleeds_df,
                    rejected_df,
                    sample_only_df
                )

                # Create tabs
                samples_to_be_packed_sorted = (
                  samples_to_be_packed
                  .assign(_sort_key=samples_to_be_packed['Sample ID'].astype(str))
                  .sort_values('_sort_key')
                  .drop(columns='_sort_key')
                )
                tab_labels = [
                    ("Product", product_df),
                    ("Samples", samples_df),
                    ("Rejected Units", rejected_df),
                    ("No Bleeds", no_bleeds_df),
                    ("Sample Only", sample_only_df),
                    ("Samples to be Packed", samples_to_be_packed_sorted),
                    ("Donation Date", dates_df),
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











