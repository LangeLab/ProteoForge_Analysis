#!/usr/bin/env python3
"""
Data Processing Module for ProteoForge Library

Version: 0.7.0
Date: 2025-07-01
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

import time
from itertools import zip_longest
from typing import List, Optional

import numpy as np
import pandas as pd


# ======================================================================================
# Core Configuration
# ======================================================================================

# Default column name candidates for auto-detection
SEQUENCE_COLUMNS = ['sequence', 'protein_sequence', 'Sequence', 'Protein_Sequence', 'protein_seq']
PEPTIDE_COLUMNS = ['peptide', 'peptide_sequence', 'Peptide', 'Peptide_Sequence', 'peptide_seq']
PROTEIN_COLUMNS = ['protein', 'protein_id', 'Protein', 'Protein_ID', 'accession', 'uniprot']


# ======================================================================================
# Protein Information Processing
# ======================================================================================

def select_representative_protein(
    proteins: str,
    separator: str = ";",
    priority_length: int = 6
) -> str:
    """
    Select a representative protein ID from a delimited string of protein IDs.

    Designed for UniProt accession IDs where shorter IDs (default: 6 chars) 
    are preferred over longer isoform/variant IDs.

    Args:
        proteins: Delimited string of protein IDs
        separator: Delimiter character. Default: ";"
        priority_length: Preferred ID length. Default: 6

    Returns:
        Representative protein ID

    Examples:
        >>> select_representative_protein("P12345")
        'P12345'
        >>> select_representative_protein("A0A075B6K5;Q12345")
        'Q12345'
    """
    protein_ids = proteins.split(separator)
    if len(protein_ids) == 1:
        return protein_ids[0]
    
    priority_ids = [id for id in protein_ids if len(id) == priority_length]
    return priority_ids[0] if priority_ids else protein_ids[0]

# Exploding multiple columns in a DataFrame while preserving alignment and padding.
# TODO: This requires more robust handling of edge cases, such as:
    # Unaligned element numbers across columns results in NaN values and wrong alignment.
def explode_aligned(df, columns, sep=';', keep_index=False, verbose=False):
    """
    Robustly and efficiently explodes multiple DataFrame columns simultaneously,
    preserving row alignment and padding where necessary.

    This function takes a DataFrame and a list of columns containing delimited
    strings (e.g., 'A;B;C'), splits them, and expands the DataFrame so that
    each split item gets its own row. The explosion is aligned across all
    specified columns. If lists of items are of unequal length for a given
    row, the shorter lists are padded with NaN to match the longest one.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list of str): A list of column names to explode.
        sep (str, optional): The delimiter to split on. Defaults to ';'.
        keep_index (bool, optional): If True, adds a column 'original_index'
          to trace rows back to the original DataFrame. Defaults to False.
        verbose (bool, optional): If True, prints the number of rows before
          and after the transformation and the total time taken. Defaults to False.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns exploded.
    """
    start_time = time.time()
    
    # 1. SETUP AND VALIDATION
    if verbose:
        print(f"🚀 Starting aligned explosion for columns: {columns}")
        print(f"Original DataFrame shape: {df.shape}")

    df_copy = df.copy()

    if keep_index:
        df_copy['original_index'] = df.index

    # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]

    # 2. ISOLATE & INDEX
    other_cols = df_copy.columns.difference(columns).tolist()
    if not other_cols:
        temp_idx_name = '_temp_idx_'
        df_copy[temp_idx_name] = df_copy.index
        other_cols = [temp_idx_name]
        
    df_indexed = df_copy.set_index(other_cols)

    # 3. CORE TRANSFORMATION: SPLIT -> ALIGN -> EXPLODE
    def _split_cell(val, separator):
        if pd.isna(val):
            return [np.nan]
        return str(val).split(separator)

    series_of_lists = [
        df_indexed[col].apply(_split_cell, separator=sep) for col in columns
    ]

    aligned_rows_gen = (
        list(zip_longest(*row_lists, fillvalue=np.nan))
        for row_lists in zip(*series_of_lists)
    )
    aligned_series = pd.Series(aligned_rows_gen, index=df_indexed.index)
    
    exploded_series = aligned_series.explode(ignore_index=False)
    exploded_series.dropna(how='all', inplace=True)
    
    # 4. RECONSTRUCT DATAFRAME
    exploded_data = pd.DataFrame(
        exploded_series.tolist(),
        index=exploded_series.index,
        columns=columns
    )
    final_df = exploded_data.reset_index()

    # 5. FINALIZE (Corrected Section)
    # Restore a logical column order
    final_order = []
    
    # Handle 'original_index' and remove it from other_cols to prevent duplication
    if keep_index and 'original_index' in final_df.columns:
        final_order.append('original_index')
        if 'original_index' in other_cols:
            other_cols.remove('original_index')
            
    # Add exploded columns first, then the remaining other original columns
    final_order.extend(columns)
    final_order.extend([c for c in other_cols if c in final_df.columns])
    
    # Clean up the temporary index name if it exists
    if '_temp_idx_' in final_order:
        final_order.remove('_temp_idx_')
    
    final_df = final_df[final_order]

    if verbose:
        end_time = time.time()
        print(f"Final DataFrame shape: {final_df.shape}")
        print(f"✅ Transformation complete in {end_time - start_time:.4f} seconds.")
        
    return final_df

# ======================================================================================
# Peptide Position Mapping
# ======================================================================================

## TODO: Expand it so that it can handle more complex cases, such as:
    # - Handling of overlapping peptides
    # - Support for different peptide lengths and modifications 
    # - Different types of modification labeling formats
def map_peptide_to_protein_positions(
    data: pd.DataFrame,
    # Input column specifications (auto-detected if None)
    sequence_col: Optional[str] = None,
    peptide_col: Optional[str] = None,
    protein_col: Optional[str] = None,
    # Output column names
    peptide_start_col: str = 'peptide_start',
    peptide_end_col: str = 'peptide_end',
    unique_id_col: str = 'unique_id',
    total_occurrences_col: str = 'total_occurrences',
    occurrence_index_col: str = 'occurrence_index',
    original_index_col: str = 'original_index',
    # Processing options
    expand_multiple: bool = True,
    add_unique_id: bool = True,
    add_occurrence_info: bool = True,
    position_ordered: bool = True, 
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Map peptides to their positions within protein sequences with high performance.
    
    Uses vectorized operations for efficient processing of large datasets (100K+ rows).
    Supports multiple occurrences of peptides within proteins and flexible output.
    
    Args:
        data: DataFrame with protein sequences and peptides
        sequence_col: Protein sequence column name (auto-detected if None)
        peptide_col: Peptide sequence column name (auto-detected if None)
        protein_col: Protein ID column name (auto-detected if None)
        peptide_start_col: Output column name for start positions
        peptide_end_col: Output column name for end positions
        unique_id_col: Output column name for unique proteoform IDs
        total_occurrences_col: Output column name for occurrence counts
        occurrence_index_col: Output column name for occurrence indices
        original_index_col: Output column name for original row indices
        expand_multiple: Create separate rows for multiple occurrences
        add_unique_id: Add unique proteoform identifiers
        add_occurrence_info: Add occurrence statistics
        position_ordered: Order peptides by their positions within the protein
        verbose: Print detailed processing information and statistics
        
    Returns:
        DataFrame with peptide positions and optional additional columns
        
    Raises:
        ValueError: If required columns not found or contain invalid data
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'protein_sequence': ['PEPTIDEKPEPTIDE', 'SEQUENCE'],
        ...     'peptide_sequence': ['PEPTIDE', 'SEQ'],
        ...     'protein_id': ['P001', 'P002']
        ... })
        >>> result = map_peptide_to_protein_positions(df, verbose=True)
    """
    # Auto-detect column names
    sequence_col = sequence_col or _detect_column_name(data, SEQUENCE_COLUMNS, required=True)
    peptide_col = peptide_col or _detect_column_name(data, PEPTIDE_COLUMNS, required=True)
    
    if protein_col is None and (add_unique_id or add_occurrence_info):
        protein_col = _detect_column_name(data, PROTEIN_COLUMNS, required=False)
    
    if verbose:
        print("=" * 60)
        print("PEPTIDE-TO-PROTEIN POSITION MAPPING")
        print("=" * 60)
        print(f"Input dataset: {len(data):,} rows")
        print(f"Detected columns:")
        print(f"  - Protein sequence: '{sequence_col}'")
        print(f"  - Peptide sequence: '{peptide_col}'")
        print(f"  - Protein ID: '{protein_col}'" if protein_col else "  - Protein ID: Not detected")
        print(f"Processing options:")
        print(f"  - Expand multiple occurrences: {expand_multiple}")
        print(f"  - Add unique IDs: {add_unique_id}")
        print(f"  - Add occurrence info: {add_occurrence_info}")
        print(f"  - Position ordered: {position_ordered}")
        print(f"  - Preserve original indices: True")  # Always true now
        print()
    
    # Validate inputs
    _validate_required_columns(data, sequence_col, peptide_col)
    
    # Prepare working data
    result_data = data.copy()
    result_data[sequence_col] = result_data[sequence_col].fillna('').astype(str)
    result_data[peptide_col] = result_data[peptide_col].fillna('').astype(str)
    
    if verbose:
        print("Finding peptide positions...")
    
    # Find all peptide positions (vectorized where possible)
    position_series = result_data.apply(
        lambda row: _find_all_positions(row[sequence_col], row[peptide_col]), 
        axis=1
    )
    
    # Calculate statistics for verbose output
    if verbose:
        _print_position_statistics(position_series, expand_multiple)
    
    # Calculate peptide lengths
    peptide_lengths = result_data[peptide_col].str.len()
    
    if expand_multiple:
        if verbose:
            print("Expanding multiple occurrences...")
        result_df = _expand_multiple_occurrences(
            result_data, position_series, peptide_lengths,
            peptide_start_col, peptide_end_col, original_index_col,
            occurrence_index_col, total_occurrences_col, add_occurrence_info
        )
    else:
        if verbose:
            print("Using first occurrence only...")
        result_df = _use_first_occurrence(
            result_data, position_series, peptide_lengths,
            peptide_start_col, peptide_end_col, total_occurrences_col, 
            add_occurrence_info, original_index_col
        )
    
    # Add unique identifiers
    if add_unique_id and protein_col and protein_col in result_df.columns:
        if verbose:
            print("Adding unique proteoform identifiers...")
        result_df[unique_id_col] = (
            result_df[protein_col].astype(str) + '-' +
            result_df[peptide_col].astype(str) + '-' +
            result_df[peptide_start_col].astype(str)
        )
    
    # Final statistics
    if verbose:
        _print_final_statistics(data, result_df, position_series, expand_multiple, peptide_start_col)

    # Order peptides by their positions within the protein if requested
    if position_ordered:
        # Build sorting columns - always include position columns
        sort_cols = [peptide_start_col, peptide_end_col]
        
        # Add protein column if available and not None
        if protein_col and protein_col in result_df.columns:
            sort_cols = [protein_col] + sort_cols
        
        if verbose:
            print(f"Ordering peptides by positions using columns: {sort_cols}")
        
        result_df = result_df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
    
    return result_df


# ======================================================================================
# Data Synchronization and Reindexing
# ======================================================================================

def reindex_quantitative_data(
    info_data: pd.DataFrame,
    quan_data: pd.DataFrame,
    original_index_col: str = 'original_index',
    remove_unmatched: bool = True,
    position_col: str = 'peptide_start',
    verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reindex quantitative data to match processed info data after position mapping.
    
    This function handles the synchronization of quantitative and info datasets
    after peptide position mapping, where info data may have been:
    - Filtered (unmatched peptides removed)
    - Expanded (multiple occurrences create additional rows)
    
    Args:
        info_data: Processed info DataFrame with original_index column
        quan_data: Original quantitative DataFrame with matching original indices
        original_index_col: Column name containing original indices
        remove_unmatched: Remove peptides that couldn't be positioned (default: True)
        position_col: Column name to check for successful positioning (default: 'peptide_start')
        verbose: Print detailed processing information
        
    Returns:
        Tuple of (filtered_info_data, reindexed_quan_data) with matching indices
        
    Raises:
        ValueError: If required columns not found or indices don't match
        
    Examples:
        >>> # After position mapping
        >>> processed_info = map_peptide_to_protein_positions(info_data, expand_multiple=True)
        >>> synced_info, synced_quan = reindex_quantitative_data(processed_info, quan_data)
        >>> assert len(synced_info) == len(synced_quan)
        >>> assert synced_info[position_col].notna().all()  # No unmatched peptides if remove_unmatched=True
    """
    if verbose:
        print("=" * 60)
        print("REINDEXING AND FILTERING QUANTITATIVE DATA")
        print("=" * 60)
        print(f"Info data shape: {info_data.shape}")
        print(f"Quantitative data shape: {quan_data.shape}")
        print(f"Remove unmatched peptides: {remove_unmatched}")
        print(f"Position column: '{position_col}'")
    
    # Validate inputs
    if original_index_col not in info_data.columns:
        raise ValueError(f"Column '{original_index_col}' not found in info_data. "
                        f"Available columns: {list(info_data.columns)}")
    
    if remove_unmatched and position_col not in info_data.columns:
        raise ValueError(f"Position column '{position_col}' not found in info_data. "
                        f"Available columns: {list(info_data.columns)}")
    
    # Filter out unmatched peptides if requested
    if remove_unmatched:
        # Identify unmatched peptides (NaN in position column)
        matched_mask = info_data[position_col].notna()
        unmatched_count = (~matched_mask).sum()
        
        if verbose and unmatched_count > 0:
            print(f"Filtering out {unmatched_count:,} unmatched peptides...")
        
        # Filter info data to keep only matched peptides
        filtered_info = info_data[matched_mask].reset_index(drop=True)
        
        if verbose:
            print(f"Filtered info data shape: {filtered_info.shape}")
    else:
        filtered_info = info_data.copy()
        unmatched_count = info_data[position_col].isna().sum() if position_col in info_data.columns else 0
    
    # Check if original indices are valid
    original_indices = filtered_info[original_index_col].dropna()
    if len(original_indices) == 0:
        raise ValueError("No valid original indices found after filtering")
    
    max_original_index = original_indices.max()
    
    if max_original_index >= len(quan_data):
        raise ValueError(f"Original index {max_original_index} exceeds quantitative data length {len(quan_data)}")
    
    if verbose:
        print(f"Original index column: '{original_index_col}'")
        print(f"Original indices range: {original_indices.min():.0f} to {max_original_index:.0f}")
        
        # Check for any remaining unmatched peptides (should be 0 if remove_unmatched=True)
        remaining_unmatched = filtered_info[original_index_col].isna().sum()
        if remaining_unmatched > 0:
            print(f"Remaining unmatched peptides (NaN original_index): {remaining_unmatched}")
    
    # Create reindexed quantitative data
    reindexed_quan = pd.DataFrame(index=filtered_info.index, columns=quan_data.columns)
    
    # For matched rows, copy quantitative data using original indices
    matched_mask = filtered_info[original_index_col].notna()
    if matched_mask.any():
        matched_original_indices = filtered_info.loc[matched_mask, original_index_col].astype(int)
        reindexed_quan.loc[matched_mask] = quan_data.iloc[matched_original_indices].values
    
    # For any remaining unmatched rows, leave as NaN (already initialized as such)
    if verbose:
        matched_rows = matched_mask.sum()
        remaining_unmatched_rows = (~matched_mask).sum()
        
        print()
        print("REINDEXING RESULTS:")
        print(f"  - Total output rows: {len(filtered_info):,}")
        print(f"  - Matched rows (copied quantitative data): {matched_rows:,}")
        print(f"  - Remaining unmatched rows (NaN values): {remaining_unmatched_rows:,}")
        
        if remove_unmatched and unmatched_count > 0:
            print(f"  - Removed unmatched peptides: {unmatched_count:,}")
        
        print(f"  - Output shape: {reindexed_quan.shape}")
        
        # Check for data integrity
        non_null_values = reindexed_quan.notna().sum().sum()
        total_possible_values = reindexed_quan.size
        fill_rate = non_null_values / total_possible_values * 100 if total_possible_values > 0 else 0
        print(f"  - Data fill rate: {fill_rate:.1f}% ({non_null_values:,}/{total_possible_values:,} values)")
        
        # Check for duplicated original indices (expansions)
        if matched_mask.any():
            expansion_factor = len(matched_original_indices) / len(matched_original_indices.unique())
            if expansion_factor > 1:
                print(f"  - Row expansion factor: {expansion_factor:.2f}x")
                duplicated_indices = matched_original_indices.value_counts()
                max_duplicates = duplicated_indices.max()
                print(f"  - Max duplications per original row: {max_duplicates}")
        
        print("=" * 60)
        print()

    # Clean-up the temporary columns
    filtered_info = filtered_info.drop(columns=[original_index_col], errors='ignore')
    
    return filtered_info, reindexed_quan

# ======================================================================================
# 
# ======================================================================================



# ======================================================================================
# Internal Helper Functions
# ======================================================================================

def _find_all_positions(protein_seq: str, peptide_seq: str) -> List[int]:
    """Find all starting positions of peptide in protein sequence."""
    if not protein_seq or not peptide_seq:
        return []
    
    positions = []
    start = 0
    while True:
        pos = protein_seq.find(peptide_seq, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def _expand_multiple_occurrences(
    result_data: pd.DataFrame,
    position_series: pd.Series,
    peptide_lengths: pd.Series,
    peptide_start_col: str,
    peptide_end_col: str,
    original_index_col: str,
    occurrence_index_col: str,
    total_occurrences_col: str,
    add_occurrence_info: bool
) -> pd.DataFrame:
    """Expand DataFrame to include multiple occurrences as separate rows."""
    # Create temporary DataFrame with positions
    temp_df = result_data.copy()
    temp_df['_positions'] = position_series
    temp_df['_peptide_length'] = peptide_lengths
    temp_df['_original_index'] = temp_df.index
    
    # Handle empty position lists
    temp_df['_positions'] = temp_df['_positions'].apply(
        lambda x: [np.nan] if not x else x
    )
    
    # Explode positions to create one row per occurrence
    exploded_df = temp_df.explode('_positions').reset_index(drop=True)
    
    # Calculate start and end positions (1-indexed), handling NaN positions
    # Handle the future warning by using explicit numeric conversion
    position_values = pd.to_numeric(exploded_df['_positions'], errors='coerce').fillna(-1)
    exploded_df[peptide_start_col] = np.where(
        exploded_df['_positions'].isna(), 
        np.nan, 
        position_values + 1
    )
    exploded_df[peptide_end_col] = np.where(
        exploded_df['_positions'].isna(), 
        np.nan, 
        position_values + exploded_df['_peptide_length']
    )
    exploded_df[original_index_col] = exploded_df['_original_index']
    
    if add_occurrence_info:
        # Add occurrence statistics
        occurrence_info = exploded_df.groupby('_original_index').cumcount() + 1
        total_counts = exploded_df.groupby('_original_index')['_original_index'].transform('size')
        
        exploded_df[occurrence_index_col] = occurrence_info
        exploded_df[total_occurrences_col] = total_counts
        
        # Handle no matches - for unmatched peptides, set occurrence to NaN and total to 0
        no_match_mask = exploded_df['_positions'].isna()
        exploded_df.loc[no_match_mask, occurrence_index_col] = np.nan
        exploded_df.loc[no_match_mask, total_occurrences_col] = 0
    
    # Clean up temporary columns
    return exploded_df.drop(['_positions', '_peptide_length', '_original_index'], axis=1)


def _use_first_occurrence(
    result_data: pd.DataFrame,
    position_series: pd.Series,
    peptide_lengths: pd.Series,
    peptide_start_col: str,
    peptide_end_col: str,
    total_occurrences_col: str,
    add_occurrence_info: bool,
    original_index_col: str
) -> pd.DataFrame:
    """Use only the first occurrence of each peptide."""
    result_df = result_data.copy()
    
    # Extract first positions or NaN
    first_positions = position_series.apply(lambda x: x[0] if x else np.nan)
    
    # Calculate start and end positions (1-indexed)
    result_df[peptide_start_col] = first_positions + 1
    result_df[peptide_end_col] = first_positions + peptide_lengths
    
    # Always add original index column for synchronization
    result_df[original_index_col] = result_df.index
    
    if add_occurrence_info:
        result_df[total_occurrences_col] = position_series.apply(len)
    
    return result_df


def _detect_column_name(data: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """Automatically detect column name from candidates."""
    available_cols = data.columns.tolist()
    
    for candidate in candidates:
        if candidate in available_cols:
            return candidate
    
    if required:
        raise ValueError(f"Could not find required column. Looked for: {candidates}. Available: {available_cols}")
    
    return None


def _validate_required_columns(data: pd.DataFrame, sequence_col: str, peptide_col: str) -> None:
    """Validate that required columns exist and contain valid data."""
    missing_cols = []
    if sequence_col not in data.columns:
        missing_cols.append(sequence_col)
    if peptide_col not in data.columns:
        missing_cols.append(peptide_col)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for empty data
    if data[sequence_col].isna().all():
        raise ValueError(f"Column '{sequence_col}' contains only NaN values")
    if data[peptide_col].isna().all():
        raise ValueError(f"Column '{peptide_col}' contains only NaN values")


def _print_position_statistics(position_series: pd.Series, expand_multiple: bool) -> None:
    """Print detailed statistics about peptide position finding."""
    total_entries = len(position_series)
    
    # Calculate occurrence statistics
    occurrence_counts = position_series.apply(len)
    unmatched = (occurrence_counts == 0).sum()
    matched = total_entries - unmatched
    
    # Multiple occurrence statistics
    multiple_occurrences = (occurrence_counts > 1).sum()
    single_occurrences = (occurrence_counts == 1).sum()
    
    # Total positions found
    total_positions = occurrence_counts.sum()
    
    print("POSITION FINDING RESULTS:")
    print(f"  - Total peptide entries: {total_entries:,}")
    print(f"  - Successfully matched: {matched:,} ({matched/total_entries*100:.3f}%)")
    print(f"  - Unmatched peptides: {unmatched:,} ({unmatched/total_entries*100:.3f}%)")
    print()
    
    if matched > 0:
        print("OCCURRENCE DETAILS:")
        print(f"  - Single occurrences: {single_occurrences:,}")
        print(f"  - Multiple occurrences: {multiple_occurrences:,}")
        print(f"  - Total positions found: {total_positions:,}")
        
        if multiple_occurrences > 0:
            max_occurrences = occurrence_counts.max()
            avg_occurrences = occurrence_counts[occurrence_counts > 0].mean()
            print(f"  - Max occurrences per peptide: {max_occurrences}")
            print(f"  - Average occurrences (matched): {avg_occurrences:.2f}")
        print()
    
    if expand_multiple:
        # Expected output: positioned rows + unmatched rows (with NaN positions)
        expected_output_rows = total_positions + unmatched
        print(f"EXPANSION INFO:")
        print(f"  - Expected output rows (with expansion): {expected_output_rows:,}")
        print(f"    → {total_positions:,} positioned rows + {unmatched:,} unmatched rows")
        if total_entries > 0:
            print(f"  - Row expansion factor: {expected_output_rows/total_entries:.2f}x")
    else:
        print(f"SINGLE OCCURRENCE INFO:")
        print(f"  - Output rows (no expansion): {total_entries:,}")
        if multiple_occurrences > 0:
            print(f"  - Multiple occurrences will use first position only")
    print()


def _print_final_statistics(
    input_data: pd.DataFrame, 
    result_df: pd.DataFrame, 
    position_series: pd.Series,
    expand_multiple: bool,
    peptide_start_col: str = 'peptide_start'
) -> None:
    """Print final processing statistics and summary."""
    input_rows = len(input_data)
    output_rows = len(result_df)
    
    # Calculate success based on original input rows, not expanded output
    original_matched = (position_series.apply(len) > 0).sum()  # Original peptides that had matches
    original_unmatched = input_rows - original_matched
    
    # Calculate actual positioned peptides in output
    positioned_peptides = result_df[peptide_start_col].notna().sum()
    unpositioned_peptides = result_df[peptide_start_col].isna().sum()
    
    print("FINAL PROCESSING SUMMARY:")
    print(f"  - Input rows: {input_rows:,}")
    print(f"  - Output rows: {output_rows:,}")
    print()
    
    if expand_multiple:
        print("POSITIONING RESULTS (EXPANDED MODE):")
        print(f"  - Original peptides matched: {original_matched:,} ({original_matched/input_rows*100:.3f}%)")
        print(f"  - Original peptides unmatched: {original_unmatched:,} ({original_unmatched/input_rows*100:.3f}%)")
        print(f"  - Total positioned peptides (expanded): {positioned_peptides:,}")
        print(f"  - Total unpositioned rows: {unpositioned_peptides:,}")
        
        # Verify the math
        total_positions = position_series.apply(len).sum()
        print(f"  - Expected positioned peptides: {total_positions:,}")
        print(f"  - Expected unpositioned rows: {original_unmatched:,}")
        print(f"  - Expected total output: {total_positions + original_unmatched:,}")
        
        if positioned_peptides != total_positions:
            print(f"  ⚠️  Warning: Positioned peptides mismatch! Expected {total_positions:,}, got {positioned_peptides:,}")
        if unpositioned_peptides != original_unmatched:
            print(f"  ⚠️  Warning: Unpositioned rows mismatch! Expected {original_unmatched:,}, got {unpositioned_peptides:,}")
    else:
        print("POSITIONING RESULTS (SINGLE OCCURRENCE MODE):")
        print(f"  - Successfully positioned peptides: {positioned_peptides:,} ({positioned_peptides/input_rows*100:.3f}%)")
        print(f"  - Unmatched peptides: {original_unmatched:,} ({original_unmatched/input_rows*100:.3f}%)")
    
    if expand_multiple and output_rows != input_rows:
        expansion_factor = output_rows / input_rows
        print(f"  - Row expansion factor: {expansion_factor:.2f}x")
    
    # Additional insights - use original matches for success rate
    if original_matched > 0:
        print()
        print("INSIGHTS:")
        
        # Multiple occurrences info
        if expand_multiple:
            total_positions = position_series.apply(len).sum()
            print(f"  - Total unique positions mapped: {total_positions:,}")
            
            # Average positions per successfully mapped peptide
            avg_positions = total_positions / original_matched
            print(f"  - Average positions per mapped peptide: {avg_positions:.2f}")
        
        # Check for any interesting patterns
        occurrence_counts = position_series.apply(len)
        if (occurrence_counts > 5).any():
            high_occurrence_count = (occurrence_counts > 5).sum()
            max_occur = occurrence_counts.max()
            print(f"  - Peptides with >5 occurrences: {high_occurrence_count:,} (max: {max_occur})")
    
    print("=" * 60)
    print()
