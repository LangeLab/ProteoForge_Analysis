#!/usr/bin/env python3
"""
Protein Sequence Coverage Calculator with Peptide Trace Assignment

A high-performance, multiprocessing-enabled tool for calculating protein sequence
coverage and assigning non-overlapping peptide traces based on positional data.
Optimized for both small and large-scale proteomics datasets with automatic
performance optimization and strict data integrity checking.

Version: 0.7.0
Date: 2025-07-01
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

import time
import warnings
from tqdm import tqdm
from typing import Union, Dict, List, Tuple
from multiprocessing import Pool, cpu_count, get_context

import numpy as np
import pandas as pd


# ======================================================================================
# Constants and Configuration
# ======================================================================================

# Default column names for coverage calculation
DEFAULT_COLUMNS = {
    'protein': 'Protein',
    'peptide': 'Peptide',
    'seq_length': 'seqLength',
    'start_pos': 'startpos',
    'end_pos': 'endpos'
}

# Multiprocessing thresholds (conservative approach)
MP_THRESHOLDS = {
    'min_proteins_for_mp': 1000,      # Minimum proteins to consider multiprocessing
    'min_peptides_for_mp': 5000,      # Minimum peptides to consider multiprocessing
    'small_dataset_max': 1000,        # Small dataset threshold
    'medium_dataset_max': 10000,      # Medium dataset threshold
    'max_processes': 8                 # Maximum number of processes
}


# ======================================================================================
# Performance and Multiprocessing Utilities
# ======================================================================================

def _estimate_dataset_size(info_data: pd.DataFrame, protein_col: str) -> Tuple[int, int]:
    """
    Estimate dataset characteristics for optimization.
    
    Args:
        info_data: Input DataFrame
        protein_col: Name of protein column
        
    Returns:
        Tuple of (n_proteins, n_peptides)
    """
    n_peptides = len(info_data)
    n_proteins = info_data[protein_col].nunique()
    return n_proteins, n_peptides


def _get_optimal_processing_params(
        n_proteins: int, 
        n_peptides: int
    ) -> Tuple[bool, int, int]:
    """
    Automatically determine optimal processing parameters based on dataset size.
    
    Conservative approach that only uses multiprocessing when the benefit 
    significantly outweighs the overhead.
    
    Args:
        n_proteins: Number of unique proteins
        n_peptides: Total number of peptides
        
    Returns:
        Tuple of (use_multiprocessing, n_processes, chunk_size)
    """
    # Conservative thresholds based on real-world performance
    if (n_proteins < MP_THRESHOLDS['min_proteins_for_mp'] or 
        n_peptides < MP_THRESHOLDS['min_peptides_for_mp']):
        # Small datasets: single process is usually faster
        return False, 1, 0
    elif n_proteins <= MP_THRESHOLDS['medium_dataset_max']:
        # Medium datasets: use minimal multiprocessing
        n_processes = min(4, max(2, cpu_count() // 4))
        chunk_size = max(50, n_proteins // (n_processes * 2))
        return True, n_processes, chunk_size
    else:
        # Large datasets: more aggressive but still conservative
        n_processes = min(MP_THRESHOLDS['max_processes'], max(2, cpu_count() // 2))
        chunk_size = max(100, n_proteins // (n_processes * 3))
        return True, n_processes, chunk_size


def _chunk_protein_list(
        protein_list: List[str], 
        chunk_size: int
    ) -> List[List[str]]:
    """
    Split protein list into chunks for parallel processing.
    
    Args:
        protein_list: List of protein identifiers
        chunk_size: Size of each chunk
        
    Returns:
        List of protein chunks
    """
    return [protein_list[i:i + chunk_size] for i in range(0, len(protein_list), chunk_size)]


def _process_protein_chunk(
        args: Tuple[List[str], pd.DataFrame, str, str, str, str, str, int, bool]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[Union[str, Tuple[str, int]], int]]]:
    """
    Process a chunk of proteins in parallel.
    
    Args:
        args: Tuple containing processing arguments
        
    Returns:
        Tuple of (protein_coverage_dict, protein_traces_dict)
    """
    (protein_chunk, data_subset, protein_col, peptide_col, 
     seq_len_col, start_pos_col, end_pos_col, gap, verbose) = args
    
    chunk_coverage = {}
    chunk_traces = {}
    
    for protein in protein_chunk:
        try:
            # Get protein subset
            if protein_col in data_subset.index.names:
                subset = data_subset.loc[protein]
            else:
                subset = data_subset[data_subset[protein_col] == protein]
            
            coverage, traces = _process_single_protein(
                protein, subset, seq_len_col, peptide_col, 
                start_pos_col, end_pos_col, gap
            )
            
            chunk_coverage[protein] = coverage
            chunk_traces[protein] = traces
            
        except Exception as e:
            if verbose:
                print(f"Warning: Error processing protein {protein}: {e}")
            chunk_coverage[protein] = 0.0
            chunk_traces[protein] = {}
    
    return chunk_coverage, chunk_traces


def _process_single_protein(
        protein: str,
        subset: Union[pd.Series, pd.DataFrame],
        seq_len_col: str,
        peptide_col: str,
        start_pos_col: str,
        end_pos_col: str,
        gap: int
    ) -> Tuple[float, Dict[Union[str, Tuple[str, int]], int]]:
    """
    Process a single protein for coverage and trace calculation.
    
    Args:
        protein: Protein identifier
        subset: Data subset for this protein
        seq_len_col: Sequence length column name
        peptide_col: Peptide column name
        start_pos_col: Start position column name
        end_pos_col: End position column name
        gap: Gap between peptides on same trace
        
    Returns:
        Tuple of (coverage_percentage, peptide_trace_mapping)
        Note: For multiple occurrences of same peptide, keys are (peptide, index) tuples
    """
    trace_map = {}
    
    if not isinstance(subset, pd.DataFrame):
        # Single peptide case
        seq_len = subset[seq_len_col]
        start_idx, end_idx = _sanitize_positions(
            subset[start_pos_col], subset[end_pos_col], seq_len
        )
        peptide_len = max(0, end_idx - start_idx)
        coverage_perc = (peptide_len / seq_len) * 100 if seq_len > 0 else 0
        trace_map = {subset[peptide_col]: 0}
    else:
        # Multiple peptides case
        subset = subset.sort_values([start_pos_col, end_pos_col]).reset_index(drop=True)
        seq_len = subset[seq_len_col].iloc[0]
        
        if seq_len > 0:
            try:
                coverage_array = np.zeros(max(1, int(seq_len)), dtype=np.int8)
                traces = []
                trace_ends = [-gap - 1]  # Initialize with impossible position
                
                # Track peptide occurrences for proper indexing
                peptide_counts = {}
                
                for idx, row in enumerate(subset.itertuples()):
                    start_idx, end_idx = _sanitize_positions(
                        getattr(row, start_pos_col), 
                        getattr(row, end_pos_col), 
                        seq_len
                    )
                    
                    # Update coverage array
                    if start_idx < len(coverage_array) and end_idx <= len(coverage_array):
                        coverage_array[start_idx:end_idx] = 1
                    
                    # For trace assignment, use original 1-based positions
                    start_pos_orig = int(getattr(row, start_pos_col))
                    end_pos_orig = int(getattr(row, end_pos_col))
                    
                    # Assign trace
                    assigned_trace = None
                    for i in range(len(trace_ends)):
                        if start_pos_orig >= trace_ends[i] + gap:
                            assigned_trace = i
                            trace_ends[i] = end_pos_orig
                            break
                    
                    if assigned_trace is None:
                        assigned_trace = len(trace_ends)
                        trace_ends.append(end_pos_orig)
                    
                    traces.append(assigned_trace)
                    
                    # Create unique key for each peptide occurrence
                    peptide = getattr(row, peptide_col)
                    if peptide in peptide_counts:
                        peptide_counts[peptide] += 1
                        key = (peptide, peptide_counts[peptide])
                    else:
                        peptide_counts[peptide] = 0
                        key = peptide  # First occurrence uses simple peptide name
                    
                    trace_map[key] = assigned_trace
                
                coverage_perc = (coverage_array.sum() / seq_len) * 100
                
            except (MemoryError, ValueError):
                coverage_perc = 0
                # Fallback: assign sequential traces
                for i, peptide in enumerate(subset[peptide_col]):
                    trace_map[peptide] = i
        else:
            coverage_perc = 0
            for i, peptide in enumerate(subset[peptide_col]):
                trace_map[peptide] = i
    
    return max(0, min(100, coverage_perc)), trace_map


def _validate_input_data(
    info_data: pd.DataFrame,
    protein_col: str,
    peptide_col: str,
    seq_len_col: str,
    start_pos_col: str,
    end_pos_col: str,
    unique_id_col: str,
    gap: int
) -> None:
    """Validates input parameters and data for the coverage calculation function."""
    if not isinstance(info_data, pd.DataFrame):
        raise TypeError(f"info_data must be a pandas DataFrame, got {type(info_data)}")
    
    if info_data.empty:
        warnings.warn("Input DataFrame is empty")
        return
    
    # Check required columns exist
    required_cols = [protein_col, peptide_col, seq_len_col, start_pos_col, end_pos_col, unique_id_col]
    missing_cols = [col for col in required_cols if col not in info_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate gap parameter
    if not isinstance(gap, int) or gap < 0:
        raise ValueError(f"gap must be a non-negative integer, got {gap}")
    
    # Check for null values in critical columns
    null_counts = info_data[required_cols].isnull().sum()
    if null_counts.any():
        warnings.warn(f"Null values found in columns: {null_counts[null_counts > 0].to_dict()}")
    
    # Validate numeric columns
    numeric_cols = [seq_len_col, start_pos_col, end_pos_col]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(info_data[col]):
            warnings.warn(f"Column {col} is not numeric, attempting conversion")
            try:
                info_data[col] = pd.to_numeric(info_data[col], errors='coerce')
            except Exception as e:
                raise ValueError(f"Failed to convert {col} to numeric: {e}")
    
    # Check for negative or zero positions (1-based positions should be >= 1)
    if (info_data[start_pos_col] < 1).any():
        raise ValueError("Start positions should be >= 1 (1-based protein sequence positions)")
    
    if (info_data[end_pos_col] < 1).any():
        raise ValueError("End positions should be >= 1 (1-based protein sequence positions)")
    
    # Check start <= end
    invalid_positions = info_data[start_pos_col] > info_data[end_pos_col]
    if invalid_positions.any():
        n_invalid = invalid_positions.sum()
        warnings.warn(f"Found {n_invalid} peptides with start > end positions")
        if n_invalid > len(info_data) * 0.1:  # More than 10% invalid
            raise ValueError(f"Too many invalid positions ({n_invalid}/{len(info_data)})")
    
    # Check positions within sequence bounds (1-based positions should be <= sequence length)
    out_of_bounds = (info_data[end_pos_col] > info_data[seq_len_col])
    if out_of_bounds.any():
        n_oob = out_of_bounds.sum()
        warnings.warn(f"Found {n_oob} peptides with positions beyond sequence length")
        if n_oob > len(info_data) * 0.1:  # More than 10% out of bounds
            raise ValueError(f"Too many out-of-bounds positions ({n_oob}/{len(info_data)})")


def _sanitize_positions(
    start_pos: Union[int, float], 
    end_pos: Union[int, float], 
    seq_len: Union[int, float]
) -> tuple[int, int]:
    """
    Sanitizes and validates peptide positions.
    
    Converts from 1-based protein sequence positions to 0-based array indices.
    Input positions are assumed to be 1-based (typical in protein sequences).
    Returns 0-based indices suitable for numpy array indexing.
    """
    # Handle NaN values
    if pd.isna(start_pos) or pd.isna(end_pos) or pd.isna(seq_len):
        return 0, 0
    
    # Convert to integers and handle 1-based to 0-based conversion
    start_pos = max(1, int(start_pos))  # Ensure at least position 1
    end_pos = max(1, int(end_pos))      # Ensure at least position 1
    seq_len = max(1, int(seq_len))
    
    # Ensure start <= end for 1-based positions
    if start_pos > end_pos:
        start_pos = end_pos
    
    # Clamp to sequence bounds (1-based)
    start_pos = min(start_pos, seq_len)
    end_pos = min(end_pos, seq_len)
    
    # Convert to 0-based indices for array indexing
    start_idx = start_pos - 1  # Convert to 0-based
    end_idx = end_pos          # end_pos is exclusive in array slicing, so keep as-is after conversion
    
    # Ensure valid array indices
    start_idx = max(0, start_idx)
    end_idx = min(end_idx, seq_len)
    
    # Final check: ensure start_idx < end_idx for valid array slice
    if start_idx >= end_idx:
        end_idx = start_idx + 1
    
    return start_idx, end_idx

def calculate_protein_coverage(
    info_data: pd.DataFrame,
    protein_col: str = "Protein",
    peptide_col: str = "Peptide",
    seq_len_col: str = "seqLength",
    start_pos_col: str = "startpos",
    end_pos_col: str = "endpos",
    unique_id_col: str = "unique_id",
    gap: int = 0,
    validate_input: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Calculates protein sequence coverage and assigns non-overlapping peptide traces
    based on provided start and end positions.

    High-performance implementation with automatic optimization for optimal
    processing method (single-process vs multiprocess) based on dataset size.
    All processing parameters are automatically determined for best performance.

    IMPORTANT: This function expects 1-based protein sequence positions (start_pos and end_pos).
    Position 1 corresponds to the first amino acid in the protein sequence.

    Args:
        info_data: The input DataFrame containing peptide data
        protein_col: The name of the column for protein identifiers
        peptide_col: The name of the column for peptide identifiers
        seq_len_col: The name of the column for protein sequence length
        start_pos_col: The name of the column for peptide start position (1-based)
        end_pos_col: The name of the column for peptide end position (1-based, inclusive)
        unique_id_col: The name of the column containing unique identifiers for Protein-Peptide-Position combinations
                       If duplicates exist in this column, an error will be raised with detailed information
        gap: The minimum allowed gap between peptides on the same trace
        validate_input: Whether to validate input data
        n_jobs: Number of parallel jobs ('auto', -1 for all cores, or specific number)
        verbose: If True, print information about processing; if False, be silent

    Returns:
        A new DataFrame with added 'trace' and 'Cov%' columns, preserving original row order
        
    Examples:
        >>> # Basic usage (automatic optimization)
        >>> df = calculate_protein_coverage(peptide_data, unique_id_col='unique_id')
        
        >>> # With custom gap and multiprocessing
        >>> df = calculate_protein_coverage(
        ...     peptide_data, 
        ...     unique_id_col='peptide_unique_id',
        ...     gap=2, 
        ...     n_jobs=4, 
        ...     verbose=True
        ... )
        
        >>> # Force single-process mode
        >>> df = calculate_protein_coverage(peptide_data, unique_id_col='unique_id', n_jobs=1)
    """
    # Input validation
    if validate_input:
        _validate_input_data(info_data, protein_col, peptide_col, seq_len_col, 
                           start_pos_col, end_pos_col, unique_id_col, gap)
    
    # Handle empty DataFrame
    if info_data.empty:
        return info_data.assign(**{'Cov%': pd.Series(dtype='float64'), 'trace': pd.Series(dtype='Int64')})

    processed_data = info_data.copy()
    
    # Check for duplicates in unique_id column - this should never happen with properly processed data
    duplicates = processed_data[unique_id_col].duplicated()
    if duplicates.any():
        n_dups = duplicates.sum()
        duplicate_ids = processed_data.loc[duplicates, unique_id_col].tolist()
        duplicate_details = processed_data[processed_data[unique_id_col].isin(duplicate_ids)]
        
        print(f"\nERROR: Found {n_dups} duplicate entries in '{unique_id_col}' column!")
        print(f"Duplicate unique_id values: {duplicate_ids}")
        print("\nDuplicate rows details:")
        print(duplicate_details[[protein_col, peptide_col, start_pos_col, end_pos_col, unique_id_col]])
        
        raise ValueError(
            f"Data integrity error: Found {n_dups} duplicate entries in '{unique_id_col}' column. "
            f"Each row should have a unique identifier for Protein-Peptide-Position combination. "
            f"Please check your data preprocessing step."
        )
    
    
    # Estimate dataset size and determine processing strategy
    n_proteins, n_peptides = _estimate_dataset_size(processed_data, protein_col)
    
    # Handle n_jobs parameter
    if n_jobs == 'auto':
        use_multiprocessing, n_processes, chunk_size = _get_optimal_processing_params(n_proteins, n_peptides)
    elif n_jobs == -1:
        use_multiprocessing = True
        n_processes = cpu_count()
        chunk_size = max(50, n_proteins // (n_processes * 2))
    elif isinstance(n_jobs, int) and n_jobs > 1:
        use_multiprocessing = True
        n_processes = min(n_jobs, cpu_count())
        chunk_size = max(50, n_proteins // (n_processes * 2))
    else:
        use_multiprocessing = False
        n_processes = 1
        chunk_size = 0
    
    if verbose:
        print(f"Processing {n_proteins} proteins with {n_peptides} total peptides")
        if use_multiprocessing:
            print(f"Auto-selected: multiprocessing with {n_processes} processes")
        else:
            print("Auto-selected: single-process mode (optimal for this dataset size)")
    
    # Ensure proper indexing for processing
    if processed_data.index.name != protein_col:
        processed_data = processed_data.reset_index().set_index(protein_col)

    unique_proteins = processed_data.index.unique().tolist()
    
    # Process based on auto-determined method
    if use_multiprocessing and len(unique_proteins) > 1:
        # Multiprocessing path with performance monitoring
        if verbose:
            print("Starting multiprocessing coverage calculation...")
        
        start_time = time.time()
        
        # Split proteins into chunks
        protein_chunks = _chunk_protein_list(unique_proteins, chunk_size)
        
        # Prepare arguments for worker processes
        chunk_args = [
            (chunk, processed_data, protein_col, peptide_col, 
             seq_len_col, start_pos_col, end_pos_col, gap, verbose)
            for chunk in protein_chunks
        ]
        
        # Process chunks in parallel using fork (faster) with spawn fallback
        try:
            ctx = get_context('fork')
        except RuntimeError:
            ctx = get_context('spawn')
        
        with ctx.Pool(n_processes) as pool:
            chunk_results = pool.map(_process_protein_chunk, chunk_args)
        
        # Combine results from all chunks
        protein_coverage = {}
        protein_peptide_traces = {}
        
        for chunk_coverage, chunk_traces in chunk_results:
            protein_coverage.update(chunk_coverage)
            protein_peptide_traces.update(chunk_traces)
        
        processing_time = time.time() - start_time
        if verbose:
            print(f"Multiprocessing completed in {processing_time:.2f}s")
    
    else:
        # Single-process path
        if verbose and use_multiprocessing:
            print("Falling back to single-process mode (too few proteins for multiprocessing benefit)")
        
        protein_coverage = {}
        protein_peptide_traces = {}
        
        progress_iter = tqdm(unique_proteins, desc="Calculating Coverage") if verbose else unique_proteins
        
        for protein in progress_iter:
            try:
                subset = processed_data.loc[protein]
                coverage, traces = _process_single_protein(
                    protein, subset, seq_len_col, peptide_col, 
                    start_pos_col, end_pos_col, gap
                )
                protein_coverage[protein] = coverage
                protein_peptide_traces[protein] = traces
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Error processing protein {protein}: {e}")
                protein_coverage[protein] = 0.0
                protein_peptide_traces[protein] = {}

    # Apply results to dataframe - preserve original row order
    if verbose:
        print("Applying results to DataFrame...")
    
    # Efficient coverage mapping using vectorized operations
    processed_data['Cov%'] = processed_data.index.map(protein_coverage).fillna(0)
    
    # Optimized trace assignment - avoid row-by-row iteration
    # Reset index to work with original data
    processed_data_with_orig_index = processed_data.reset_index()
    
    # Create an efficient trace mapping
    if verbose and len(protein_peptide_traces) > 10000:
        print("Building trace mappings (large dataset)...")
    
    trace_map = {}
    
    # Build comprehensive trace mapping in one pass
    for protein, peptide_traces in protein_peptide_traces.items():
        for key, trace_val in peptide_traces.items():
            if isinstance(key, tuple):
                # Handle (peptide, occurrence_index) tuple keys
                peptide, occurrence_idx = key
                trace_map[(protein, peptide, occurrence_idx)] = trace_val
            else:
                # Handle simple peptide keys (first occurrence)
                trace_map[(protein, key, 0)] = trace_val
    
    # Efficient vectorized trace assignment using pandas operations
    protein_series = processed_data_with_orig_index[protein_col]
    peptide_series = processed_data_with_orig_index[peptide_col]
    
    # Create occurrence indices efficiently
    occurrence_indices = processed_data_with_orig_index.groupby([protein_col, peptide_col]).cumcount()
    
    # Create trace lookup keys
    trace_keys = list(zip(protein_series, peptide_series, occurrence_indices))
    
    # Vectorized lookup
    traces_list = [trace_map.get(key, np.nan) for key in trace_keys]
    
    processed_data_with_orig_index['trace'] = traces_list
    
    # Final DataFrame preparation
    if verbose:
        print("Finalizing results...")
    
    # Remove the temporary ordering column if it exists
    if '_orig_row_order' in processed_data_with_orig_index.columns:
        result_data = processed_data_with_orig_index.drop('_orig_row_order', axis=1)
    else:
        result_data = processed_data_with_orig_index
    
    # Ensure proper data types
    if 'trace' in result_data.columns:
        result_data['trace'] = result_data['trace'].astype('Int64')

    if verbose:
        total_coverage = result_data.groupby(protein_col)['Cov%'].first()
        avg_coverage = total_coverage.mean()
        print(f"Average protein coverage: {avg_coverage:.1f}%")
        print(f"Generated DataFrame with {len(result_data)} rows and {len(result_data.columns)} columns")

    return result_data
