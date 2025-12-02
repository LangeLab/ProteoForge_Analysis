"""
Peptide Overlap Processor

A high-performance tool for grouping overlapping peptides and aggregating 
quantitative data across protein groups. Uses optimized algorithms for 
large-scale proteomics data processing.

Version: 0.7.0
Date: 2025-07-01
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

import time
from collections import defaultdict
from typing import List, Tuple, Dict, Set

import numpy as np
import pandas as pd

# ======================================================================================
# Core Algorithm Functions
# ======================================================================================

def _find_overlapping_peptide_groups_optimized(
    peptides: List[Tuple[int, int]], 
    max_diff: int = 3
) -> List[Set[int]]:
    """
    Efficiently find groups of overlapping peptides using Union-Find.
    
    Uses a sweep line algorithm with Union-Find data structure for O(n log n)
    complexity instead of naive O(n²) approach.
    
    Args:
        peptides: List of (start_pos, end_pos) tuples
        max_diff: Maximum allowed difference for overlap detection
        
    Returns:
        List of sets, each containing indices of overlapping peptides
    """
    n = len(peptides)
    if n == 0:
        return []
    
    # Sort peptides by start position for sweep line algorithm
    indexed_peptides = [(start, end, idx) for idx, (start, end) in enumerate(peptides)]
    indexed_peptides.sort()
    
    # Union-Find with path compression
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Sweep line: only check peptides that could potentially overlap
    for i in range(n):
        start1, end1, idx1 = indexed_peptides[i]
        
        for j in range(i + 1, n):
            start2, end2, idx2 = indexed_peptides[j]
            
            # Early termination optimization
            if start2 > end1 + max_diff:
                break
                
            # Check overlap condition
            if (abs(start1 - start2) <= max_diff and abs(end1 - end2) <= max_diff):
                union(idx1, idx2)
    
    # Collect connected components
    groups_dict = defaultdict(set)
    for i in range(n):
        root = find(i)
        groups_dict[root].add(i)
    
    return list(groups_dict.values())


def _select_group_leaders(
    group_indices: np.ndarray, 
    start_positions: np.ndarray, 
    end_positions: np.ndarray
) -> int:
    """
    Select the leader (longest peptide) for a group of overlapping peptides.
    
    Args:
        group_indices: Array of indices for peptides in the group
        start_positions: Array of all start positions
        end_positions: Array of all end positions
        
    Returns:
        Index of the longest peptide in the group
    """
    if len(group_indices) == 1:
        return group_indices[0]
    
    # Vectorized length calculation
    group_lengths = end_positions[group_indices] - start_positions[group_indices]
    max_length_pos = np.argmax(group_lengths)
    return group_indices[max_length_pos]


def _aggregate_quantitative_data(
    quan_array: np.ndarray,
    is_nan_mask: np.ndarray,
    group_leader_map: Dict[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficiently aggregate quantitative data using numpy operations.
    
    Args:
        quan_array: Quantitative data array (peptides x samples)
        is_nan_mask: Boolean mask for NaN values
        group_leader_map: Mapping from peptide index to group leader
        
    Returns:
        Tuple of (aggregated_data, unique_leaders)
    """
    n_peptides = len(group_leader_map)
    
    # Create efficient grouping labels
    grouping_labels = np.empty(n_peptides, dtype=np.int32)
    for idx, leader in group_leader_map.items():
        grouping_labels[idx] = leader
    
    unique_leaders, integer_codes = np.unique(grouping_labels, return_inverse=True)
    n_groups = len(unique_leaders)
    n_samples = quan_array.shape[1]
    
    # Pre-allocate output arrays
    aggregated_array = np.zeros((n_groups, n_samples), dtype=np.float64)
    aggregated_nan_counts = np.zeros((n_groups, n_samples), dtype=np.int32)
    
    # Single-pass aggregation using advanced indexing
    np.add.at(aggregated_array, integer_codes, quan_array)
    np.add.at(aggregated_nan_counts, integer_codes, is_nan_mask)
    
    # Restore NaN values where all inputs were NaN
    group_sizes = np.bincount(integer_codes, minlength=n_groups)
    all_nan_mask = aggregated_nan_counts == group_sizes[:, np.newaxis]
    aggregated_array[all_nan_mask] = np.nan
    
    return aggregated_array, unique_leaders


# ======================================================================================
# Main Processing Function
# ======================================================================================

def process_peptide_overlaps(
    info_df: pd.DataFrame,
    quan_df: pd.DataFrame,
    protein_col: str = 'Protein',
    startpos_col: str = 'startpos',
    endpos_col: str = 'endpos',
    max_diff: int = 3,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process peptide data to group overlaps and aggregate quantitative values.
    
    Groups overlapping peptides within each protein based on position similarity,
    selects the longest peptide as the representative, and aggregates quantitative
    data across grouped peptides.
    
    Args:
        info_df: DataFrame with peptide information including positions
        quan_df: DataFrame with quantitative data (aligned with info_df)
        protein_col: Column name for protein identifiers
        startpos_col: Column name for peptide start positions
        endpos_col: Column name for peptide end positions  
        max_diff: Maximum allowed difference for overlap detection
        verbose: Enable detailed progress output
        
    Returns:
        Tuple of (processed_info_df, processed_quan_df) with matching indices
        
    Raises:
        ValueError: If required columns are missing
        
    Example:
        >>> info_df = pd.DataFrame({
        ...     'Protein': ['P1', 'P1', 'P1'],
        ...     'Peptide': ['A', 'B', 'C'], 
        ...     'startpos': [10, 12, 50],
        ...     'endpos': [20, 22, 60]
        ... })
        >>> quan_df = pd.DataFrame({
        ...     'Sample1': [100, 110, 200],
        ...     'Sample2': [105, 115, 210]
        ... })
        >>> info_result, quan_result = process_peptide_overlaps(info_df, quan_df)
    """
    start_time = time.time() if verbose else None
    
    # Input validation
    required_cols = [protein_col, startpos_col, endpos_col]
    missing_cols = [col for col in required_cols if col not in info_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in info_df: {missing_cols}")
    
    if len(info_df) != len(quan_df):
        raise ValueError(f"Length mismatch: info_df ({len(info_df)}) vs quan_df ({len(quan_df)})")
    
    if verbose:
        print(f"Processing {len(info_df)} peptides across {info_df[protein_col].nunique()} proteins")
    
    # Setup processing DataFrames
    if info_df.index.name == protein_col:
        info_df_proc = info_df
        quan_df_proc = quan_df
    else:
        info_df_proc = info_df.set_index(protein_col)
        quan_df_proc = quan_df.set_index(info_df_proc.index)
    
    # One-time data type optimization
    if quan_df_proc.dtypes.eq('object').any():
        if verbose:
            print("Converting object columns to numeric...")
        object_cols = quan_df_proc.select_dtypes(include=['object']).columns
        for col in object_cols:
            quan_df_proc[col] = pd.to_numeric(quan_df_proc[col], errors='coerce')
    
    # Process each protein group
    processed_info_list = []
    processed_quan_list = []
    total_groups_processed = 0
    
    for group_name, info_group in info_df_proc.groupby(protein_col):
        quan_group = quan_df_proc.loc[group_name]
        
        # Handle single peptide case
        info_group = info_group.reset_index(drop=True)
        if isinstance(quan_group, pd.Series):
            quan_group = quan_group.to_frame().T
        quan_group = quan_group.reset_index(drop=True)
        
        # Fast quantitative data processing
        quan_array = quan_group.to_numpy(dtype=np.float64)
        is_nan_mask = np.isnan(quan_array)
        quan_array[is_nan_mask] = 0.0
        
        # Find overlapping groups
        start_positions = info_group[startpos_col].to_numpy()
        end_positions = info_group[endpos_col].to_numpy()
        peptides_pos = list(zip(start_positions, end_positions))
        
        peptide_groups = _find_overlapping_peptide_groups_optimized(peptides_pos, max_diff)
        if not peptide_groups:
            continue
        
        # Select group leaders and create mapping
        group_leader_map = {}
        for group_set in peptide_groups:
            group_indices = np.array(list(group_set))
            leader_idx = _select_group_leaders(group_indices, start_positions, end_positions)
            
            for member_idx in group_indices:
                group_leader_map[member_idx] = leader_idx
        
        # Aggregate quantitative data
        aggregated_array, unique_leaders = _aggregate_quantitative_data(
            quan_array, is_nan_mask, group_leader_map
        )
        
        # Create result DataFrames with shared indices
        aggregated_quan = pd.DataFrame(aggregated_array, columns=quan_group.columns)
        aggregated_info = info_group.iloc[unique_leaders].copy()
        aggregated_info[protein_col] = group_name
        
        # Generate consistent unique identifiers
        if 'unique_id' in aggregated_info.columns:
            shared_ids = aggregated_info['unique_id'].values
        else:
            shared_ids = [f"{group_name}_leader_{idx}" for idx in unique_leaders]
        
        aggregated_info.index = shared_ids
        aggregated_quan.index = shared_ids
        
        processed_info_list.append(aggregated_info)
        processed_quan_list.append(aggregated_quan)
        total_groups_processed += len(unique_leaders)
    
    # Combine results
    if not processed_info_list:
        if verbose:
            print("Warning: No overlapping peptide groups found")
        return pd.DataFrame(), pd.DataFrame()
    
    final_info_df = pd.concat(processed_info_list, ignore_index=False, sort=False)
    final_quan_df = pd.concat(processed_quan_list, ignore_index=False, sort=False)
    
    # Ensure index alignment
    if not final_info_df.index.equals(final_quan_df.index):
        if verbose:
            print("Warning: Reordering quantitative data for index alignment")
        final_quan_df = final_quan_df.reindex(final_info_df.index)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f}s: {len(final_info_df)} representative peptides from {total_groups_processed} groups")
    
    return final_info_df, final_quan_df

