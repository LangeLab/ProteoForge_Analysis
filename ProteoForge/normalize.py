#!/usr/bin/env python3
"""
ProteoForge Normalization Module
This module provides a function to normalize proteomics data against a specified condition.
It is designed to work with long-format data, pivoting it to wide format for normalization,
and then returning it back to long format with normalized intensity values.

Version: 0.7.0
Date: 2025-07-01
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

from typing import Optional
from typing import Dict, List

import numpy as np
import pandas as pd

# ======================================================================================
# Global Variables and Settings
# ======================================================================================


# ======================================================================================
# Normalization Functions
# ======================================================================================

def against_condition(
        long_data: pd.DataFrame,                   
        cond_run_dict: dict,                       
        run_col: str = "filename",                 
        index_cols: list = ["protein_id", "peptide_id"], 
        norm_against: str = "day1",                
        intensity_col: str = "intensity",          
        is_log2: bool = False,                     
        norm_intensity_col: str = "ms1adj"         

    ) -> pd.DataFrame:
    """
        Function to normalize the data against a condition for my method

        Args:
            long_data (pd.DataFrame): Long data with intensity values
            cond_run_dict (dict): Dictionary with condition: [run1, run2, ...]
            run_col (str): Column with the run names (used for pivoting)
            index_cols (list): Used for wide data and merging
            norm_against (str): Has to be one of the conditions
            intensity_col (str): Column with the intensity values
            is_log2 (bool): If the data is log2 transformed already
            norm_intensity_col (str): Column name for the normalized intensities

        Returns:
            pd.DataFrame: Long data with normalized intensities

        Raises:
            ValueError: If run_col, intensity_col, or index_cols are not in the long_data
            ValueError: If norm_against is not in the cond_run_dict keys

        Examples:
    """
    
    # index_cols must be a list
    if not isinstance(index_cols, list):
        index_cols = [index_cols]

    # check necessary columns if they are in the long_data
    for col in [run_col, intensity_col, *index_cols]:
        if col not in long_data.columns:
            raise ValueError(f"{col} not found in the columns of long_data")
        
    # Check if norm_against is key in the cond_run_dict with list of values
    if norm_against not in cond_run_dict.keys():
        raise ValueError(f"{norm_against} not found in the cond_run_dict keys")
    else:
        normCols = cond_run_dict[norm_against]
        
    # Move to wide for faster calculations
    wide_data = long_data.pivot_table(
        index=index_cols,
        columns=[run_col],
        values=intensity_col,
    )
    if not is_log2:
        wide_data = np.log2(wide_data)

    # Center the data by
    wide_data = (wide_data - wide_data.mean()) / wide_data.std()
    # Calculate the row-means for norm_against samples
    cntrRowMean = wide_data[normCols].mean(axis=1)
    # Subtract the row-means from the data
    wide_data = wide_data.sub(cntrRowMean, axis=0)

    # Return the data back to long format
    normalized_long = wide_data.reset_index().melt(
        id_vars=index_cols,
        var_name=run_col,
        value_name=norm_intensity_col,
    )

    # Merge the normalized data with the original data
    long_data = pd.merge(
        # Original data
        long_data,
        # Return the data back to long format
        normalized_long,
        # Merge on the index_cols and filename
        on=index_cols + [run_col],
        # Use left join
        how="left",
    )

    return long_data


def by_median_centering(
    df: pd.DataFrame,
    rescale_to_original_magnitude: bool = True,
    condition_map: Optional[Dict[str, List[str]]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Applies median centering to a pandas DataFrame, either globally or per condition.

    This method normalizes each sample (column) by its median value.
    If a condition_map is provided, normalization is applied independently
    to each group of columns defined in the map.

    Args:
        df (pd.DataFrame): The input data. Assumes features are rows and
                           samples are columns.
        rescale_to_original_magnitude (bool): If True, multiplies the centered
            data by a scaling factor to restore its original magnitude.
            This is applied globally or per-condition based on `condition_map`.
        condition_map (Optional[Dict[str, List[str]]]): A dictionary mapping a
            condition name to a list of column names belonging to that condition.
            If None, global normalization is performed. Defaults to None.
        verbose (bool): If True, prints status messages during execution.
                        Defaults to True.

    Returns:
        pd.DataFrame: The normalized data.

    Raises:
        ValueError: If the normalization process results in negative values,
                    introduces new missing values, or if columns in
                    condition_map are not in the DataFrame.
    """
    indent = "  "
    if verbose:
        print("\nApplying Median Centering...")
    # Make a copy to avoid modifying the original DataFrame
    data_to_normalize = df.copy()
    initial_missing_count = data_to_normalize.isna().sum().sum()

    if condition_map is None:
        # --- Global Normalization (Original Logic) ---
        if verbose:
            print(f"{indent}- Applying global median centering...")
        col_medians = data_to_normalize.median(axis=0)
        if (col_medians == 0).any():
            if verbose:
                print(
                    f"{indent*2}- Warning: One or more columns have a median of zero. "
                    "Replacing with 1 to avoid division by zero."
                )
            col_medians[col_medians == 0] = 1
        normalized_df = data_to_normalize / col_medians

        if rescale_to_original_magnitude:
            global_scaler = 2**(int(np.log2(np.nanmedian(data_to_normalize.mean(axis=1)))))
            if verbose:
                print(f"{indent*2}- Rescaling by a global factor of: {global_scaler:.2f}")
            normalized_df *= global_scaler
    else:
        # --- Per-Condition Normalization (New Logic) ---
        if verbose:
            print(f"{indent}- Applying per-condition median centering...")
        normalized_parts = []
        all_mapped_columns = []

        for condition, columns in condition_map.items():
            # Check that all specified columns exist in the DataFrame
            if not all(col in data_to_normalize.columns for col in columns):
                missing = [col for col in columns if col not in data_to_normalize.columns]
                raise ValueError(f"Columns for condition '{condition}' not found in DataFrame: {missing}")

            if verbose:
                print(f"{indent*2}- Normalizing condition '{condition}'...")
            condition_df = data_to_normalize[columns]
            all_mapped_columns.extend(columns)

            # 1. Median Centering for the condition
            col_medians = condition_df.median(axis=0)
            if (col_medians == 0).any():
                if verbose:
                    print(f"{indent*3}- Warning: Columns in '{condition}' have a median of zero. Replacing with 1.")
                col_medians[col_medians == 0] = 1
            normalized_part = condition_df / col_medians

            # 2. Optional Rescaling for the condition
            if rescale_to_original_magnitude:
                scaler = 2**(int(np.log2(np.nanmedian(condition_df.mean(axis=1)))))
                if verbose:
                    print(f"{indent*3}- Rescaling by a factor of: {scaler:.2f}")
                normalized_part *= scaler

            normalized_parts.append(normalized_part)

        # Concatenate all the normalized parts back together
        normalized_df = pd.concat(normalized_parts, axis=1)
        
        # Handle unmapped columns by carrying them over unmodified
        unmapped_columns = [col for col in df.columns if col not in all_mapped_columns]
        if unmapped_columns:
            if verbose:
                print(f"{indent*2}- Carrying over unmapped columns: {unmapped_columns}")
            normalized_df = pd.concat([normalized_df, df[unmapped_columns]], axis=1)

        # Ensure the final DataFrame has the same column order as the input
        normalized_df = normalized_df[df.columns]

    # --- Robustness Check for Negative Values ---
    if (normalized_df < 0).any().any():
        raise ValueError(
            "Normalization resulted in negative values. "
            "This is unexpected and may indicate issues with the input data."
        )

    # --- Robustness Check for New Missing Values ---
    final_missing_count = normalized_df.isna().sum().sum()
    if final_missing_count > initial_missing_count:
        raise ValueError(
            f"Normalization introduced {final_missing_count - initial_missing_count} new missing values. "
            "This should not happen."
        )

    # --- Post-Normalization Check ---
    if verbose:
        print(f"{indent}- Data after median centering has shape: {normalized_df.shape}")
        print(f"{indent}- Missing values (Before/After): ({initial_missing_count}/{final_missing_count})")

    return normalized_df

def build_robust_intensity(
    df: pd.DataFrame,
    log_col: str = 'log10Intensity',
    adj_col: str = 'AdjIntensity',
    protein_col: str = 'Protein',
    cond_col: str = 'Condition',
    out_col: str = 'RobIntensity',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Optimized version of improved_robust_intensity that precomputes
    per-group statistics and merges them back to the frame. This avoids
    repeated groupby.transform calls and can be noticeably faster on
    larger data frames while preserving numeric output.

    The algorithm and outputs match `improved_robust_intensity` (within
    floating point tolerance).
    
    This implementation is intended to build robust intensity values used
    downstream for correlation and clustering analyses.
    """
    indent = "  "
    if verbose:
        print("\nComputing improved robust intensity (fast)...")

    # Basic input checks
    required_cols = {log_col, adj_col, protein_col, cond_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"improved_robust_intensity_fast: missing required columns: {missing}")

    df_copy = df.copy()
    if verbose:
        print(f"{indent}- Input shape: {df_copy.shape}; computing aggregated stats...")

    # --- Per-protein stats for z-scoring ---
    prot_stats = df_copy.groupby(protein_col)[[log_col, adj_col]].agg(['mean', 'std'])
    # flatten columns
    prot_stats.columns = [f"{c[0]}_{c[1]}" for c in prot_stats.columns]
    # Map back to rows
    df_copy = df_copy.join(prot_stats, on=protein_col)

    # compute z-scores using mapped stats (preserve nan if std is zero)
    df_copy[f'{log_col}_z'] = (df_copy[log_col] - df_copy[f'{log_col}_mean']) / df_copy[f'{log_col}_std']
    df_copy[f'{adj_col}_z'] = (df_copy[adj_col] - df_copy[f'{adj_col}_mean']) / df_copy[f'{adj_col}_std']

    # --- Per-(protein,condition) stats for stability metrics ---
    cond_stats = (
        df_copy.groupby([protein_col, cond_col])[[log_col, adj_col]]
        .agg(['mean', 'std'])
        .reset_index()
    )
    cond_stats.columns = [
        protein_col, cond_col,
        f'{log_col}_mean_cond', f'{log_col}_std_cond',
        f'{adj_col}_mean_cond', f'{adj_col}_std_cond'
    ]

    # Merge cond_stats into df_copy
    df_copy = df_copy.merge(cond_stats, on=[protein_col, cond_col], how='left')

    # Compute stability metrics
    log_stability = df_copy[f'{log_col}_std_cond']
    with np.errstate(divide='ignore', invalid='ignore'):
        adj_stability = (df_copy[f'{adj_col}_std_cond'] / df_copy[f'{adj_col}_mean_cond'].abs()).replace([np.inf, -np.inf], np.nan)

    log_stability = log_stability.fillna(1.0)
    adj_stability = adj_stability.fillna(1.0)

    # Weights
    log_weight = 1 / (1 + log_stability)
    adj_weight = 1 / (1 + adj_stability)
    total_weight = log_weight + adj_weight
    safe_total_weight = total_weight.replace(0, 1)
    log_weight_norm = log_weight / safe_total_weight
    adj_weight_norm = adj_weight / safe_total_weight

    # Combine z-scores
    robust_z = (log_weight_norm * df_copy[f'{log_col}_z'].fillna(0) +
                adj_weight_norm * df_copy[f'{adj_col}_z'].fillna(0))

    # Rescale to log intensity scale
    log_total_mean = df_copy[log_col].mean()
    log_total_std = df_copy[log_col].std()
    robust_intensity = robust_z * log_total_std + log_total_mean

    if verbose:
        print(f"{indent}- Finished fast computation. Appending column '{out_col}'")

    return df.assign(**{out_col: robust_intensity})

