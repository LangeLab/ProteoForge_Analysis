#!/usr/bin/env python3
"""
ProteoForge Imputation Module


Version: 0.7.0
Date: 2025-07-01
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

from typing import Union
from typing import Any, Optional
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer

# ======================================================================================
# Global Variables
# ======================================================================================


# ======================================================================================
# Helper Functions
# ======================================================================================

def generate_imputation_weights(
        data: pd.DataFrame,
        cond_dict: Optional[Dict[str, List[str]]] = None,
        default_weight: float = 1.0,
        missing_weight: float = 1e-5,
        absence_weight: float = 0.5,
        absence_threshold: Union[float, int] = 1.0,
        verbose: bool = True,
    ) -> pd.DataFrame:
    """
    Generates a feature-rich weight matrix for imputation.

    This function creates a weight matrix of the same dimensions as the input
    DataFrame. The weights are assigned based on a clear hierarchy:
    1.  Quantified (non-missing) values receive the `default_weight`.
    2.  Missing (NaN) values receive the `missing_weight`.
    3.  If `cond_dict` is provided, values can be assigned an `absence_weight`.
        This occurs for a feature within a specific condition if the number of
        missing values meets the `absence_threshold`. This weight will
        override both default and missing weights for all samples in that
        condition for that specific feature.

    Parameters
    ----------
    data : pd.DataFrame
        The input data, with samples in columns and features in rows.
        Missing values should be represented as `np.nan`.
    cond_dict : Optional[Dict[str, List[str]]], default=None
        A dictionary mapping condition names to lists of their corresponding
        sample (column) names. If None, the function only distinguishes
        between quantified and missing values.
        Example: {'Control': ['C1', 'C2'], 'Treatment': ['T1', 'T2', 'T3']}
    default_weight : float, default=1.0
        The weight assigned to all quantified (non-NaN) values.
    missing_weight : float, default=1e-5
        The base weight assigned to all missing (NaN) values.
    absence_weight : float, default=0.5
        The weight assigned to values (both quantified and missing) for a
        feature within a condition block if it's deemed "absent." This is
        only used if `cond_dict` is provided.
    absence_threshold : Union[float, int], default=1.0
        The threshold to determine "absence" for a feature within a condition.
        - If float (e.g., 1.0): A ratio of replicates. A feature is "absent"
          if the proportion of missing values is >= this ratio. A value of
          1.0 means all replicates in the condition must be missing.
        - If int (e.g., 3): An absolute count. A feature is "absent" if
          the number of missing values is >= this count.
    verbose : bool, default=True
        If True, prints informative messages during execution, such as the
        calculated absence count for each condition.

    Returns
    -------
    pd.DataFrame
        A DataFrame of the same shape as `data` containing the calculated
        weights for imputation.

    Raises
    ------
    ValueError
        If `absence_threshold` is an invalid value (e.g., a float > 1.0 or an
        integer greater than the number of replicates in a condition).
    TypeError
        If `absence_threshold` is not a float or an int.
    """
    # 1. Initialize the weight matrix with the default weight for all values.
    weights = pd.DataFrame(default_weight, index=data.index, columns=data.columns)

    # 2. Apply the base weight for all missing values.
    missing_mask = data.isna()
    weights[missing_mask] = missing_weight

    # 3. If no condition dictionary is provided, return the basic weight matrix.
    if cond_dict is None:
        return weights

    if verbose:
        print("Calculating conditional absence weights...")

    # 4. Iterate through each condition to apply conditional absence weights.
    for condition, samples in cond_dict.items():
        # Filter for samples that actually exist in the dataframe's columns
        valid_samples = [s for s in samples if s in data.columns]
        if not valid_samples:
            continue
        
        n_reps = len(valid_samples)

        # Calculate the numeric threshold count for absence for this specific condition
        if isinstance(absence_threshold, float):
            if not (0.0 < absence_threshold <= 1.0):
                raise ValueError("If a float, `absence_threshold` must be between 0.0 and 1.0.")
            threshold_count = int(np.ceil(n_reps * absence_threshold))
        elif isinstance(absence_threshold, int):
            if not (0 < absence_threshold <= n_reps):
                raise ValueError(
                    f"If an int, `absence_threshold` ({absence_threshold}) must be "
                    f"greater than 0 and no more than the number of replicates "
                    f"in the condition '{condition}' ({n_reps})."
                )
            threshold_count = absence_threshold
        else:
            raise TypeError("`absence_threshold` must be a float or an int.")

        if verbose:
            print(
                f"  Condition '{condition}' ({n_reps} replicates): "
                f"Absence defined as >= {threshold_count} missing values."
            )

        # Identify features (rows) that meet the absence criteria for this condition
        condition_missing_sum = missing_mask[valid_samples].sum(axis=1)
        absent_features_mask = condition_missing_sum >= threshold_count

        # Apply absence weight, overriding previous weights for these features
        # within this condition's samples.
        if absent_features_mask.any():
            weights.loc[absent_features_mask, valid_samples] = absence_weight

    return weights

# --- --- --- Helper Functions for Printing and Imputation Steps --- --- ---

def _print_step_header(title: str, params: Dict[str, Any]):
    """Prints a standardized header for each pipeline step."""
    print(f"\n--- Running Step: {title} ---")
    if params:
        param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in params.items()])
        print(f"  Parameters: {param_str}")

def _find_downshift_params(
    data: pd.DataFrame, shift_magnitude: float, low_percentile: float
) -> Tuple[float, float]:
    """Internal helper to find the downshifted mean and low-value threshold."""
    quantified_values = data.values.flatten()
    quantified_values = quantified_values[~np.isnan(quantified_values)]
    if quantified_values.size == 0:
        return 0, 0
    low_value_threshold = np.percentile(quantified_values, low_percentile * 100)
    low_value_distribution = quantified_values[quantified_values < low_value_threshold]
    if low_value_distribution.size == 0:
        downshifted_mean = low_value_threshold - shift_magnitude
    else:
        downshifted_mean = low_value_distribution.mean() - shift_magnitude
    return downshifted_mean, low_value_threshold


def amputate_sparse_features(
    data: pd.DataFrame,
    cond_dict: Dict[str, List[str]],
    min_quantified: Union[int, float] = 1,
    verbose: bool = True
) -> pd.DataFrame:
    """Amputates (sets to NaN) features that are too sparsely quantified within a condition."""
    amputated_data = data.copy()
    if verbose:
        _print_step_header("Sparse Feature Amputation", {'min_quantified': min_quantified})

    total_amputated = 0
    for condition, samples in cond_dict.items():
        valid_samples = [s for s in samples if s in amputated_data.columns]
        if not valid_samples: continue
        n_reps = len(valid_samples)
        if isinstance(min_quantified, float):
            threshold_count = int(np.ceil(n_reps * min_quantified))
        else:
            threshold_count = min_quantified
        
        quantified_counts = amputated_data[valid_samples].notna().sum(axis=1)
        rows_to_amputate_mask = (quantified_counts < threshold_count) & (quantified_counts > 0)
        num_rows_amputated = rows_to_amputate_mask.sum()

        if num_rows_amputated > 0:
            if verbose:
                print(f"  Condition '{condition}': Amputating {num_rows_amputated} features with < {threshold_count} quantified values.")
            # Count how many NaNs are newly created
            total_amputated += amputated_data.loc[rows_to_amputate_mask, valid_samples].notna().sum().sum()
            amputated_data.loc[rows_to_amputate_mask, valid_samples] = np.nan
    
    if verbose:
        print(f"  Summary: Created {total_amputated} new missing values by amputation.")
    return amputated_data


def fill_dense_features(
    data: pd.DataFrame,
    cond_dict: Dict[str, List[str]],
    max_missing: Union[int, float] = 1,
    strategy: str = 'mean',
    verbose: bool = True
) -> pd.DataFrame:
    """Fills sporadic missing values in densely quantified features within a condition."""
    filled_data = data.copy()
    if strategy not in ['mean', 'median']: raise ValueError("Strategy must be 'mean' or 'median'.")
    if verbose:
        _print_step_header("Dense Feature Filling", {'max_missing': max_missing, 'strategy': strategy})

    total_filled = 0
    for condition, samples in cond_dict.items():
        valid_samples = [s for s in samples if s in filled_data.columns]
        if not valid_samples: continue
        n_reps = len(valid_samples)
        if isinstance(max_missing, float):
            threshold_count = int(np.floor(n_reps * max_missing))
        else:
            threshold_count = max_missing

        missing_counts = filled_data[valid_samples].isna().sum(axis=1)
        rows_to_fill_mask = (missing_counts > 0) & (missing_counts <= threshold_count)
        num_rows_to_fill = rows_to_fill_mask.sum()

        if num_rows_to_fill > 0:
            if verbose:
                print(f"  Condition '{condition}': Filling {num_rows_to_fill} features with 1 to {threshold_count} missing values.")
            
            target_rows = filled_data.loc[rows_to_fill_mask, valid_samples]
            total_filled += target_rows.isna().sum().sum()
            fill_values = target_rows.mean(axis=1) if strategy == 'mean' else target_rows.median(axis=1)
            filled_data.loc[rows_to_fill_mask, valid_samples] = target_rows.T.fillna(fill_values).T
    
    if verbose:
        print(f"  Summary: Filled {total_filled} missing values.")
    return filled_data


def downshifted_imputation(
    data: pd.DataFrame,
    cond_dict: Dict[str, List[str]],
    missingness_threshold: float = 1.0,
    shift_magnitude: float = 1.5,
    low_percentile: float = 0.10,
    verbose: bool = True
) -> pd.DataFrame:
    """Imputes missing values using a downshifted low-value distribution."""
    imputed_data = data.copy()
    params = {
        'missingness_threshold': missingness_threshold,
        'shift_magnitude': shift_magnitude,
        'low_percentile': low_percentile
    }
    if verbose:
        _print_step_header("Downshifted Imputation", params)

    total_imputed = 0
    for condition, samples in cond_dict.items():
        valid_samples = [s for s in samples if s in imputed_data.columns]
        if not valid_samples: continue
        n_reps = len(valid_samples)
        threshold_count = int(np.ceil(n_reps * missingness_threshold))
        
        missing_counts = imputed_data[valid_samples].isna().sum(axis=1)
        rows_to_impute_mask = missing_counts >= threshold_count
        num_rows_to_impute = rows_to_impute_mask.sum()

        if num_rows_to_impute == 0: continue
        if verbose:
            print(f"  Condition '{condition}': Imputing {num_rows_to_impute} features with >= {threshold_count} missing values.")
        
        mean_val, low_val = _find_downshift_params(imputed_data[valid_samples], shift_magnitude, low_percentile)
        all_sds = imputed_data[valid_samples].std(axis=1, skipna=True).dropna()
        if all_sds.empty: continue
        random_sds = np.random.choice(all_sds, size=num_rows_to_impute, replace=True)
        
        target_rows = imputed_data.loc[rows_to_impute_mask, valid_samples]
        total_imputed += target_rows.isna().sum().sum()
        imputed_block = target_rows.copy()
        for i, (idx, row) in enumerate(target_rows.iterrows()):
            missing_in_row = row.isna()
            n_missing = missing_in_row.sum()
            if n_missing > 0:
                    imputed_vals = np.random.normal(mean_val, random_sds[i], n_missing)
                    imputed_vals = np.clip(imputed_vals, a_min=None, a_max=low_val)
                    # Explicitly cast to the dtype of the block to avoid FutureWarning
                    dtype = imputed_block.loc[idx, missing_in_row].dtype
                    imputed_block.loc[idx, missing_in_row] = imputed_vals.astype(dtype)
        imputed_data.loc[rows_to_impute_mask, valid_samples] = imputed_block

    if verbose:
        print(f"  Summary: Imputed {total_imputed} missing values.")
    return imputed_data


def knn_imputation(
    data: pd.DataFrame,
    cond_dict: Dict[str, List[str]],
    n_neighbors: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Imputes remaining missing values using K-Nearest Neighbors within conditions."""
    imputed_data = data.copy()
    if verbose:
        _print_step_header("k-NN Imputation", {'n_neighbors': n_neighbors})

    total_imputed = 0
    for condition, samples in cond_dict.items():
        valid_samples = [s for s in samples if s in imputed_data.columns]
        if not valid_samples: continue
        
        condition_data = imputed_data[valid_samples]
        missing_count = condition_data.isna().sum().sum()
        if missing_count == 0: continue

        if verbose:
            print(f"  Condition '{condition}': Applying k-NN to {missing_count} remaining missing values.")
        total_imputed += missing_count
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data[valid_samples] = imputer.fit_transform(condition_data)
    
    if verbose:
        print(f"  Summary: Imputed {total_imputed} missing values.")
    return imputed_data


# ======================================================================================
# Main Pipeline Function
# ======================================================================================

def run_imputation_pipeline(
    data: pd.DataFrame,
    cond_dict: Dict[str, List[str]],
    scheme: List[Dict[str, Any]],
    is_log2: bool = False,
    return_log2: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Runs a configurable, multi-step imputation pipeline.
    """
    if is_log2:
        imputed_data = data.copy()
    else:
        imputed_data = np.log2(data).copy()
    
    method_map = {
        'amputate': amputate_sparse_features,
        'fill_dense': fill_dense_features,
        'downshift': downshifted_imputation,
        'knn': knn_imputation
    }

    if verbose:
        print("====== Starting Imputation Pipeline ======")
        print(f"Initial missing values: {imputed_data.isna().sum().sum()}")

    for i, step in enumerate(scheme):
        method_name = step.get('method')
        params = step.get('params', {})
        
        if method_name not in method_map:
            raise ValueError(f"Unknown method '{method_name}' in scheme.")
        
        imputation_func = method_map[method_name]
        
        # We pass verbose=False to sub-functions because the pipeline controls printing
        imputed_data = imputation_func(
            data=imputed_data,
            cond_dict=cond_dict,
            verbose=verbose,
            **params
        )
        if verbose:
            print(f"Total missing values after step {i+1} ('{method_name}'): {imputed_data.isna().sum().sum()}")

    if verbose:
        print("\n====== Imputation Pipeline Finished ======")
    if return_log2:
        return imputed_data
    else:
        return np.exp2(imputed_data)