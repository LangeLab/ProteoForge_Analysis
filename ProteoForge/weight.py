#!/usr/bin/env python3
"""
Weight Generation Module

Containing modular weight component functions as well as generating optimum weights 
using top weights calculated and opimized by using PLS (Partial Least Squares).
Provides a suite of vectorized, high-performance weighting strategies and can return  
a single array of weights that can be used in ProteoForge's discordant peptide 
identification algorithm when wls or glm models are selected.

Version: 0.7.0
Date: 2025-07-01
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

import time
import numpy as np
import pandas as pd
from typing import Optional, Sequence, List
from sklearn.cross_decomposition import PLSRegression

# ======================================================================================
# Utility Functions
# ======================================================================================

def pretty_time(t: float) -> str:
    """
    Convert a time duration in seconds to a human-readable string.
    """
    if t < 1e-6:
        return f"{t*1e9:.2f} ns"
    elif t < 1e-3:
        return f"{t*1e6:.2f} µs"
    elif t < 1:
        return f"{t*1e3:.2f} ms"
    elif t < 60:
        return f"{t:.2f} s"
    elif t < 3600:
        return f"{t/60:.2f} min"
    else:
        return f"{t/3600:.2f} h"

def min_max_scale(
    x: pd.Series | np.ndarray,
    epsilon: float = 1e-9
) -> np.ndarray:
    """
    Scale input values to the [0, 1] range using min-max normalization.

    Parameters
    ----------
    x : pd.Series or np.ndarray
        Input array or Series to scale.
    epsilon : float, optional
        Small value to avoid division by zero, by default 1e-9.

    Returns
    -------
    np.ndarray
        Scaled values in the range [0, 1]. If all values are identical, 
        returns an array of ones.

    Raises
    ------
    ValueError
        If input is not a 1-dimensional array or Series.

    Notes
    -----
    - NaN values are preserved in the output.
    - If all values are NaN, returns an array of NaNs.
    - If input is not a pandas Series or numpy array, raises ValueError.
    """
    if not isinstance(x, (pd.Series, np.ndarray)):
        raise ValueError("Input must be a pandas Series or numpy ndarray.")
    x = pd.Series(x) if not isinstance(x, pd.Series) else x
    if x.ndim != 1:
        raise ValueError("Input must be 1-dimensional.")
    if x.isna().all():
        return np.full_like(x, np.nan, dtype=float)
    min_val = x.min(skipna=True)
    max_val = x.max(skipna=True)
    denominator = max_val - min_val
    if denominator == 0:
        return np.ones_like(x, dtype=float)
    res = ((x - min_val) / denominator).to_numpy(dtype=float)
    res = np.clip(res, (0.0 + epsilon), 1.0)
    return res


def align_weights_to_dataframe(
    df: pd.DataFrame,
    weight_df: pd.DataFrame,
    on_cols: Sequence[str],
    weight_col_name: str
) -> np.ndarray:
    """
    Merge calculated weights back to the original dataframe, ensuring alignment and shape.

    Parameters
    ----------
    df : pd.DataFrame
        The original dataframe to which weights should be aligned.
    weight_df : pd.DataFrame
        DataFrame containing calculated weights and the columns to join on.
    on_cols : Sequence[str]
        Column names to merge on (e.g., ['protein_id', 'peptide_id']).
    weight_col_name : str
        Name of the column in `weight_df` containing the weights.

    Returns
    -------
    np.ndarray
        Numpy array of weights, aligned with the order and length of `df`.

    Raises
    ------
    ValueError
        If required columns are missing or if the merge results in a shape mismatch.

    Notes
    -----
    - Any missing weights after merging are filled with 0.0.
    - Ensures the returned array matches the original dataframe's order and length.
    """
    missing_df_cols = set(on_cols) - set(df.columns)
    missing_weight_cols = set(on_cols) - set(weight_df.columns)
    if missing_df_cols:
        raise ValueError(f"Missing columns in `df`: {sorted(missing_df_cols)}")
    if missing_weight_cols:
        raise ValueError(f"Missing columns in `weight_df`: {sorted(missing_weight_cols)}")
    if weight_col_name not in weight_df.columns:
        raise ValueError(f"Column '{weight_col_name}' not found in `weight_df`.")

    df_with_index = df[on_cols].copy()
    df_with_index['original_index'] = df.index
    merged_df = pd.merge(df_with_index, weight_df, on=on_cols, how='left', sort=False)
    merged_df = merged_df.sort_values('original_index').set_index('original_index')
    if len(merged_df) != len(df):
        raise ValueError(
            "Merged dataframe does not match the original length." \
            " Check for duplicate keys in `weight_df`."
        )
    return merged_df[weight_col_name].reindex(df.index).fillna(0.0).to_numpy(dtype=float)

# ======================================================================================
# Modular Weight Component Functions
# ======================================================================================

def calculate_imputation_weights(
    df: pd.DataFrame,
    is_real_col: Optional[str] = None,
    is_comp_miss_col: Optional[str] = None,
    true_val: float = 1.0,
    sparse_imputed_val: float = 1e-5,
    dense_imputed_val: float = 0.5,
    verbose: bool = False
) -> np.ndarray:
    """
    Assign weights to data points based on imputation status.

    If both `is_real_col` and `is_comp_miss_col` are None, assumes all values are real 
        and returns an array of ones.
    Otherwise:
      - Real (not imputed) values (where `is_real_col` is True) get `true_val`.
      - Imputed values that are part of a completely missing peptide for a condition 
            (dense imputation, as indicated by `is_comp_miss_col`) get `dense_imputed_val`.
      - Imputed values that are not part of a completely missing peptide 
            (sparse imputation) get `sparse_imputed_val`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    is_real_col : Optional[str]
        Column name indicating if the value is real (not imputed).
    is_comp_miss_col : Optional[str]
        Column name indicating if the peptide is completely missing for a condition.
    true_val : float
        Weight for real (not imputed) values.
    sparse_imputed_val : float
        Weight for sparsely imputed values.
    dense_imputed_val : float
        Weight for densely imputed values.
    verbose : bool
        Whether to print a message if imputation columns are not provided.

    Returns
    -------
    np.ndarray
        Array of weights aligned with the input dataframe.

    Raises
    ------
    ValueError
        If specified columns are not found in the dataframe or if weight values are invalid.
    """
    if is_real_col is None or is_comp_miss_col is None:
        if verbose:
            print(
                "\n    Imputation columns not provided. Assuming all data points are real values.\n"
                "    If this is a mistake, please provide the column names for 'is_real_col' and 'is_comp_miss_col'."
            )
        return np.ones(len(df), dtype=float)
    missing_cols = [col for col in [is_real_col, is_comp_miss_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"    Required column(s) {missing_cols} not found in the dataframe. Please check your column names."
        )
    true_val = float(true_val)
    sparse_imputed_val = float(sparse_imputed_val)
    dense_imputed_val = float(dense_imputed_val)
    if not (0 <= true_val <= 1 and 0 <= sparse_imputed_val <= 1 and 0 <= dense_imputed_val <= 1):
        raise ValueError("    Impute Weight Values must be between 0 and 1.")
    if true_val <= sparse_imputed_val or true_val <= dense_imputed_val:
        raise ValueError(
            "    True value must be greater than both sparse and dense imputed values." \
            "    Please adjust the values accordingly."
        )
    is_real = df[is_real_col].astype(bool)
    is_comp_miss = df[is_comp_miss_col].astype(bool)
    weights = np.where(
        is_real,
        true_val,
        np.where(
            is_comp_miss,
            dense_imputed_val,
            sparse_imputed_val
        )
    )
    return weights


def calculate_inverse_metric_weights(
    df: pd.DataFrame,
    metric: str = 'var',
    intensity_col: str = 'ms1adj',
    group_cols: Optional[list[str]] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Calculate weights as the inverse of a group-wise summary metric 
        (e.g., variance, std, mean) for replicate groups.
    Lower metric values (e.g., lower variance) yield higher weights. 
        The result is min-max scaled to [0, 1].

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    metric : str
        Group metric to use for weighting. 
        Supported: 'var', 'std', 'mean', 'median', 'min', 'max', 'sum'.
    intensity_col : str
        Column with intensity values.
    group_cols : list[str], optional
        Columns to group by. Defaults to ['protein_id', 'peptide_id', 'day'].

    Returns
    -------
    np.ndarray
        Min-max scaled array of inverse-metric weights, aligned with input df.

    Raises
    ------
    ValueError
        If required columns are missing or metric is unsupported.
    """
    if group_cols is None:
        group_cols = ['protein_id', 'peptide_id', 'day']
    missing_cols = [col for col in group_cols + [intensity_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")
    supported_metrics = {'var', 'std', 'mean', 'median', 'min', 'max', 'sum'}
    if metric not in supported_metrics:
        raise ValueError(
            f"Unsupported metric '{metric}'. Supported metrics: {sorted(supported_metrics)}"
        )
    group_metric = df.groupby(group_cols, observed=True)[intensity_col].transform(metric)
    weights = 1.0 / group_metric
    weights = pd.Series(weights).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return min_max_scale(weights)


def calculate_correlation_discordance_weights(
    df: pd.DataFrame,
    control_condition: str,
    intensity_col: str = 'ms1adj',
    condition_col: str = 'day',
    protein_col: str = 'protein_id',
    peptide_col: str = 'peptide_id',
    verbose: bool = False
) -> np.ndarray:
    """
    Compute discordance weights based on the correlation of a peptide's 
        fold-change profile with the mean profile of its parent protein.
    A low or negative correlation indicates high discordance.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    control_condition : str
        The label of the control condition.
    intensity_col : str
        Column with intensity values.
    condition_col : str
        Column with condition labels.
    protein_col : str
        Column with protein identifiers.
    peptide_col : str
        Column with peptide identifiers.

    Returns
    -------
    np.ndarray
        Min-max scaled array of discordance weights, aligned with input df.

    Raises
    ------
    ValueError
        If required columns are missing or control condition is not found.
    """
    required_cols = {protein_col, peptide_col, condition_col, intensity_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required column(s): {sorted(missing_cols)}")
    id_cols = [protein_col, peptide_col]
    fc_matrix = df.groupby(
        id_cols + [condition_col], observed=True
    )[intensity_col].median().unstack()
    if control_condition not in fc_matrix.columns:
        raise ValueError(f"Control condition '{control_condition}' not found in the data.")
    exp_conditions = [c for c in fc_matrix.columns if c != control_condition]
    if not exp_conditions:
        return np.zeros(len(df), dtype=float)
    fc_matrix = fc_matrix.div(fc_matrix[control_condition], axis=0)
    fc_matrix_exp = fc_matrix[exp_conditions].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    protein_means = fc_matrix_exp.groupby(protein_col, observed=True).transform('mean')
    correlations = fc_matrix_exp.corrwith(protein_means, axis=1)
    discordance_score = 1 - correlations.fillna(1.0)
    fc_matrix['correlation_discordance_weight'] = min_max_scale(discordance_score.fillna(0.0))
    return align_weights_to_dataframe(
        df,
        fc_matrix[['correlation_discordance_weight']].reset_index(),
        id_cols,
        'correlation_discordance_weight'
    )


def calculate_relative_variability_weights(
    df: pd.DataFrame,
    intensity_col: str = 'ms1adj',
    condition_col: str = 'day',
    protein_col: str = 'protein_id',
    peptide_col: str = 'peptide_id',
    verbose: bool = False
) -> np.ndarray:
    """
    Compute weights based on a peptide's replicate variability (CV) relative to the 
        median variability of its parent protein's other peptides.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    intensity_col : str
        Column with intensity values.
    condition_col : str
        Column with condition labels.
    protein_col : str
        Column with protein identifiers.
    peptide_col : str
        Column with peptide identifiers.

    Returns
    -------
    np.ndarray
        Min-max scaled array of relative variability weights, aligned with input df.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    id_cols = [protein_col, peptide_col]
    group_cols = id_cols + [condition_col]
    required_cols = {protein_col, peptide_col, condition_col, intensity_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required column(s): {sorted(missing_cols)}")
    replicate_stats = df.groupby(
        group_cols, observed=True
    )[intensity_col].agg(['mean', 'std']).reset_index()
    replicate_stats['cv'] = (replicate_stats['std'] / replicate_stats['mean']).fillna(0.0)
    peptide_avg_cv = replicate_stats.groupby(
        id_cols, observed=True
    )['cv'].mean().to_frame(name='peptide_avg_cv')
    peptide_avg_cv = peptide_avg_cv.reset_index()
    protein_median_cv = peptide_avg_cv.groupby(
        protein_col, observed=True
    )['peptide_avg_cv'].transform('median')
    peptide_avg_cv['protein_median_cv'] = protein_median_cv
    peptide_avg_cv['relative_variability'] = (
        peptide_avg_cv['peptide_avg_cv'] / (peptide_avg_cv['protein_median_cv'] + 1e-9)
    )
    peptide_avg_cv['relative_variability_weight'] = min_max_scale(
        peptide_avg_cv['relative_variability'].fillna(0.0)
    )
    return align_weights_to_dataframe(
        df,
        peptide_avg_cv,
        id_cols,
        'relative_variability_weight'
    )


def calculate_generalized_signal_to_noise_weights(
    df: pd.DataFrame,
    intensity_col: str = 'ms1adj',
    condition_col: str = 'day',
    protein_col: str = 'protein_id',
    peptide_col: str = 'peptide_id',
    verbose: bool = False
) -> np.ndarray:
    """
    Calculates weights based on a generalized signal-to-noise ratio using a
    fully vectorized approach for maximum performance.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    intensity_col : str
        Column with intensity values.
    condition_col : str
        Column with condition labels.
    protein_col : str
        Column with protein identifiers.
    peptide_col : str
        Column with peptide identifiers.

    Returns
    -------
    np.ndarray
        Min-max scaled array of signal-to-noise weights.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    id_cols = [protein_col, peptide_col]
    required_cols = id_cols + [condition_col, intensity_col]
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"Missing one or more required columns: {required_cols}")

    # Use a minimal temporary DataFrame to avoid altering the original
    df_temp = df[required_cols].copy()

    # == Vectorized Calculation ==
    
    # 1. Calculate peptide-level (global) and condition-level stats using transform.
    # `transform` is highly efficient as it performs the aggregation and then
    # broadcasts the result back to match the original DataFrame's index.
    peptide_groups = df_temp.groupby(id_cols, observed=True)[intensity_col]
    condition_groups = df_temp.groupby(id_cols + [condition_col], observed=True)[intensity_col]

    df_temp['global_mean'] = peptide_groups.transform('mean')
    df_temp['condition_mean'] = condition_groups.transform('mean')
    df_temp['condition_std'] = condition_groups.transform('std').fillna(0)

    # 2. Calculate signal-to-noise for every row in a single, vectorized step.
    signal = np.abs(df_temp['condition_mean'] - df_temp['global_mean'])
    noise = df_temp['condition_std']
    
    # Use np.divide for safe division, yielding 0 where noise is 0.
    s_to_n_scores = np.divide(
        signal, noise, out=np.zeros_like(signal, dtype=float), where=(noise != 0)
    )
    df_temp['s_to_n_score'] = s_to_n_scores
    
    # 3. Find the maximum S-to-N score for each peptide.
    # This is much faster than the original apply loop.
    peptide_weights = df_temp.groupby(
        id_cols, observed=True
    )['s_to_n_score'].max().reset_index(name='s_to_n_weight')
    
    # Min-max scale and align
    peptide_weights['s_to_n_weight'] = min_max_scale(peptide_weights['s_to_n_weight'].fillna(0.0))

    # 5. Align weights back to the original DataFrame structure.
    # This typically involves a merge operation.
    return align_weights_to_dataframe(df, peptide_weights, id_cols, 's_to_n_weight')


def calculate_replicate_concordance_weights(
    df: pd.DataFrame,
    intensity_col: str = 'ms1adj',
    condition_col: str = 'day',
    protein_col: str = 'protein_id',
    peptide_col: str = 'peptide_id',
    verbose: bool = False
) -> np.ndarray:
    """
    Calculates weights based on replicate concordance using a fully vectorized
    approach for maximum performance.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    intensity_col : str
        Column with intensity values.
    condition_col : str
        Column with condition labels.
    protein_col : str
        Column with protein identifiers.
    peptide_col : str
        Column with peptide identifiers.

    Returns
    -------
    np.ndarray
        Min-max scaled array of replicate concordance weights.
    """
    id_cols = [protein_col, peptide_col]
    required_cols = id_cols + [condition_col, intensity_col]
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"Missing one or more required columns: {required_cols}")

    # Use a minimal temporary DataFrame to avoid altering the original
    df_temp = df[required_cols].copy()

    # == Vectorized Calculation ==

    # 1. Calculate peptide-level (overall) stats using transform.
    # This broadcasts the total sum and count for a peptide to every row
    # belonging to that peptide.
    peptide_groups = df_temp.groupby(id_cols, observed=True)[intensity_col]
    df_temp['peptide_total_sum'] = peptide_groups.transform('sum')
    df_temp['peptide_total_count'] = peptide_groups.transform('count')

    # 2. Calculate condition-level stats using transform.
    # This gets the sum and count for each specific peptide-condition group.
    condition_groups = df_temp.groupby(id_cols + [condition_col], observed=True)[intensity_col]
    df_temp['condition_sum'] = condition_groups.transform('sum')
    df_temp['condition_replicates'] = condition_groups.transform('count')

    # 3. Calculate the "leave-one-condition-out" baseline mean for every row.
    # This is the most critical vectorized step. For each row, the baseline is
    # the mean of all *other* conditions within the same peptide.
    sum_of_other_conditions = df_temp['peptide_total_sum'] - df_temp['condition_sum']
    count_of_other_conditions = df_temp['peptide_total_count'] - df_temp['condition_replicates']

    # Use np.divide for safe division, yielding 0 where the denominator is 0.
    # This handles peptides that appear in only one condition.
    df_temp['baseline_mean'] = np.divide(
        sum_of_other_conditions,
        count_of_other_conditions,
        out=np.zeros_like(sum_of_other_conditions, dtype=float),
        where=(count_of_other_conditions != 0)
    )

    # 4. Count replicates above/below the baseline for each condition.
    # We create boolean flags and then use transform('sum') to count them per group.
    df_temp['is_up'] = (df_temp[intensity_col] > df_temp['baseline_mean']).astype(int)
    df_temp['is_down'] = (df_temp[intensity_col] < df_temp['baseline_mean']).astype(int)

    n_up = df_temp.groupby(id_cols + [condition_col], observed=True)['is_up'].transform('sum')
    n_down = df_temp.groupby(id_cols + [condition_col], observed=True)['is_down'].transform('sum')

    # 5. Calculate the concordance score for each condition.
    # The score is the fraction of replicates that agree on the direction of change.
    max_agreement = np.maximum(n_up, n_down)
    
    # Safe division for the final score calculation
    df_temp['concordance_score'] = np.divide(
        max_agreement,
        df_temp['condition_replicates'],
        out=np.zeros_like(max_agreement, dtype=float),
        where=(df_temp['condition_replicates'] != 0)
    )
    
    # 6. Find the maximum concordance score for each peptide.
    # This replaces the inner loop of the original function.
    peptide_weights = df_temp.groupby(
        id_cols, observed=True
    )['concordance_score'].max().reset_index(name='concordance_weight')

    # The score is already in [0, 1], but min-max scaling ensures consistency
    peptide_weights['concordance_weight'] = min_max_scale(
        peptide_weights['concordance_weight'].fillna(0.0)
    )

    # 8. Align weights back to the original DataFrame structure.
    return align_weights_to_dataframe(df, peptide_weights, id_cols, 'concordance_weight')

def calculate_normalization_impact_weights(
    df: pd.DataFrame,
    log_intensity_col: str = 'log_intensity',
    adj_intensity_col: str = 'adj_intensity',
    protein_col: str = 'protein_id',
    peptide_col: str = 'peptide_id',
    verbose: bool = False
) -> np.ndarray:
    """
    Calculates weights by penalizing peptides whose intensity values are heavily
    modified by the normalization process.

    It computes the average absolute difference between two intensity types
    (e.g., log10 and adjusted) for each peptide. A smaller difference suggests
    higher data quality and receives a higher weight.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    log_intensity_col : str
        Column with log-transformed intensity.
    adj_intensity_col : str
        Column with normalized/adjusted intensity.
    protein_col : str
        Column with protein identifiers.
    peptide_col : str
        Column with peptide identifiers.

    Returns
    -------
    np.ndarray
        Min-max scaled array of normalization impact weights.
    """
    id_cols = [protein_col, peptide_col]
    required_cols = id_cols + [log_intensity_col, adj_intensity_col]
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"Missing one or more required columns: {required_cols}")

    # Calculate the absolute difference for each row
    temp_df = df[required_cols].copy()
    temp_df['diff'] = (temp_df[adj_intensity_col] - temp_df[log_intensity_col]).abs()

    # Calculate the mean difference per peptide
    peptide_impact = temp_df.groupby(
        id_cols, observed=True
    )['diff'].mean().reset_index(name='impact_score')

    # The weight is INVERSE to the impact score. High impact -> low weight.
    # Add a small epsilon to avoid division by zero if impact is 0
    peptide_impact['impact_weight'] = 1 / (1 + peptide_impact['impact_score'])
    
    # Min-max scale and align
    peptide_impact['impact_weight'] = min_max_scale(peptide_impact['impact_weight'].fillna(0.0))
    
    return align_weights_to_dataframe(df, peptide_impact, id_cols, 'impact_weight')

def calculate_directional_agreement_weights(
    df: pd.DataFrame,
    control_condition: str,
    intensity_col: str = 'ms1adj',
    condition_col: str = 'day',
    protein_col: str = 'protein_id',
    peptide_col: str = 'peptide_id',
    verbose: bool = False
) -> np.ndarray:
    """
    Calculates weights based on antagonistic discordance.

    This weight gives a high score to peptides whose fold-change direction
    (relative to control) is opposite to the mean fold-change direction of
    their parent protein in at least one condition. It specifically rewards
    strong, opposing signals, which are often of biological interest.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    control_condition : str
        The label of the control condition.
    intensity_col : str
        Column with intensity values to use for fold-change.
    condition_col : str
        Column with condition labels.
    protein_col : str
        Column with protein identifiers.
    peptide_col : str
        Column with peptide identifiers.

    Returns
    -------
    np.ndarray
        Min-max scaled array of directional agreement weights.
    """
    id_cols = [protein_col, peptide_col]
    required_cols = id_cols + [condition_col, intensity_col]
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"Missing one or more required columns: {required_cols}")

    # 1. Pivot to get median intensity per peptide and condition
    fc_matrix = df.groupby(
        id_cols + [condition_col], observed=True
    )[intensity_col].median().unstack()
    if control_condition not in fc_matrix.columns:
        raise ValueError(f"Control condition '{control_condition}' not found in data.")

    # 2. Calculate fold-change relative to control
    fc_matrix = fc_matrix.div(fc_matrix[control_condition], axis=0)
    exp_conditions = [c for c in fc_matrix.columns if c != control_condition]
    fc_matrix_exp = fc_matrix[exp_conditions].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # 3. Calculate mean protein profile
    protein_means = fc_matrix_exp.groupby(protein_col, observed=True).transform('mean')

    # 4. Get the sign of log2 fold-change for peptides and proteins
    peptide_log2fc_sign = np.sign(np.log2(fc_matrix_exp))
    protein_log2fc_sign = np.sign(np.log2(protein_means))

    # 5. Find disagreements (where signs are opposite)
    # A disagreement is where peptide_sign * protein_sign = -1
    disagreement_matrix = (peptide_log2fc_sign * protein_log2fc_sign) == -1
    
    # The weight is the proportion of conditions showing antagonistic behavior
    antagonistic_score = disagreement_matrix.sum(axis=1) / len(exp_conditions)

    # 6. Min-max scale and align back to original dataframe
    weight_df = antagonistic_score.reset_index(name='directional_weight')
    weight_df['directional_weight'] = min_max_scale(weight_df['directional_weight'].fillna(0.0))

    return align_weights_to_dataframe(df, weight_df, id_cols, 'directional_weight')

def calculate_replicate_imbalance_weights(
    df: pd.DataFrame,
    condition_col: str = 'day',
    protein_col: str = 'protein_id',
    peptide_col: str = 'peptide_id',
    is_real_col: Optional[str] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Calculates weights based on replicate imbalance.

    If 'is_real_col' is provided, this function up-weights data points
    belonging to smaller *real* (non-imputed) replicate groups. If not, it
    weights based on the total number of replicates per group.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    condition_col : str
        Column with condition labels.
    protein_col : str
        Column with protein identifiers.
    peptide_col : str
        Column with peptide identifiers.
    is_real_col : Optional[str], optional
        Column with boolean flags (True if real, False if imputed). Defaults to None.

    Returns
    -------
    np.ndarray
        Min-max scaled array of replicate imbalance weights.
    """
    id_cols = [protein_col, peptide_col, condition_col]
    required_cols = id_cols
    if is_real_col:
        required_cols.append(is_real_col)

    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"Missing one or more required columns: {required_cols}")

    # --- KEY CHANGE ---
    # Conditionally calculate counts based on whether is_real_col is provided.
    if is_real_col:
        # If is_real_col is specified, count only the TRUE values.
        if not pd.api.types.is_bool_dtype(df[is_real_col]):
            raise TypeError(f"Column '{is_real_col}' must be of boolean type.")
        # Summing a boolean column counts the number of True values.
        counts = df.groupby(id_cols, observed=True)[is_real_col].transform('sum')
    else:
        # If is_real_col is None, revert to counting all replicates in the group.
        counts = df.groupby(id_cols, observed=True)[id_cols[0]].transform('count')

    # The weight is the inverse of the count.
    # We add 1 to the denominator to handle groups with zero (real) replicates
    # gracefully and to ensure smaller counts get higher weights.
    imbalance_weights = 1.0 / (counts + 1.0)

    # The result is already aligned with the dataframe, just needs scaling.
    return min_max_scale(imbalance_weights.fillna(0.0))

def calculate_profile_correlation_weights(
    df: pd.DataFrame,
    log_intensity_col: str = 'log_intensity',
    adj_intensity_col: str = 'adj_intensity',
    condition_col: str = 'day',
    protein_col: str = 'protein_id',
    peptide_col: str = 'peptide_id',
    verbose: bool = False
) -> np.ndarray:
    """
    Calculates weights based on the correlation of intensity profiles.

    This is a high-speed alternative to the model-disagreement weight. It
    checks for signal robustness by calculating the Pearson correlation between a
    peptide's mean-intensity profiles across conditions, as derived from two
    different intensity columns (e.g., log vs. adjusted). A high positive
    correlation suggests a robust signal.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    log_intensity_col : str
        Column with the first intensity type.
    adj_intensity_col : str
        Column with the second intensity type.
    condition_col : str
        Column with condition labels.
    protein_col : str
        Column with protein identifiers.
    peptide_col : str
        Column with peptide identifiers.

    Returns
    -------
    np.ndarray
        Min-max scaled array of profile correlation weights.
    """
    id_cols = [protein_col, peptide_col]
    required_cols = id_cols + [condition_col, log_intensity_col, adj_intensity_col]
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"Missing one or more required columns: {required_cols}")

    # Calculate the mean intensity per condition for each peptide for both types
    log_profile = df.groupby(
        id_cols + [condition_col], observed=True
    )[log_intensity_col].mean().unstack()
    adj_profile = df.groupby(
        id_cols + [condition_col], observed=True
    )[adj_intensity_col].mean().unstack()

    # Align the two profile dataframes to ensure we have matching peptides and conditions
    log_profile, adj_profile = log_profile.align(adj_profile, join='inner', axis=1)
    log_profile, adj_profile = log_profile.align(adj_profile, join='inner', axis=0)
    
    # Calculate the row-wise (peptide-wise) correlation between the two profiles
    # The .corrwith() method is highly efficient for this.
    correlations = log_profile.corrwith(adj_profile, axis=1)

    # The correlation is between -1 and 1. We scale it to [0, 1] to use as a weight.
    # A correlation of 1 (perfect agreement) becomes weight 1.
    # A correlation of -1 (perfect disagreement) becomes weight 0.
    # NaN correlations (e.g., from peptides with no variance) are treated as 0 agreement.
    correlation_score = (correlations.fillna(0) + 1) / 2

    # Prepare the weights for alignment
    weight_df = correlation_score.reset_index(name='profile_corr_weight')
    
    return align_weights_to_dataframe(
        df,
        weight_df,
        id_cols,
        'profile_corr_weight'
    )

# ======================================================================================
# --- MAIN FUNCTIONS TO ACCESS ---
# ======================================================================================

def generate_weights_data(
    df: pd.DataFrame,
    sample_cols: list[str] = None,
    log_intensity_col: str = 'log10Intensity',
    adj_intensity_col: str = 'adjIntensity',
    control_condition: str = 'day1',
    condition_col: str = 'day',
    protein_col: str = 'protein_id',
    peptide_col: str = 'peptide_id',
    # Imputation Parameters
    is_real_col: str = None,
    is_comp_miss_col: str = None,
    true_val: float = 1.0,
    sparse_imputed_val: float = 1e-5,
    dense_imputed_val: float = 0.25,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate a DataFrame with all standard weight components calculated for given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing all required columns.
    sample_cols : list[str], optional
        Sample identifier columns, by default ['filename']
    log_intensity_col : str, optional
        Name of the log intensity column, by default 'log10Intensity'
    adj_intensity_col : str, optional
        Name of the adjusted intensity column, by default 'adjIntensity'
    control_condition : str, optional
        Name of the control condition, by default 'day1'
    condition_col : str, optional
        Name of the condition column, by default 'day'
    protein_col : str, optional
        Name of the protein column, by default 'protein_id'
    peptide_col : str, optional
        Name of the peptide column, by default 'peptide_id'
    verbose : bool, optional
        Whether to print progress and timing, by default True
    is_real_col : str, optional
        Column name indicating if the value is real (not imputed).
    is_comp_miss_col : str, optional
        Column name indicating if the peptide is completely missing for a condition.
    true_val : float, optional
        Weight for real (not imputed) values.
    sparse_imputed_val : float, optional
        Weight for sparsely imputed values.
    dense_imputed_val : float, optional
        Weight for densely imputed values.

    Returns
    -------
    pd.DataFrame
        DataFrame with all weight columns added (prefixed with 'W_').
    """
    group_cols = [protein_col, peptide_col, condition_col]
    if sample_cols is None:
        sample_cols = ['filename']
    intensity_cols = [log_intensity_col, adj_intensity_col]

    weight_names_with_intensity = [
        ('InverseMean', 'mean'),
        ('InverseVar', 'var'),
        ('InverseStd', 'std'),
        ('CorrDiscordance', None),
        ('RelativeVariability', None),
        ('SignalToNoise', None),
        ('ReplicateConcordance', None),
        ('DirectionalAgreement', None),
    ]

    weights_data = df[group_cols + sample_cols + intensity_cols].copy()
    # Add is_real and is_comp_miss columns if provided
    if is_real_col:
        if is_real_col not in df.columns:
            raise ValueError(f"Column '{is_real_col}' not found in the DataFrame.")
        weights_data[is_real_col] = df[is_real_col].astype(bool)
    if is_comp_miss_col:
        if is_comp_miss_col not in df.columns:
            raise ValueError(f"Column '{is_comp_miss_col}' not found in the DataFrame.")
        weights_data[is_comp_miss_col] = df[is_comp_miss_col].astype(bool)

    # Imputation Weights
    if verbose:
        print(" 📏 Calculating Imputation Weights...", end="")
    t0 = time.time()
    weights_data['W_Impute'] = calculate_imputation_weights(
        weights_data,
        is_real_col=is_real_col,
        is_comp_miss_col=is_comp_miss_col,
        true_val=true_val,
        sparse_imputed_val=sparse_imputed_val,
        dense_imputed_val=dense_imputed_val,
        verbose=verbose
    )
    t1 = time.time()
    if verbose:
        print(f" (done in {pretty_time(t1-t0)})")

    # # Replicate Imbalance Weights
    # if verbose:
    #     print(" 📏 Calculating Replicate Imbalance Weights...", end="")
    # t0 = time.time()
    # weights_data['W_RepImbalance'] = calculate_replicate_imbalance_weights(
    #     weights_data,
    #     condition_col=condition_col,
    #     protein_col=protein_col,
    #     peptide_col=peptide_col,
    #     is_real_col=is_real_col,
    #     verbose=verbose
    # )
    # t1 = time.time()
    # if verbose:
    #     print(f" (done in {pretty_time(t1-t0)})")

    # # Normalization Impact Weights
    # if verbose:
    #     print(" 📏 Calculating Normalization Impact Weights...", end="")
    # t0 = time.time()
    # weights_data['W_NormalizationImpact'] = calculate_normalization_impact_weights(
    #     weights_data,
    #     log_intensity_col=log_intensity_col,
    #     adj_intensity_col=adj_intensity_col,
    #     protein_col=protein_col,
    #     peptide_col=peptide_col,
    #     verbose=verbose
    # )
    # t1 = time.time()
    # if verbose:
    #     print(f" (done in {pretty_time(t1-t0)})")

    # Reverse Technical Variation Weights
    if verbose:
        print(" 📏 Calculating Reverse Technical Variation Weights...", end="")
    t0 = time.time()
    weights_data['W_RevTechVar'] = weights_data.groupby(
        [protein_col, peptide_col, condition_col], observed=True
    )[adj_intensity_col].transform('var').fillna(0.0)
    weights_data['W_RevTechVar'] = 1 - min_max_scale(weights_data['W_RevTechVar'])
    t1 = time.time()
    if verbose:
        print(f" (done in {pretty_time(t1-t0)})")

    # # Profile Correlation Weights
    # if verbose:
    #     print(" 📏 Calculating Profile Correlation Weights...", end="")
    # t0 = time.time()
    # weights_data['W_ProfileCorr'] = calculate_profile_correlation_weights(
    #     weights_data,
    #     log_intensity_col=log_intensity_col,
    #     adj_intensity_col=adj_intensity_col,
    #     condition_col=condition_col,
    #     protein_col=protein_col,
    #     peptide_col=peptide_col, 
    #     verbose=verbose
    # )
    # t1 = time.time()
    # if verbose:
    #     print(f" (done in {pretty_time(t1-t0)})")

    # Doesn't Really Improve...
    # # Weights with intensity
    # for wName, metric in weight_names_with_intensity:
    #     for icol, suffix in zip([log_intensity_col, adj_intensity_col], ['(log)', '(adj)']):
    #         colname = f'W_{wName}{suffix}'
    #         t0 = time.time()
    #         if wName == 'InverseMean':
    #             if verbose:
    #                 print(f" 📏 Calculating {wName} Weights ({suffix})...", end="")
    #             weights_data[colname] = calculate_inverse_metric_weights(
    #                 weights_data,
    #                 metric='mean',
    #                 intensity_col=icol,
    #                 group_cols=group_cols
    #             )
    #         elif wName == 'InverseVar':
    #             if verbose:
    #                 print(f" 📏 Calculating {wName} Weights ({suffix})...", end="")
    #             weights_data[colname] = calculate_inverse_metric_weights(
    #                 weights_data,
    #                 metric='var',
    #                 intensity_col=icol,
    #                 group_cols=group_cols
    #             )
    #         elif wName == 'InverseStd':
    #             if verbose:
    #                 print(f" 📏 Calculating {wName} Weights ({suffix})...", end="")
    #             weights_data[colname] = calculate_inverse_metric_weights(
    #                 weights_data,
    #                 metric='std',
    #                 intensity_col=icol,
    #                 group_cols=group_cols
    #             )
    #         elif wName == 'CorrDiscordance':
    #             if verbose:
    #                 print(f" 📏 Calculating {wName} Weights ({suffix})...", end="")
    #             weights_data[colname] = calculate_correlation_discordance_weights(
    #                 weights_data,
    #                 intensity_col=icol,
    #                 control_condition=control_condition,
    #                 condition_col=condition_col,
    #                 protein_col=protein_col,
    #                 peptide_col=peptide_col, 
    #                 verbose=verbose
    #             )
    #         elif wName == 'RelativeVariability':
    #             if verbose:
    #                 print(f" 📏 Calculating {wName} Weights ({suffix})...", end="")
    #             weights_data[colname] = calculate_relative_variability_weights(
    #                 weights_data,
    #                 intensity_col=icol,
    #                 condition_col=condition_col,
    #                 protein_col=protein_col,
    #                 peptide_col=peptide_col, 
    #                 verbose=verbose
    #             )
    #         elif wName == 'SignalToNoise':
    #             if verbose:
    #                 print(f" 📏 Calculating {wName} Weights ({suffix})...", end="")
    #             weights_data[colname] = calculate_generalized_signal_to_noise_weights(
    #                 weights_data,
    #                 intensity_col=icol,
    #                 condition_col=condition_col,
    #                 protein_col=protein_col,
    #                 peptide_col=peptide_col, 
    #                 verbose=verbose
    #             )
    #         elif wName == 'ReplicateConcordance':
    #             if verbose:
    #                 print(f" 📏 Calculating {wName} Weights ({suffix})...", end="")
    #             weights_data[colname] = calculate_replicate_concordance_weights(
    #                 weights_data,
    #                 intensity_col=icol,
    #                 condition_col=condition_col,
    #                 protein_col=protein_col,
    #                 peptide_col=peptide_col, 
    #                 verbose=verbose
    #             )
            # TODO: There is a bug in the adj intensity for this...
            # elif wName == 'DirectionalAgreement':
            #     if verbose:
            #         print(f" 📏 Calculating {wName} Weights ({suffix})...", end="")
            #     weights_data[colname] = calculate_directional_agreement_weights(
            #         weights_data,
            #         intensity_col=icol,
            #         control_condition=control_condition,
            #         condition_col=condition_col,
            #         protein_col=protein_col,
            #         peptide_col=peptide_col, 
            #         verbose=verbose
            #     )
            # t1 = time.time()
            # if verbose:
            #     print(f" (done in {pretty_time(t1-t0)})")

    # Normalize all W_ columns
    # weights_cols = [col for col in weights_data.columns if col.startswith('W_')]
    # weights_data[weights_cols] = weights_data[weights_cols].apply(min_max_scale, axis=0)
    if verbose:
        print("All weights calculated and normalized.")
    return weights_data


def select_components(
    df: pd.DataFrame, 
    std_threshold: float = 0.01, 
    corr_threshold: float = 0.95,
    verbose: bool = True
) -> List[str]:
    """
    Performs feature (weighting component) selection on a DataFrame by removing l
    ow-variance and highly correlated features.

    This function operates in two main steps:
    1.  Low-Variance Filtering: It removes features whose standard deviation is below a specified
        threshold. This is useful for eliminating constant or near-constant features.
    2.  High-Correlation Filtering: It calculates the correlation matrix and removes one feature
        from any pair with a correlation coefficient above a specified threshold. This helps to
        reduce multicollinearity. The implementation is optimized for speed on large datasets.

    Args:
        df (pd.DataFrame): 
            The input DataFrame containing the features to be analyzed.
        std_threshold (float, optional): 
            The minimum standard deviation a feature must have to be kept. 
            Defaults to 0.01.
        corr_threshold (float, optional): 
            The absolute correlation coefficient threshold. For any pair of features
            exceeding this value, one will be removed. Defaults to 0.95.
        verbose (bool, optional): 
            If True, prints detailed information about the selection process. 
            Defaults to True.

    Returns:
        List[str]: 
            A list of column names that should be kept after filtering.
    """
    if verbose:
        print("--- Starting Feature Selection ---")
        print(f"Initial number of features: {len(df.columns)}\n")

    # --- 1. Low Variance Check ---
    if verbose:
        print(f"Step 1: Removing features with standard deviation < {std_threshold}")
    
    stds = df.std()
    low_variance_cols = stds[stds < std_threshold].index.tolist()
    
    columns_to_keep = [col for col in df.columns if col not in low_variance_cols]
    
    if verbose:
        if low_variance_cols:
            print(f"Found and removed {len(low_variance_cols)} low-variance columns: {low_variance_cols}")
        else:
            print("No low-variance columns found.")
        print("-" * 20)

    # --- 2. High Correlation Check (Optimized for Speed) ---
    if verbose:
        print(f"\nStep 2: Removing highly correlated features with threshold > {corr_threshold}")

    # Create a new DataFrame with only the columns we are keeping so far
    df_filtered = df[columns_to_keep]
    
    # Calculate the absolute correlation matrix
    corr_matrix = df_filtered.corr().abs()
    
    # Create a boolean mask for the upper triangle of the matrix to avoid duplicate pairs
    upper_tri_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_tri = corr_matrix.where(upper_tri_mask)
    
    # Use stack() to convert the matrix to a Series, which is much faster to filter
    # This avoids slow Python loops.
    highly_correlated_series = upper_tri.stack()
    correlated_pairs = highly_correlated_series[highly_correlated_series > corr_threshold]
    
    # Get the names of the second column in each highly correlated pair to drop
    # Using a set ensures each column is only added once.
    columns_to_drop_corr = set(correlated_pairs.index.get_level_values(1))
    
    if verbose:
        if columns_to_drop_corr:
            print(f"Found and removed {len(columns_to_drop_corr)} highly correlated columns: {list(columns_to_drop_corr)}")
        else:
            print("No highly correlated columns found.")
        print("-" * 20)

    # Filter the final list of columns
    final_columns_to_keep = [col for col in columns_to_keep if col not in columns_to_drop_corr]
    
    if verbose:
        total_dropped = len(low_variance_cols) + len(columns_to_drop_corr)
        print("\n--- Feature Selection Summary ---")
        print(f"Total columns removed: {total_dropped}")
        print(f"Final columns to keep: {len(final_columns_to_keep)}")

    return final_columns_to_keep


def create_pls_component_weights(
    weights_df: pd.DataFrame,
    weights_cols: List[str],
    y_target: pd.Series,
    n_components: int = 10,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Reduces a set of weight vectors into a few supervised components using PLS.

    This function applies Partial Least Squares (PLS) to find composite weights
    (PLS components) that are highly correlated with the target variable (y_target).
    Each component is then scaled to a [0, 1] range.

    Args:
        weights_df (pd.DataFrame): DataFrame containing the pre-scaled weight columns.
        weights_cols (List[str]): A list of the names of the weight columns to analyze.
        y_target (pd.Series): The target variable for the model (e.g., adjIntensity).
        n_components (int, optional): The number of PLS components to generate.
                                     Defaults to 10.
        verbose (bool, optional): If True, prints a report on the components.

    Returns:
        pd.DataFrame: A DataFrame where columns are the scaled PLS components
                      (PLS1, PLS2, ...), ready for model testing.
    """
    if not isinstance(weights_df, pd.DataFrame):
        raise TypeError("weights_df must be a pandas DataFrame.")
    if not all(col in weights_df.columns for col in weights_cols):
        raise ValueError("One or more columns in weights_cols not found in weights_df.")
    if not isinstance(y_target, pd.Series):
        raise TypeError("y_target must be a pandas Series.")
    if not weights_df.index.equals(y_target.index):
        raise ValueError("Index of weights_df and y_target must match.")

    # 1. Initialize PLS to generate the desired number of components
    # We choose the number of components directly, rather than by variance explained.
    pls = PLSRegression(n_components=n_components)

    # 2. Fit PLS and get the component scores (the "X_scores_")
    # This is the supervised step, using both X (weights) and y (target)
    pls.fit(weights_df[weights_cols], y_target)
    pls_components = pls.x_scores_

    # 3. Create a new DataFrame to hold the results
    pls_df = pd.DataFrame(
        data=pls_components,
        index=weights_df.index,
        columns=[f'PLS{i+1}' for i in range(pls_components.shape[1])]
    )

    # 4. Scale each PLS component to a [0, 1] range
    for col in pls_df.columns:
        pls_df[col] = min_max_scale(pls_df[col])

    if verbose:
        print("--- PLS Supervised Dimensionality Reduction Report ---")
        print(f"Generated {n_components} components supervised by the target variable.")
        print("\nPLS Component 1 is engineered to have the highest possible covariance with the target.")
        print("Subsequent components explain residual variance.")
        # R-squared score shows how well the components predict the target.
        r_squared = pls.score(weights_df[weights_cols], y_target)
        print(f"\nModel Fit (R-squared using {n_components} components): {r_squared:.2%}")
        print("This shows how much variance in the target is explained by the PLS components.")

    return pls_df

def generate_refined_top5_mean_weights(
    weights_data: pd.DataFrame,
    weights_cols: list,
    y_target: pd.Series,
    verbose: bool = False
) -> pd.Series:
    """
    Generate the Refined_Top5_mean_all weight scheme based on PLS importance analysis.
    This function efficiently identifies the top 5 most important weights and creates
    an optimized PLS-based combination.
    
    Args:
        weights_data: DataFrame containing weight columns
        weights_cols: List of weight column names
        y_target: Target variable for PLS optimization
        verbose: Whether to print progress information
        
    Returns:
        pd.Series: The Refined_Top5_mean_all weight values
    """
    
    if verbose:
        print("🔧 Generating Refined_Top5_mean_all weights...")
    
    # Step 1: Fit PLS model to get component importance
    n_components = min(3, len(weights_cols), len(y_target) - 1)
    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(weights_data[weights_cols], y_target)
    
    # Step 2: Calculate overall weight importance
    loadings = pls_model.x_loadings_  # Shape: (n_features, n_components)
    abs_loadings = np.abs(loadings)
    
    # Get R² for each component to weight their importance
    r_squared_by_component = []
    for i in range(1, n_components + 1):
        pls_temp = PLSRegression(n_components=i)
        pls_temp.fit(weights_data[weights_cols], y_target)
        r_squared = pls_temp.score(weights_data[weights_cols], y_target)
        r_squared_by_component.append(r_squared)
    
    # Calculate component weights (incremental R² contribution)
    component_weights = []
    for i, r2 in enumerate(r_squared_by_component):
        if i == 0:
            component_weights.append(r2)
        else:
            increment = r2 - r_squared_by_component[i-1]
            component_weights.append(max(increment, 0))
    
    # Calculate overall importance scores
    overall_importance = np.zeros(len(weights_cols))
    for i in range(n_components):
        overall_importance += abs_loadings[:, i] * component_weights[i]
    
    # Step 3: Get top 5 most important weights
    importance_series = pd.Series(overall_importance, index=weights_cols)
    top_5_weights = importance_series.nlargest(5).index.tolist()
    
    if verbose:
        print(f"  📊 Top 5 most important weights: {top_5_weights}")
        max_r2 = max(r_squared_by_component)
        print(f"  📈 PLS performance: R² = {max_r2:.4f} ({max_r2:.2%})")
    
    # Step 4: Create refined PLS using only top 5 weights
    n_comp_refined = min(3, len(top_5_weights))
    refined_pls = create_pls_component_weights(
        weights_df=weights_data,
        weights_cols=top_5_weights,
        y_target=y_target,
        n_components=n_comp_refined,
        verbose=False
    )
    
    # Step 5: Calculate mean of all refined PLS components
    refined_top5_mean_all = min_max_scale(refined_pls.mean(axis=1))
    
    if verbose:
        # Test performance of refined weights
        pls_test = PLSRegression(n_components=n_comp_refined)
        pls_test.fit(weights_data[top_5_weights], y_target)
        r2_refined = pls_test.score(weights_data[top_5_weights], y_target)
        print(f"  ✅ Refined weights R²: {r2_refined:.4f} ({r2_refined:.2%})")
        print(f"  📏 Weight range: [{refined_top5_mean_all.min():.4f}, {refined_top5_mean_all.max():.4f}]")
    
    return refined_top5_mean_all

def _find_elbow_modl(values: List[float], threshold_ratio: float = 0.05) -> int:
    """Finds the elbow point in a list of increasing values (like R-squared)."""
    # **FIXED**: Ensure values is a numpy array to handle both list and array inputs consistently.
    values = np.asarray(values)

    # An elbow cannot be determined for 0 or 1 values. The check `values.size`
    # correctly handles numpy arrays, resolving the ambiguous truth value error.
    if values.size <= 1:
        return values.size
    
    # Calculate the change (gain) between consecutive values
    gains = np.diff(values)

    # Find the maximum gain
    max_gain = np.max(gains)
    
    # If there's no improvement anywhere (e.g., all R² values are the same),
    # just use the first component.
    if max_gain == 0:
        return 1

    # The elbow is where the gain drops below a certain percentage of the max gain.
    # This finds the first point of diminishing returns.
    try:
        # Find the index of the first gain that is smaller than the threshold.
        elbow_index = np.where(gains < max_gain * threshold_ratio)[0][0]
        # The number of components is the index + 1.
        return elbow_index + 1
    except IndexError:
        # If all gains are significant, it means we should use all components.
        return values.size
    
def _find_elbow_conf(values: List[float], threshold_ratio: float = 0.05) -> int:
    """Finds the elbow point in a list of increasing values (like R-squared)."""
    values = np.asarray(values)
    if values.size <= 1:
        return values.size
    
    gains = np.diff(values)
    if gains.size == 0:
        return 1
        
    max_gain = np.max(gains)
    if max_gain == 0:
        return 1

    try:
        elbow_index = np.where(gains < max_gain * threshold_ratio)[0][0]
        return elbow_index + 1
    except IndexError:
        return values.size


def generate_auto_pls_weights(
    weights_data: pd.DataFrame,
    weights_cols: List[str],
    y_target: pd.Series,
    max_components: int = 10,
    verbose: bool = False
) -> pd.Series:
    """
    Generates an optimal, data-agnostic weight scheme using PLS regression.

    This function automatically determines the optimal number of PLS components and the
    optimal number of features ('topN') to include, making it robust across different
    datasets. It works by:
    1.  Finding the optimal number of PLS components by looking for an "elbow" in the R² score.
    2.  Calculating feature importances based on the optimal components.
    3.  Automatically selecting the 'topN' most important features by finding an elbow in the
        sorted importance scores.
    4.  Building a final, refined PLS model using only the topN features to generate the final weights.

    Args:
        weights_data (pd.DataFrame): 
            DataFrame containing the feature columns.
        weights_cols (List[str]): 
            A list of the column names to be used as features.
        y_target (pd.Series): 
            The target variable for PLS optimization.
        max_components (int, optional): 
            The maximum number of PLS components to consider for importance analysis. 
            Defaults to 10.
        verbose (bool, optional): 
            If True, prints detailed progress and diagnostic information. 
            Defaults to False.

    Returns:
        pd.Series: 
            A min-max scaled Series representing the final, optimized weights.
    """
    if verbose:
        print("🔧 Auto-generating optimal PLS weights...")

    X = weights_data[weights_cols]
    n_features = X.shape[1]
    
    # Ensure n_components is valid
    actual_max_components = min(max_components, n_features, len(y_target) - 1)
    if actual_max_components < 1:
        raise ValueError("Not enough samples or features to run PLS.")

    # --- Step 1: Find the optimal number of components ---
    # **FIXED**: Use a robust loop to calculate R² for compatibility.
    # This is more reliable than using internal attributes of a single model.
    r_squared_by_comp = []
    for i in range(1, actual_max_components + 1):
        pls_temp = PLSRegression(n_components=i)
        pls_temp.fit(X, y_target)
        r_squared_by_comp.append(pls_temp.score(X, y_target))

    optimal_n_components = _find_elbow_modl(r_squared_by_comp)
    if verbose:
        print(f"  📈 R² scores by component: {[f'{r:.3f}' for r in r_squared_by_comp]}")
        print(f"  🎯 Determined optimal number of components for importance: {optimal_n_components}")

    # --- Step 2: Calculate overall feature importance ---
    # Fit one model with the optimal number of components to get the loadings.
    pls_full = PLSRegression(n_components=optimal_n_components)
    pls_full.fit(X, y_target)
    
    loadings = pls_full.x_loadings_
    abs_loadings = np.abs(loadings)

    # Incremental R² contribution for each component
    r2_gains = np.insert(np.diff(r_squared_by_comp[:optimal_n_components]), 0, r_squared_by_comp[0])
    component_weights = np.maximum(0, r2_gains)

    # Calculate overall importance scores by weighting loadings by component R² gain
    overall_importance = np.dot(abs_loadings, component_weights)
    importance_series = pd.Series(overall_importance, index=weights_cols).sort_values(ascending=False)

    # --- Step 3: Automatically determine the 'topN' features ---
    sorted_importances = importance_series.values
    # Find elbow in the importance scores to find the natural cutoff
    top_n = _find_elbow_modl(np.cumsum(sorted_importances) / np.sum(sorted_importances)) # Elbow on cumulative importance
    top_n_features = importance_series.nlargest(top_n).index.tolist()
    
    if verbose:
        print(f"  📊 Determined Top-{top_n} most important features: {top_n_features}")
        
    # --- Step 4: Build a refined PLS model using only the top features ---
    X_refined = weights_data[top_n_features]
    n_comp_refined = min(optimal_n_components, len(top_n_features))
    
    if n_comp_refined < 1:
        if verbose:
            print("  ⚠️ Warning: No features selected or not enough components. Returning a neutral weight.")
        return pd.Series(np.ones(len(weights_data)) * 0.5, index=weights_data.index)

    pls_refined = PLSRegression(n_components=n_comp_refined)
    pls_refined.fit(X_refined, y_target)

    # Transform the data into the new PLS component space
    final_components = pls_refined.transform(X_refined)

    # --- Step 5: Calculate the final weight as the mean of the components and scale ---
    # The mean combines the information from all refined components into a single score
    final_weight = pd.Series(final_components.mean(axis=1), index=weights_data.index)
    scaled_final_weight = pd.Series(min_max_scale(final_weight), index=weights_data.index)
    
    if verbose:
        r2_final = pls_refined.score(X_refined, y_target)
        print(f"  ✅ Final refined model (Top-{top_n} features, {n_comp_refined} comps) R²: {r2_final:.4f}")
        print(f"  📏 Final weight range: [{scaled_final_weight.min():.4f}, {scaled_final_weight.max():.4f}]")

    return scaled_final_weight

def generate_confidence_weights(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    y_target: pd.Series,
    max_components: int = 10,
    verbose: bool = False
) -> pd.Series:
    """
    Generates confidence weights for each data point for use in weighted correlation.

    This function builds an optimal PLS model to understand the relationship between
    the features and the target. It then calculates weights based on how well the
    model can predict each individual data point. Points that are well-explained by the
    model (low prediction error) are considered high-confidence and receive a high
    weight. Outliers or noisy points (high prediction error) receive a low weight.

    Args:
        features_df (pd.DataFrame): 
            DataFrame containing the feature columns.
        feature_cols (List[str]): 
            A list of the column names to be used as features.
        y_target (pd.Series): 
            The target variable for PLS optimization.
        max_components (int, optional): 
            The maximum number of PLS components to consider. Defaults to 10.
        verbose (bool, optional): 
            If True, prints detailed progress and diagnostic information. 
            Defaults to False.

    Returns:
        pd.Series: 
            A min-max scaled Series of confidence weights [0, 1] for each row.
    """
    if verbose:
        print("⚖️ Generating confidence weights for weighted correlation...")

    X = features_df[feature_cols]
    n_features = X.shape[1]
    
    # Ensure n_components is valid
    actual_max_components = min(max_components, n_features, len(y_target) - 1)
    if actual_max_components < 1:
        raise ValueError("Not enough samples or features to run PLS.")

    # --- Step 1: Find the optimal number of components for the model ---
    r_squared_by_comp = []
    for i in range(1, actual_max_components + 1):
        pls_temp = PLSRegression(n_components=i)
        pls_temp.fit(X, y_target)
        r_squared_by_comp.append(pls_temp.score(X, y_target))

    optimal_n_components = _find_elbow_conf(r_squared_by_comp)
    
    if verbose:
        print(f"  📈 R² scores by component: {[f'{r:.3f}' for r in r_squared_by_comp]}")
        print(f"  🎯 Determined optimal model complexity: {optimal_n_components} components")

    # --- Step 2: Build the final model and get predictions ---
    pls_final = PLSRegression(n_components=optimal_n_components)
    pls_final.fit(X, y_target)
    y_pred = pls_final.predict(X).flatten()

    if verbose:
        final_r2 = pls_final.score(X, y_target)
        print(f"  ✅ Final model R²: {final_r2:.4f}")

    # --- Step 3: Calculate weights based on model residuals ---
    # The absolute error (residual) tells us how "off" the prediction was for each point.
    residuals = np.abs(y_target - y_pred)
    
    # We scale the residuals to a [0, 1] range.
    scaled_residuals = min_max_scale(residuals)
    
    # The confidence weight is the inverse of the scaled residual.
    # Low error -> high weight; High error -> low weight.
    confidence_weights = 1 - scaled_residuals
    
    final_weights_series = pd.Series(confidence_weights, index=features_df.index)

    if verbose:
        print(f"  📏 Generated confidence weights. Range: [{final_weights_series.min():.4f}, {final_weights_series.max():.4f}]")

    return final_weights_series
