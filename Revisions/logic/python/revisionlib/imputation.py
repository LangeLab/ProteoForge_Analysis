#!/usr/bin/env python3
"""
Alternative imputation strategies for sensitivity analysis.

Each function operates on a wide-format DataFrame (rows = (Protein, Peptide),
columns = samples) with NaN representing missing values. Input is assumed to be
in LINEAR (raw) intensity scale, consistent with sims.amputation() output.

All functions accept a condition_sample_map dict and impute per-condition.
Output is a fully imputed DataFrame in the same format, suitable for passing
to sims.build_test_data(data=imputed).

Strategies implemented
----------------------
1. minprob_imputation  — Perseus-style MinProb: N(q1, 0.3*sigma) per condition
2. qrilc_imputation    — Truncated normal left-tail draw per condition
3. zero_fill           — Replace NaN with per-condition minimum observed value
4. global_mean_fill    — Replace NaN with per-condition mean
5. knn_imputation      — sklearn KNNImputer (k=5) per condition

Usage (from Simulation/ directory):
    import imputation_strategies as imp
    imputed = imp.minprob_imputation(missing_data, condition_sample_map)
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.impute import KNNImputer


def complete_condition_missing_mask(
    data: pd.DataFrame,
    condition_sample_map: dict,
) -> pd.DataFrame:
    """Return a boolean mask for values belonging to fully missing condition blocks."""
    mask = pd.DataFrame(False, index=data.index, columns=data.columns)
    for samples in condition_sample_map.values():
        block_missing = data[samples].isna()
        full_missing_rows = block_missing.all(axis=1)
        if full_missing_rows.any():
            mask.loc[full_missing_rows, samples] = True
    return mask


def hybrid_mnar_impute(
    data: pd.DataFrame,
    condition_sample_map: dict,
    complete_condition_imputer,
    sparse_n_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Keep sparse missingness handling fixed with kNN and only swap the
    complete-condition imputation strategy.

    This matches the revision plan: alternative strategies should replace the
    MNAR complete-condition step, not the sparse within-condition fill.
    """
    sparse_filled = knn_impute(
        data,
        condition_sample_map,
        n_neighbors=sparse_n_neighbors,
        is_log2=False,
    )
    complete_filled = complete_condition_imputer(data.copy(), condition_sample_map)
    complete_mask = complete_condition_missing_mask(data, condition_sample_map)

    hybrid = sparse_filled.copy()
    hybrid[complete_mask] = complete_filled[complete_mask]
    return hybrid


# ==============================================================================
# 1. MinProb (Perseus default)
# ==============================================================================

def minprob_imputation(
    data: pd.DataFrame,
    condition_sample_map: dict,
    quantile: float = 0.01,
    sd_factor: float = 0.3,
    is_log2: bool = False,
    impute_all: bool = False,
    seed: int = None,
) -> pd.DataFrame:
    """
    Perseus-style MinProb imputation.

    For each condition draws imputed values from N(q1, 0.3 * sigma_obs) where
    q1 is the `quantile`-th percentile of the observed log2 distribution in
    that condition and sigma_obs is the observed standard deviation.

    Args:
        data:               Wide DataFrame (linear scale unless is_log2=True).
        condition_sample_map: {condition: [sample_columns]}.
        quantile:           Lower quantile for the mean of the imputation
                            distribution (default 0.01 = 1st percentile).
        sd_factor:          Fraction of observed SD to use (default 0.3).
        is_log2:            Whether the data is already in log2 scale.
        impute_all:         If True, impute ALL NaN; if False, impute only
                            complete-condition NaN (all reps missing).
        seed:               Random seed.

    Returns:
        DataFrame with imputed values (same scale as input).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    imputed = np.log2(data).copy() if not is_log2 else data.copy()

    for _cond, samples in condition_sample_map.items():
        block = imputed[samples]
        observed = block.values[~np.isnan(block.values)]
        if len(observed) == 0:
            continue

        q_val = float(np.percentile(observed, quantile * 100))
        sd_obs = float(np.std(observed, ddof=1))
        sd_imp = sd_factor * sd_obs

        is_miss = block.isna()
        if impute_all:
            mask = is_miss.any(axis=1)
        else:
            mask = is_miss.all(axis=1)

        n_rows = mask.sum()
        if n_rows == 0:
            continue

        n_cols = len(samples)
        vals = rng.normal(q_val, sd_imp, size=(n_rows, n_cols))
        # Only fill where actually missing
        target = block.loc[mask].copy()
        target_miss = target.isna()
        arr = target.to_numpy(dtype=float, copy=True)
        arr[target_miss.to_numpy()] = vals[target_miss.to_numpy()]
        imputed.loc[mask, samples] = pd.DataFrame(arr, index=target.index, columns=target.columns)

    if not is_log2:
        imputed = np.power(2, imputed)
    return imputed


# ==============================================================================
# 2. QRILC (Quantile Regression Imputation of Left-Censored data)
# ==============================================================================

def qrilc_imputation(
    data: pd.DataFrame,
    condition_sample_map: dict,
    is_log2: bool = False,
    impute_all: bool = False,
    seed: int = None,
) -> pd.DataFrame:
    """
    QRILC-style imputation using truncated normal draws from the left tail.

    For each condition: estimate the mean and SD of the observed log2 values,
    then draw imputed values from a truncated normal distribution whose upper
    bound equals the minimum observed value in that condition.

    Args:
        data:               Wide DataFrame (linear scale unless is_log2=True).
        condition_sample_map: {condition: [sample_columns]}.
        is_log2:            Whether the data is already in log2 scale.
        impute_all:         If True, impute ALL NaN; if False, impute only
                            complete-condition NaN.
        seed:               Random seed.

    Returns:
        DataFrame with imputed values.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    imputed = np.log2(data).copy() if not is_log2 else data.copy()

    for _cond, samples in condition_sample_map.items():
        block = imputed[samples]
        observed = block.values[~np.isnan(block.values)]
        if len(observed) < 3:
            continue

        mu_obs = float(np.mean(observed))
        sd_obs = float(np.std(observed, ddof=1))
        if sd_obs == 0:
            sd_obs = 1e-6
        upper = float(np.min(observed))

        # Truncated normal: (a, b) in standard-normal units
        a_std = -np.inf
        b_std = (upper - mu_obs) / sd_obs

        is_miss = block.isna()
        if impute_all:
            mask = is_miss.any(axis=1)
        else:
            mask = is_miss.all(axis=1)

        n_rows = mask.sum()
        if n_rows == 0:
            continue

        n_cols = len(samples)
        vals = scipy_stats.truncnorm.rvs(
            a_std, b_std, loc=mu_obs, scale=sd_obs,
            size=(n_rows, n_cols), random_state=rng,
        )
        target = block.loc[mask].copy()
        target_miss = target.isna()
        arr = target.to_numpy(dtype=float, copy=True)
        arr[target_miss.to_numpy()] = vals[target_miss.to_numpy()]
        imputed.loc[mask, samples] = pd.DataFrame(arr, index=target.index, columns=target.columns)

    if not is_log2:
        imputed = np.power(2, imputed)
    return imputed


# ==============================================================================
# 3. Zero fill (min-value fill)
# ==============================================================================

def zero_fill(
    data: pd.DataFrame,
    condition_sample_map: dict,
    is_log2: bool = False,
    impute_all: bool = False,
) -> pd.DataFrame:
    """
    Replace NaN with the per-condition minimum observed value.

    Args:
        data:               Wide DataFrame (linear scale unless is_log2=True).
        condition_sample_map: {condition: [sample_columns]}.
        is_log2:            Whether the data is already in log2 scale.
        impute_all:         If True, fill ALL NaN; if False, fill only
                            complete-condition NaN.

    Returns:
        DataFrame with imputed values.
    """
    imputed = np.log2(data).copy() if not is_log2 else data.copy()

    for _cond, samples in condition_sample_map.items():
        block = imputed[samples]
        observed = block.values[~np.isnan(block.values)]
        if len(observed) == 0:
            continue

        fill_val = float(np.min(observed))

        is_miss = block.isna()
        if impute_all:
            mask = is_miss.any(axis=1)
        else:
            mask = is_miss.all(axis=1)

        target = block.loc[mask].copy()
        target = target.fillna(fill_val)
        imputed.loc[mask, samples] = target

    if not is_log2:
        imputed = np.power(2, imputed)
    return imputed


# ==============================================================================
# 4. Global mean fill (per condition)
# ==============================================================================

def global_mean_fill(
    data: pd.DataFrame,
    condition_sample_map: dict,
    is_log2: bool = False,
    impute_all: bool = False,
) -> pd.DataFrame:
    """
    Replace NaN with the per-condition mean of observed values.

    Args:
        data:               Wide DataFrame (linear scale unless is_log2=True).
        condition_sample_map: {condition: [sample_columns]}.
        is_log2:            Whether the data is already in log2 scale.
        impute_all:         If True, fill ALL NaN; if False, fill only
                            complete-condition NaN.

    Returns:
        DataFrame with imputed values.
    """
    imputed = np.log2(data).copy() if not is_log2 else data.copy()

    for _cond, samples in condition_sample_map.items():
        block = imputed[samples]
        observed = block.values[~np.isnan(block.values)]
        if len(observed) == 0:
            continue

        fill_val = float(np.mean(observed))

        is_miss = block.isna()
        if impute_all:
            mask = is_miss.any(axis=1)
        else:
            mask = is_miss.all(axis=1)

        target = block.loc[mask].copy()
        target = target.fillna(fill_val)
        imputed.loc[mask, samples] = target

    if not is_log2:
        imputed = np.power(2, imputed)
    return imputed


# ==============================================================================
# 5. kNN imputation (per condition, k=5)
# ==============================================================================

def knn_impute(
    data: pd.DataFrame,
    condition_sample_map: dict,
    n_neighbors: int = 5,
    is_log2: bool = False,
) -> pd.DataFrame:
    """
    K-nearest-neighbours imputation per condition.

    Imputes ALL missing values (both sparse and complete-condition) in one
    pass — kNN does not distinguish MNAR vs MAR. This is intentional: it
    tests the MAR assumption against the MNAR reality.

    Args:
        data:               Wide DataFrame (linear scale unless is_log2=True).
        condition_sample_map: {condition: [sample_columns]}.
        n_neighbors:        k for KNNImputer.
        is_log2:            Whether the data is already in log2 scale.

    Returns:
        DataFrame with imputed values.
    """
    imputed = np.log2(data).copy() if not is_log2 else data.copy()

    for _cond, samples in condition_sample_map.items():
        block = imputed[samples]
        if block.isna().sum().sum() == 0:
            continue
        imp = KNNImputer(n_neighbors=n_neighbors)
        imputed[samples] = imp.fit_transform(block)

    if not is_log2:
        imputed = np.power(2, imputed)
    return imputed
