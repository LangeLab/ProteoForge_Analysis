import re
import time
from collections import Counter
from itertools import combinations
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import minmax_scale

# ======================================================================================
# Global Variables and Settings
# ======================================================================================
# TODO list:
# - Move the styling defaults to here
# - 


# ======================================================================================
# Notebook Utility Functions
# ======================================================================================

def getTime() -> float:
    """
        Get the current time for timer

        Returns:
            float: The current time in seconds.
    """
    return time.time()

def prettyTimer(
        seconds: float
    ) -> str:
    """
        Better way to show elapsed time

        Args:
            seconds (float): The number of seconds to convert to a pretty format.

        Returns:
            str: The elapsed time in a pretty format.
        
        Examples:
            >>> prettyTimer(100)
            '00h:01m:40s'

            >>> prettyTimer(1000)
            '00h:16m:40s'

            >>> prettyTimer(10000)
            '02h:46m:40s'
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02dh:%02dm:%02ds" % (h, m, s)

def view_table(
        data: pd.DataFrame, 
        page_size: int = 25, 
        page_number: int = None
    ):
    """
        Displays a table with pages with current page or 
            with all pages if not specified. 
            Going around the VScodes view limit.
    
    """
    from IPython.display import display
    n_pages = len(data) // page_size + 1
    if page_number is not None:
        print(f"Total pages: {n_pages}, Current page: {page_number}")
        if page_number < 1 or page_number > n_pages:
            print(f"Invalid page number. Please select a page between 1 and {n_pages}.")
        else:
            display(data.iloc[(page_number - 1) * page_size:page_number * page_size])
    else:
        print(f"Total pages: {n_pages}")
        for i in range(n_pages):
            display(data.iloc[i * page_size:(i + 1) * page_size])

def print_shape(
        df: pd.DataFrame, 
        identifier: str ="", 
        behavior: str ="print"
    ) -> None:
    """
        Print the shape of a pandas dataframe.

        Args:
            df (pd.DataFrame): The pandas dataframe to print.
            identifier (str, optional): The identifier to print. Defaults to "".
            behavior (str, optional): The behavior of the function. 
                "print" to print the shape, "return" to return the shape. Defaults to "print".  
        
        Raises:
            TypeError: If df is not a pandas dataframe.
            TypeError: If identifier is not a string.
            ValueError: If behavior is not "print" or "return".

        Examples:
            >>> print_shape(pd.DataFrame(), "My Data", "print")
            My Data data has 0 rows and 0 columns

            >>> print_shape(pd.DataFrame(), "My Data", "return")
            'My Data data has 0 rows and 0 columns'

            >>> print_shape(pd.DataFrame(), "My Data", "invalid")
            ValueError: behavior must be either "print" or "return"
    """
    if behavior == "print":
        print(f"{identifier} data has {df.shape[0]:,} rows and {df.shape[1]:,} columns")
    elif behavior == "return":
        return f"{identifier} data has {df.shape[0]:,} rows and {df.shape[1]:,} columns"
    else:
        raise ValueError("behavior must be either 'print' or 'return'")

def print_series(
        series: pd.Series, 
        header: str = None, 
        tab: int = 0,
        elements_with_order: list = None
    ) -> None:
    """
        Print a pandas series with an optional header

        Args:
            series (pd.Series): The pandas series to print.
            header (str, optional): The header to print. Defaults to None.
            tab (int, optional): The number of spaces to print before each element. Defaults to 0.
            elements_with_order (list, optional): A list of elements to print. Defaults to None.

        Raises:
            TypeError: If series is not a pandas series.
            TypeError: If header is not a string.
            TypeError: If tab is not an integer.
            TypeError: If elements_with_order is not a list.
            ValueError: If tab is less than 0.

        Examples:
            >>> print_series(pd.Series([1, 2, 3]), "My Series", 4, ["a", "b", "c"])
            My Series
                a -> 1
                b -> 2
                c -> 3

            >>> print_series(pd.Series([1, 2, 3]), "My Series", 4)
            My Series
                0 -> 1
                1 -> 2
                2 -> 3

            >>> print_series(pd.Series([1, 2, 3]))
                0 -> 1
                1 -> 2
                2 -> 3
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas series")
    if not isinstance(header, str) and header is not None:
        raise TypeError("header must be a string")
    if not isinstance(tab, int):
        raise TypeError("tab must be an integer")
    if not isinstance(elements_with_order, list) and elements_with_order is not None:
        raise TypeError("elements_with_order must be a list")
    if tab < 0:
        raise ValueError(
            """
            tab must be a positive integer amount.Indicating the empty space prior to printing each element
            """
        )

    if header is not None:
        print(header)
    if elements_with_order is not None:
        for i in elements_with_order:
            if i in series.index:
                print(" "*tab, i, "->", series[i])
    else:
        for index, value in series.items():
                print(" "*tab, index, "->", value)

def print_list(
        data: list,
        n_elements: 3,
        description: str = "Preview of the list: ",
    ):
    """ 
        Print a preview of a list with a description and a specified 
        number of elements to show from the start and end of the list.
    """
    # Print the description
    print(
        f"{description} {data[:n_elements]}...{data[-n_elements:]}"
    )

# ======================================================================================
# Data Quality Check Functions
# ======================================================================================

def cv_numpy(
        x: np.ndarray, 
        axis: int = 1, 
        ddof: int = 1, 
        ignore_nan: bool = True, 
        format: str = "percent"
    ) -> np.ndarray:
    """
        Calculates the coefficient of variation of the values in the passed array.
        
        Args:
            x (np.ndarray): The array of values.
            axis (int, optional): The axis along which to calculate the coefficient of variation. Defaults to 1.
            ddof (int, optional): The degrees of freedom. Defaults to 1.
            ignore_nan (bool, optional): If True, NaN values are ignored. Defaults to False.
            format (str, optional): The format of the output. "percent" for percentage, "ratio" for ratio. 
                Defaults to "percent".

        Returns:
            np.ndarray: The coefficient of variation of the values in the passed array.

        Raises:
            TypeError: If x is not a numpy array.
            TypeError: If axis is not an integer.
            TypeError: If ddof is not an integer.

        Examples:
            >>> cv_numpy(np.array([1, 2, 3, 4, 5]))
            47.14045207910317

            >>> cv_numpy(np.array([[1, 2, 3], [4, 5, 6]]), axis=0)
            array([47.14045208, 47.14045208, 47.14045208])

            >>> cv_numpy(np.array([1, 2, 3, 4, 5]), ignore_nan=True)
            47.14045207910317

            >>> cv_numpy(np.array([1, 2, 3, 4, 5]), format="ratio")
            0.4714045207910317

            >>> cv_numpy(np.array([1, 2, 3, 4, 5]), format="percent")
            47.14045207910317
    """
    # Check if x is a numpy array
    if not isinstance(x, np.ndarray):
        try: 
            x = np.asarray(x)
        except:
            raise TypeError("Input x must be an array-like object.")
        
    # Check if axis is an integer
    if not isinstance(axis, int):
        raise TypeError("Input axis must be an integer. [0,1]")
    
    # Check if ddof is an integer
    if not isinstance(ddof, int):
        raise TypeError("Input ddof must be an integer.")

    # If ignore_nan use np.nanstd and np.nanmean
    # If input is scalar or empty, return np.nan
    if np.size(x) <= 1:
        return np.nan
    if ignore_nan:
        mean = np.nanmean(x, axis=axis)
        std = np.nanstd(x, axis=axis, ddof=ddof)
    else:
        mean = np.mean(x, axis=axis)
        std = np.std(x, axis=axis, ddof=ddof)
    # If mean is zero, return np.nan to avoid division by zero
    if np.any(mean == 0):
        return np.nan
    cv = std / mean
    if format == "ratio":
        return cv
    elif format == "percent":
        return cv * 100
    else:
        return cv

def scale_the_data(
        data: pd.DataFrame, 
        method: str="zscore",
        axis: int=1, 
        is_log: bool=False
    ):

    """
    Scale a DataFrame using various methods along rows or columns.

    Args:
        data (pd.DataFrame): The data to be scaled.
        method (str, optional): Scaling method. 
            One of 'zscore', 'minmax', 'foldchange', 'log2', 'log10'. Defaults to 'zscore'.
        axis (int, optional): Axis to scale along. 
            0 = column-wise, 1 = row-wise. Defaults to 1.
        is_log (bool, optional): If True, treat data as log-transformed for foldchange. 
            Defaults to False.
        verbose (bool, optional): If True, print debug info. 
            Defaults to False.

    Returns:
        pd.DataFrame: The scaled data.

    Raises:
        ValueError: If input is not a DataFrame, axis is not 0/1, or method is invalid.

    Examples:
        >>> scale_the_data(df, method='zscore', axis=1)
        >>> scale_the_data(df, method='minmax', axis=0)
        >>> scale_the_data(df, method='foldchange', axis=1, is_log=True)
    """


    # --- Input validation ---
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if axis not in (0, 1):
        raise ValueError("Axis must be 0 (columns) or 1 (rows).")
    valid_methods = ["zscore", "minmax", "foldchange", "log2", "log10"]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}.")

    idx, cols = data.index, data.columns
    # For debug/trace
    verbose = False  # Set to True for debug output

    # --- Z-score Standardization ---
    if method == "zscore":
        if verbose: print(f"Z-score scaling (axis={axis})")
        if axis == 0:
            mean = data.mean(axis=axis)
            std = data.std(axis=axis)
            res = (data - mean) / std
        else:
            mean = data.mean(axis=axis)
            std = data.std(axis=axis)
            res = (data.sub(mean, axis=0)).div(std, axis=0)
        # Handle division by zero std
        res = res.replace([np.inf, -np.inf], np.nan)

    # --- Min-Max Scaling ---
    elif method == "minmax":
        if verbose: print(f"Min-max scaling (axis={axis})")
        if axis == 0:
            min_ = data.min(axis=axis)
            max_ = data.max(axis=axis)
            denom = max_ - min_
            denom[denom == 0] = np.nan
            res = (data - min_) / denom
        else:
            min_ = data.min(axis=axis)
            max_ = data.max(axis=axis)
            denom = max_ - min_
            denom[denom == 0] = np.nan
            res = data.sub(min_, axis=0).div(denom, axis=0)
        res = res.replace([np.inf, -np.inf], np.nan)

    # --- Fold-change Scaling ---
    elif method == "foldchange":
        if verbose: print(f"Foldchange scaling (axis={axis}, is_log={is_log})")
        mean = data.mean(axis=axis)
        if axis == 0:
            if is_log:
                res = data - mean
            else:
                mean[mean == 0] = np.nan
                res = data / mean
        else:
            if is_log:
                res = data.sub(mean, axis=0)
            else:
                mean[mean == 0] = np.nan
                res = data.div(mean, axis=0)
        res = res.replace([np.inf, -np.inf], np.nan)

    # --- Log2 Scaling ---
    elif method == "log2":
        if verbose: print("Log2 scaling")
        res = np.log2(data)

    # --- Log10 Scaling ---
    elif method == "log10":
        if verbose: print("Log10 scaling")
        res = np.log10(data)

    # --- Output ---
    # Preserve dtypes if possible, always preserve index/columns
    result = pd.DataFrame(res, index=idx, columns=cols)
    if verbose:
        print(f"Result shape: {result.shape}, dtypes: {result.dtypes.value_counts().to_dict()}")
    return result

def get_quantification_stats(df: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """
    Calculates the count and percentage of quantified (non-NA) values per row for specified conditions.

    This function is useful for assessing data completeness across different experimental groups.

    Args:
        df (pd.DataFrame): The input DataFrame containing measurement columns.
        conditions (list[str]): A list of keywords to identify columns for each condition
                                (e.g., ['Control', 'AFII', 'Treated']).

    Returns:
        pd.DataFrame: A DataFrame with columns for the count and percentage of
                      quantified values for each condition.
    """

    all_stats = {}
    for condition in conditions:
        # Find all columns that match the condition keyword
        condition_cols = df.columns[df.columns.str.contains(condition)]
        
        # Gracefully handle cases where no columns match
        if condition_cols.empty:
            print(f"⚠️ Warning: No columns found for condition '{condition}'. Skipping.")
            continue
        
        # Count non-NA values per row for the matched columns
        quantified_count = df[condition_cols].notna().sum(axis=1)
        
        # Calculate the percentage based on the total number of columns for that condition
        total_cols_in_condition = len(condition_cols)
        quantified_pctl = (quantified_count / total_cols_in_condition) * 100
        
        # Add the results to our dictionary
        all_stats[f'{condition} Quantified Count'] = quantified_count
        all_stats[f'{condition} Quantified Percentage'] = quantified_pctl.round(2) # Rounding for clarity

    return pd.DataFrame(all_stats)

def calculate_correlation_matrix(
        data_df: pd.DataFrame, 
        method: str = 'spearman', 
        verbose: bool = False
    ) -> pd.DataFrame:
    """Calculates the pairwise correlation matrix for a given dataset."""
    if verbose: print(f"  Calculating {method} correlation matrix...")
    return data_df.corr(method=method)


def calculate_quantification_matrix(
        data_df: pd.DataFrame, 
        as_ratio: bool = False, 
        verbose=False
    ) -> pd.DataFrame:
    """Calculates the pairwise protein quantification matrix."""
    if verbose: print("  Calculating pairwise quantification matrix...")
    samples = data_df.columns
    total_proteins = len(data_df)
    quant_matrix = pd.DataFrame(1.0, index=samples, columns=samples, dtype=float)

    for s1, s2 in combinations(samples, 2):
        common_count = data_df[[s1, s2]].dropna().shape[0]
        quant_ratio = common_count / total_proteins
        quant_matrix.loc[s1, s2] = quant_matrix.loc[s2, s1] = quant_ratio
        
    return quant_matrix if as_ratio else quant_matrix * 100


def calculate_cv_matrix(
        data_df: pd.DataFrame, 
        as_ratio: bool = False, 
        verbose=False
    ) -> pd.DataFrame:
    """Calculates the pairwise median Coefficient of Variation (CV) matrix."""
    if verbose: print("  Calculating pairwise CV matrix...")
    samples = data_df.columns
    cv_matrix = pd.DataFrame(0.0, index=samples, columns=samples, dtype=float)

    for s1, s2 in combinations(samples, 2):
        subset = data_df[[s1, s2]].dropna()
        if len(subset) > 1:
            cv_vals = cv_numpy(subset.values, axis=1, ignore_nan=True, format="ratio")
            cv_matrix.loc[s2, s1] = np.nanmedian(cv_vals)
        else:
            cv_matrix.loc[s2, s1] = np.nan
            
    return cv_matrix if as_ratio else cv_matrix * 100


def create_composite_pairwise_matrix(
        matrices: dict[str, pd.DataFrame],
        weights: dict[str, float] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
    """
    Creates a robust, composite pairwise score matrix from multiple metric matrices.

    This function normalizes each metric to a 0-1 scale before applying weights,
    preventing any single metric from disproportionately influencing the result.

    Args:
        matrices (dict): A dictionary where keys are metric names (e.g., 'correlation')
                         and values are the corresponding pairwise metric DataFrames.
                         It assumes 'cv' is the only metric where lower is better.
        weights (dict, optional): A dictionary with the same keys as `matrices`,
                                  specifying the importance of each metric.
                                  Defaults to equal weights if None.

    Returns:
        pd.DataFrame: A single, composite pairwise score matrix.
    """
    if weights is None:
        # Default to equal weights if none are provided
        weights = {key: 1.0 for key in matrices.keys()}

    # Ensure all matrices and weights are aligned
    sample_names = list(matrices.values())[0].columns
    composite_matrix = pd.DataFrame(0.0, index=sample_names, columns=sample_names)
    total_weight = sum(weights.values())

    if total_weight == 0:
        return composite_matrix # Avoid division by zero

    for name, matrix in matrices.items():
        if name not in weights or weights[name] == 0:
            continue

        # --- Score Transformation (Higher is always better) ---
        if name == 'cv':
            # For CV, lower is better, so we invert the score.
            # Adding a small epsilon to avoid division by zero if CV is 0.
            score_matrix = 1 / (1 + matrix)
        else:
            # For correlation and quantification, higher is already better.
            score_matrix = matrix.copy()

        # --- Normalization and Weighting ---
        # Normalize the score from 0 to 1 to ensure fair contribution
        # We must handle the diagonal (self-to-self score) separately
        diag_vals = np.diagonal(score_matrix).copy()
        np.fill_diagonal(score_matrix.values, np.nan) # Ignore diagonal for scaling
        
        normalized_values = minmax_scale(score_matrix.values[np.isfinite(score_matrix.values)])
        score_matrix.values[np.isfinite(score_matrix.values)] = normalized_values
        
        # Restore the diagonal (usually a perfect score of 1.0)
        np.fill_diagonal(score_matrix.values, diag_vals)
        score_matrix.fillna(1.0, inplace=True) # Fill diagonal NaNs with 1

        # Add the weighted, normalized score to the composite matrix
        composite_matrix += score_matrix * weights[name]

    # Final score is the weighted average
    final_matrix = composite_matrix / total_weight
    return final_matrix

def find_outliers_from_matrix(
    composite_matrix: pd.DataFrame,
    iqr_multiplier: float = 1.5,
    verbose: bool = True
) -> list:
    """
    Identifies outliers from a composite score matrix using the IQR method.

    Args:
        composite_matrix (pd.DataFrame): The final pairwise composite score matrix.
        iqr_multiplier (float): The multiplier for the IQR range.
        verbose (bool): If True, prints the detection threshold and results.

    Returns:
        list: A list of sample names identified as outliers.
    """
    # Calculate the mean score for each sample, strictly excluding the diagonal
    matrix = composite_matrix.copy()
    np.fill_diagonal(matrix.values, np.nan)
    mean_scores = matrix.mean(axis=1)

    # Use the IQR method to find outliers on the lower end
    q1 = mean_scores.quantile(0.25)
    q3 = mean_scores.quantile(0.75)
    iqr = q3 - q1
    threshold = q1 - (iqr * iqr_multiplier)

    outliers = mean_scores[mean_scores < threshold].index.tolist()

    if verbose:
        print(f"  [IQR Method] Outlier Threshold: < {threshold:.3f}. Found {len(outliers)} outlier(s): {outliers}")
    
    return outliers

def run_metric_combination_analysis(
    data_df: pd.DataFrame,
    metric_weights: dict = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Performs an exhaustive outlier analysis by testing all combinations of metrics
    and provides a cleanly formatted summary.

    Args:
        data_df (pd.DataFrame): The data for a single condition.
        metric_weights (dict, optional): Weights for combining metrics.
        verbose (bool): If True, prints detailed step-by-step analysis.
    """
    # Pre-condition check: A minimum of 4 samples are needed for robust detection.
    if data_df.shape[1] <= 3:
        print("\nSKIPPING ANALYSIS: A minimum of 4 samples is required.")
        return pd.DataFrame()

    outlier_tally = Counter()
    outlier_details = {}

    if verbose:
        print("  Pre-calculating base metric matrices...")
    base_matrices = {
        'correlation': calculate_correlation_matrix(data_df),
        'quantification': calculate_quantification_matrix(data_df, as_ratio=True),
        'cv': calculate_cv_matrix(data_df, as_ratio=True)
    }

    metric_names = list(base_matrices.keys())
    
    for r in range(1, len(metric_names) + 1):
        for metric_group in combinations(metric_names, r):
            analysis_name = " & ".join(metric_group)
            if verbose:
                print(f"\n--- Analyzing with metrics: {analysis_name} ---")

            matrices_for_analysis = {name: base_matrices[name] for name in metric_group}
            composite_matrix = create_composite_pairwise_matrix(matrices_for_analysis, weights=metric_weights)
            
            # The outlier detection function is now simpler
            outliers = find_outliers_from_matrix(composite_matrix, verbose=verbose)

            if outliers:
                outlier_tally.update(outliers)
                for outlier in outliers:
                    outlier_details.setdefault(outlier, []).append(analysis_name)

    if verbose:
        # --- Generate and Print Final Report ---
        header = f" FINAL REPORT FOR {data_df.columns.name or 'CONDITION'} "
        print("\n" + f"{header:=^64}")

    if not outlier_tally:
        if verbose:
            print("No consistent outliers were identified.")
            print("=" * 64)
        return pd.DataFrame()

    report_df = pd.DataFrame(outlier_tally.items(), columns=['Sample', 'Outlier_Vote_Count'])
    report_df['Flagged_In'] = report_df['Sample'].map(outlier_details)
    report_df = report_df.sort_values(by='Outlier_Vote_Count', ascending=False).reset_index(drop=True)
    
    if verbose:
        # Use to_string() for pretty table formatting
        print(report_df.to_string(index=False))
        print("=" * 64)
    
    return report_df


def identify_outlier_samples(
        data: pd.DataFrame,
        sample_groups: Dict[str, List[str]],
        analysis_func: Callable[..., pd.DataFrame],
        min_outlier_occurrence: int = 3,
        verbose: bool = True,
        **analysis_kwargs: Any
    ) -> tuple[List[str], pd.DataFrame]:
    """
    Identifies outlier samples across all groups and returns a consolidated report.

    Args:
        data: The full data matrix (samples as columns).
        sample_groups: Dictionary mapping group names to lists of sample names.
        analysis_func: Function to run outlier analysis on a group's data.
                       Must return a DataFrame with ['Sample', 'Outlier_Vote_Count'].
        min_outlier_occurrence: Minimum votes to consider a sample an outlier.
        verbose: If True, prints progress and a summary.
        **analysis_kwargs: Additional keyword arguments for analysis_func.

    Returns:
        A tuple containing:
        - all_outliers (List[str]): A single list of unique outlier sample names from all groups.
        - combined_report (pd.DataFrame): A single DataFrame containing all analysis
                                          reports, with a 'Condition' column added.
    """
    all_outliers = []
    report_list = []

    for group, samples in sample_groups.items():
        if verbose:
            print(f"\n{'='*20} RUNNING META-ANALYSIS FOR: {group} {'='*20}")
            
        group_data = data[samples].copy()
        group_data.columns.name = f"{group} Samples"
        
        report = analysis_func(group_data, verbose=verbose, **analysis_kwargs)

        if not report.empty:
            # Add the condition label to the report
            report['Condition'] = group
            report_list.append(report)
            
            # Find and store outliers for this group
            outliers = report.loc[report['Outlier_Vote_Count'] >= min_outlier_occurrence, 'Sample'].tolist()
            all_outliers.extend(outliers)
            
            if verbose:
                print(f" - Outliers found in {group} (>= {min_outlier_occurrence} votes): {outliers}")

    # Combine all individual reports into a single DataFrame
    combined_report = pd.concat(report_list, ignore_index=True) if report_list else pd.DataFrame()
    
    # Return a unique list of all outliers and the combined report
    return list(set(all_outliers)), combined_report


def create_cv_group_plot_data(
    cv_data: pd.DataFrame, 
    cv_group_palettes: Dict[str, str], 
    id_col: Union[str, List[str]] = "Protein"
    ) -> pd.DataFrame:
    """
    Transforms coefficient of variation (CV) data into a long-format DataFrame
    for plotting, categorizing CV values into groups based on a provided palette dictionary.

    Args:
        cv_data (pd.DataFrame): DataFrame with features as index (can be single or MultiIndex),
                    conditions as columns, and CV values as data.
        cv_group_palettes (Dict[str, str]): Dict where keys are string labels for CV groups and values are color codes.
        id_col (Union[str, List[str]]): Name(s) for the feature identifier column(s). If MultiIndex, provide list of names.

    Returns:
        pd.DataFrame: Long-format DataFrame with columns for feature ID(s), Condition, CV, and CVGroup.
    """
    # 1. Dynamically determine bins and labels from the palette
    labels = list(cv_group_palettes.keys())
    bin_edges = [int(re.search(r'\d+', key).group()) for key in labels[:-1]]
    bins = [-np.inf] + bin_edges + [np.inf]

    # 2. Prepare id_vars for melt
    if isinstance(cv_data.index, pd.MultiIndex):
        if isinstance(id_col, str):
            id_col = list(cv_data.index.names)
        elif isinstance(id_col, (list, tuple)):
            id_col = list(id_col)
        else:
            raise ValueError("id_col must be a string or a list of strings for MultiIndex.")
        plot_data = cv_data.reset_index().melt(
            id_vars=id_col,
            value_name="CV",
            var_name="Condition"
        )
    else:
        id_name = id_col if isinstance(id_col, str) else (cv_data.index.name or "Feature")
        plot_data = cv_data.reset_index().melt(
            id_vars=[id_name],
            value_name="CV",
            var_name="Condition"
        )

    # 3. Create the CVGroup column by binning the CV values
    plot_data["CVGroup"] = pd.cut(
        x=plot_data["CV"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False
    )

    return plot_data

# ======================================================================================
# Performance Metrics and Calculations
# ======================================================================================

def generate_thresholds(
        base: float, 
        start_exp: int, 
        end_exp: int, 
        seq_start: float, 
        seq_end: float, 
        seq_step: float
    ) -> np.ndarray:
    """
        Generate a sorted array of unique values from the outer product 
        of a base raised to a range of exponents and a sequence of values.

        Args:
            base: The base for the outer product.
            start_exp: The start exponent for the outer product.
            end_exp: The end exponent for the outer product.
            seq_start: The start value of the sequence.
            seq_end: The end value of the sequence.
            seq_step: The step value of the sequence.

        Returns:
            A sorted array of unique values.
    """
    # Outer product with floating-point base
    outer_product = np.outer(1, base**np.arange(start_exp, end_exp + 1))

    # Sequence from seq_start to seq_end with step seq_step
    sequence = np.arange(seq_start, seq_end + seq_step, seq_step)

    # Combine arrays, get unique values, and sort
    result = np.sort(np.unique(np.concatenate((outer_product.flatten(), sequence))))

    # Don't keep values below 0 or above 1
    result = result[(result >= 0) & (result <= 1)]

    return result

def calculate_f1_score(
        precision: float,
        recall: float
    ) -> float:
    """
        Calculate the F1 Score using precision and recall values.

        Args:
            precision (float): The precision value.
            recall (float): The recall value.

        Returns:
            float: The F1 Score.
    """
    # Validations
    if (precision + recall) == 0: return 0

    # Calculate the F1 Score
    f1 = 2 * ((precision * recall) / (precision + recall))

    return f1

def calculate_metrics(
        true_labels: pd.Series,
        pred_labels: pd.Series,        
        verbose: bool = False,
        return_metrics: bool = False
    ):
    """
        Calculate the True Positive Rate (TPR), False Positive Rate (FPR), 
        False Discovery Rate (FDR), Accuracy (ACC), Matthews Correlation Coefficient (MCC),
        and F1 Score based on a given threshold.

        Args:
            true_labels (pd.Series): The true labels.
            pred_labels (pd.Series): The predicted labels.
            verbose (bool): If True, print the metrics.
            return_metrics (bool): If True, return the metrics as a dictionary.

        Returns:
            dict: A dictionary containing the metrics.

    """
    
    # Calculate the confusion matrix
    conf_matrix = pd.crosstab(
        index=true_labels,      # True labels
        columns=pred_labels     # Predicted labels
    )
    # Reindex to ensure both True and False are present for both columns and rows
    conf_matrix = conf_matrix.reindex(
        index=[True, False], columns=[True, False], fill_value=0
    )
    
    # Extract the values from the confusion matrix
    TP = conf_matrix.loc[True, True]
    FP = conf_matrix.loc[False, True]
    TN = conf_matrix.loc[False, False]
    FN = conf_matrix.loc[True, False]   

    # Calculate metrics
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    FDR = FP / (FP + TP) if (FP + TP) != 0 else 0
    # MCC = calculate_mcc(TP, TN, FP, FN)
    MCC = matthews_corrcoef(true_labels, pred_labels)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1 = calculate_f1_score(precision, recall)

    if verbose:
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print(f"Metrics:\n==========")
        for metric, value in zip(
            ["TPR", "FPR", "FDR", "MCC", "F1", "Precision", "Recall"],
            [TPR, FPR, FDR, MCC, F1, precision, recall]
        ):
            print(f" {metric}: {value:.2f}")

    if return_metrics:
        return {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            
            "TPR": TPR,
            "FPR": FPR,
            "FDR": FDR,
            "MCC": MCC,

            "Precision": precision,
            "Recall": recall,
            "F1": F1
        }

def create_metric_data(
        data: pd.DataFrame, 
        pvalue_thresholds: list,
        label_col: str ="perturbed_peptide", 
        pvalue_col: str ="adj.pval", 
    ):
    """
        Create a DataFrame containing performance metrics for various p-value and score thresholds.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.
            pvalue_thresholds (iterable): The p-value thresholds to evaluate.
            label_col (str): The column with the true labels.
            pvalue_col (str): The column with the threshold values (p-values).

        Returns:
            pd.DataFrame: A DataFrame containing the metrics for each threshold combination.
    """

    # Initialize a list to store metric dictionaries
    metrics_data = []

    # Iterate over p-value and score thresholds
    for pthr in pvalue_thresholds:
        true_labels = data[label_col]
        pred_labels = data[pvalue_col] <= pthr
        # Calculate metrics for the current threshold combination
        metrics = calculate_metrics(
            true_labels=true_labels, pred_labels=pred_labels, 
            verbose=False, return_metrics=True
        )

        # Add threshold values to the metrics dictionary
        metrics["threshold"] = pthr

        # Append the metrics dictionary to the list
        metrics_data.append(metrics)

    # Create a DataFrame from the list of metric dictionaries
    return pd.DataFrame(metrics_data)

def complete_curve_data(df, curve_type, x_col, y_col):
    """
    Ensures data for ROC or PR curves is complete by adding boundary points.

    This version intelligently fills other columns for new points using the
    first row as a template, ensuring identifiers like 'method' and
    'perturbation' are preserved. Other metrics are set to NaN.

    Args:
        df (pd.DataFrame): The input data for the curve.
        curve_type (str): The type of curve, either 'ROC' or 'PR'.
        x_col (str): The name of the column for the x-axis (e.g., 'FPR').
        y_col (str): The name of the column for the y-axis (e.g., 'TPR').

    Returns:
        pd.DataFrame: The completed DataFrame with necessary boundary points.
    """
    if df.empty:
        return df

    if curve_type == 'ROC':
        required_points = [(0, 0), (1, 1)]
    elif curve_type == 'PR':
        required_points = [(0, 1), (1, 0)]
    else:
        raise ValueError("curve_type must be either 'ROC' or 'PR'")

    # Use a list to collect dataframes to concat at the end for efficiency
    dfs_to_concat = [df]
    
    # Get the first row as a template. This preserves all other column values.
    template_row_df = df.iloc[[0]].copy()

    for x, y in required_points:
        # Check if the point exists
        if not ((df[x_col] == x) & (df[y_col] == y)).any():
            # Create a new row from the template
            new_row = template_row_df.copy()
            
            # Set the x and y values for the new point
            new_row[x_col] = x
            new_row[y_col] = y
            
            # Set other non-identifier metrics to NaN as they are not applicable
            for col in new_row.columns:
                if col not in df.columns.tolist() or col in [x_col, y_col]:
                    continue
                # Preserve identifiers by checking if the column has a single unique value
                if df[col].nunique() == 1:
                    continue
                new_row[col] = np.nan
            
            dfs_to_concat.append(new_row)

    # Concatenate the original DataFrame with any new rows
    completed_df = pd.concat(dfs_to_concat, ignore_index=True)

    # Sort the final DataFrame by the x-axis column
    return completed_df.sort_values(by=x_col).reset_index(drop=True)

def grouping_performance_proteoforge(
        data: pd.DataFrame,
        thresholds: list,
        pvalue_col: str = 'proteoform_score_pval',
        protein_col: str = 'protein_id',
        cluster_col: str = 'cluster',
        perturbation_col: str = 'pertPFG',
    ):
        metrics_data = []
        for thr in thresholds:
            tmp = data.copy()
            # Determine the significant peptides
            tmp['isSignificant'] = tmp[pvalue_col] < thr
            # Create protein-level aggregated data
            grouping_data = tmp.groupby(protein_col).agg({
                perturbation_col: 'nunique',  # Get the pertPFG value for this protein
                'isSignificant': 'any'  # Check if any peptide was significant
            }).reset_index()
            
            grp = tmp.groupby([protein_col, cluster_col])
            cluster_stats = grp['isSignificant'].agg(['any', 'sum']).reset_index()
            cluster_stats.columns = [protein_col, cluster_col, 'has_significant', 'peptide_count']

            is_single_sig = cluster_stats['has_significant'] & (cluster_stats['peptide_count'] == 1)
            is_multi_sig = cluster_stats['has_significant'] & (cluster_stats['peptide_count'] > 1)

            cluster_stats['dPF'] = 0
            cluster_stats.loc[is_single_sig, 'dPF'] = -1
            cluster_stats.loc[is_multi_sig, 'dPF'] = 1
            # Derive per-protein summary: mark protein as having proteoforms if any cluster has dPF == 1.
            # Encode True -> 2, False -> 0 as requested.
            protein_level = cluster_stats.groupby(protein_col).agg(
                pos_clusters=('dPF', lambda x: (x == 1).sum()),
                neg_clusters=('dPF', lambda x: (x == -1).sum())
            )
            protein_level['has_proteoform'] = protein_level['pos_clusters'] > 0
            # Encode: True -> 2, False -> 0
            protein_level['dPF'] = (protein_level['has_proteoform'].astype(int) * 2)
            protein_level = protein_level[['dPF']]
            protein_level = protein_level.reset_index()  # bring 'Protein' back as column
            grouping_data = grouping_data.merge(protein_level, on=protein_col, how='left')
            grouping_data['predPFG'] = grouping_data['dPF'].fillna(0).astype(int)
            grouping_data.drop(columns=['dPF'], inplace=True)

            # True labels: pertPFG > 0 (protein has proteoforms)
            # Note: pertPFG == -1 means no proteoforms, pertPFG >= 1 means proteoforms exist
            y_true_protein = (grouping_data[perturbation_col] > 1).astype(bool)
            # Predicted labels: dPF > 0 (detected proteoforms)
            y_pred_protein = (grouping_data['predPFG'] > 1).astype(bool)
            metrics = calculate_metrics(
                true_labels=y_true_protein,
                pred_labels=y_pred_protein,
                verbose=False, return_metrics=True
            )
            metrics['threshold'] = thr
            metrics_data.append(metrics)    
        metrics_data = pd.DataFrame(metrics_data)
        return metrics_data

def grouping_performance_copf(
        data: pd.DataFrame,
        thresholds: list,
        pvalue_col: str = 'proteoform_score_pval',
        protein_col: str = 'protein_id',
        cluster_col: str = 'cluster',
        perturbation_col: str = 'pertPFG',
    ):
    metrics_data = []
    tmp = data.copy()
    for thr in thresholds:
        tmp['isSignificant'] = tmp[pvalue_col] < thr
        # Create protein-level aggregated data
        grouping_data = tmp.groupby(protein_col).agg({
            perturbation_col: 'nunique',  # Get the pertPFG value for this protein
            cluster_col: 'nunique',  # Get the number of unique clusters (proteoforms) detected
            'isSignificant': 'any'  # Check if any peptide was significant
        }).reset_index()
        # True labels: pertPFG > 0 (protein has proteoforms)
        # Note: pertPFG == -1 means no proteoforms, pertPFG >= 1 means proteoforms exist
        y_true_protein = (grouping_data[perturbation_col] > 1).astype(bool)
        # Predicted labels: isSignificant and cluster > 1 (detected proteoforms)
        y_pred_protein = (grouping_data['isSignificant']) & (grouping_data[cluster_col] > 1)
        metrics = calculate_metrics(
            true_labels=y_true_protein,
            pred_labels=y_pred_protein,
            verbose=False, return_metrics=True
        )
        metrics['threshold'] = thr
        metrics_data.append(metrics)

    metrics_data = pd.DataFrame(metrics_data)
    return metrics_data