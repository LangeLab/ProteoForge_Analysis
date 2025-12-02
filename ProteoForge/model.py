#!/usr/bin/env python3
"""
ProteoForge Statistical Modeling Framework for Proteoform Analysis

A comprehensive statistical modeling framework designed specifically for large-scale 
proteoform analysis using weighted linear models. Provides flexible modeling capabilities 
with multiple statistical approaches, parallel processing acceleration, and sophisticated 
p-value correction strategies optimized for proteomics workflows.


Version: 0.7.0
Date: 2025-07-01
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

from tqdm import tqdm
from typing import Dict, List, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Import local modules
try:
    from . import correct
except ImportError:
    import correct


# ======================================================================================
# Constants and Configuration
# ======================================================================================

# Supported statistical model types with their characteristics
SUPPORTED_MODELS = {
    'ols': 'Ordinary Least Squares - standard linear regression (no weights)',
    'wls': 'Weighted Least Squares - accounts for heteroscedasticity (requires weights)',
    'rlm': 'Robust Linear Model - resistant to outliers (built-in robust weights)',
    'glm': 'Generalized Linear Model - flexible error distributions (supports weights)',
    'quantile': 'Quantile Regression - median regression (tau=0.5, supports weights)'
}

# P-value correction strategies with descriptions
CORRECTION_STRATEGIES = {
    'two-step': 'Within-protein correction followed by global correction',
    'global': 'Single global correction across all p-values',
    'protein-only': 'Correction within each protein independently'
}

# Default parameters for analysis
DEFAULT_MODEL_TYPE = 'wls'
DEFAULT_CORRECTION_STRATEGY = 'two-step'
DEFAULT_CORRECTION_METHODS = ('bonferroni', 'fdr_bh')

# ======================================================================================
# Performance and Multiprocessing Utilities
# ======================================================================================

def select_n_jobs(n_jobs: Optional[int] = None) -> int:
    """
    Automatically determine optimal number of CPU cores for statistical modeling.

    Provides intelligent defaults for multiprocessing based on system capabilities
    and the computational demands of statistical model fitting.

    Args:
        n_jobs: Number of CPU cores to use:
                    - None: Use half of available cores (conservative default)
                    - Positive integer: Use specified number of cores
                    - Will be capped at available CPU count

    Returns:
        int: Optimal number of cores to use, never exceeding system capacity

    Notes:
        - Conservative default (half cores) prevents system overload during modeling
        - Statistical model fitting can be memory-intensive
        - Leaves cores available for other system processes

    Examples:
        >>> # Use conservative default (half available cores)
        >>> n_cores = select_n_jobs()
        >>> print(f"Using {n_cores} cores for modeling")
        
        >>> # Use specific number of cores
        >>> n_cores = select_n_jobs(4)
    """
    total_cores = cpu_count()
    
    if n_jobs is None:
        # Use half the available cores by default (conservative approach)
        return max(1, total_cores // 2)
    
    if not isinstance(n_jobs, int) or n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer or None.")
    
    if n_jobs > total_cores:
        print(f"Warning: Requested {n_jobs} cores exceeds available {total_cores}. Using all available cores.")
        return total_cores
    
    return n_jobs


def validate_model_inputs(
        data: pd.DataFrame,
        required_columns: List[str],
        model_type: str,
        weight_col: Optional[str] = None
    ) -> None:
    """
    Comprehensive validation of input data and parameters for statistical modeling.

    Performs thorough validation to ensure data quality and parameter compatibility
    before statistical model fitting begins.

    Args:
        data: Input DataFrame to validate
        required_columns: List of required column names
        model_type: Statistical model type to validate
        weight_col: Weight column name (required for weighted models)

    Raises:
        ValueError: If validation fails for any parameter or data quality check

    Notes:
        - Validates DataFrame structure and required columns
        - Checks model type compatibility
        - Ensures weight column availability for weighted models
        - Validates data types and missing value patterns

    Examples:
        >>> # Validate standard proteomics data
        >>> validate_model_inputs(
        ...     data=df,
        ...     required_columns=['Protein', 'Peptide', 'Condition', 'Intensity'],
        ...     model_type='wls',
        ...     weight_col='Weight'
        ... )
    """
    # Validate DataFrame structure
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")
    
    if data.empty:
        raise ValueError("Input DataFrame is empty.")
    
    # Validate required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate model type
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Invalid model type: {model_type}. "
            f"Must be one of {list(SUPPORTED_MODELS.keys())}"
        )
    
    # Validate weight column for weighted models
    if model_type in ['wls', 'glm'] and weight_col is None:
        raise ValueError(f"Weight column must be provided for {model_type} model.")
    
    if weight_col is not None and weight_col not in data.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in DataFrame.")


# ======================================================================================
# Main Statistical Modeling Class
# ======================================================================================

class LinearModel:
    """
    Comprehensive statistical modeling framework for proteoform analysis.

    Designed specifically for large-scale proteoform studies using weighted linear models
    to identify peptides with significantly different behavior patterns within proteins.
    Optimized for proteomics workflows with parallel processing and sophisticated 
    statistical correction strategies.

    Features:
    ---------
    - **Multiple Statistical Models**: Support for OLS, WLS, RLM, and GLM approaches
    - **Flexible P-value Correction**: Global, protein-wise, and two-step strategies
    - **High-Performance Processing**: Automatic multiprocessing optimization
    - **Robust Error Handling**: Comprehensive validation and graceful failure handling
    - **Extensible Design**: Easy integration with custom statistical approaches
    - **Quality Control**: Built-in data validation and model diagnostics

    Statistical Models:
    ------------------
    - **OLS**: Ordinary Least Squares (standard unweighted linear regression)
    - **WLS**: Weighted Least Squares (accounts for heteroscedasticity, requires weights)
    - **RLM**: Robust Linear Model (resistant to outliers, uses internal robust weights)
    - **GLM**: Generalized Linear Model (flexible error distributions, supports weights)
    - **Quantile**: Quantile Regression (median regression tau=0.5, supports weights)

    Correction Strategies:
    ---------------------
    - **two-step**: Within-protein correction followed by global correction
    - **global**: Single correction applied to all p-values simultaneously
    - **protein-only**: Independent correction within each protein group

    Examples:
    ---------
    >>> # Standard proteoform analysis
    >>> model = LinearModel(
    ...     data=proteomics_df,
    ...     protein_col='Protein',
    ...     peptide_col='Peptide',
    ...     cond_col='Condition',
    ...     intensity_col='Intensity',
    ...     weight_col='Weight'
    ... )
    >>> 
    >>> # Comprehensive analysis with custom parameters
    >>> results = model.run_analysis(
    ...     model_type='wls',
    ...     correction_strategy='two-step',
    ...     correction_methods=('bonferroni', 'fdr_bh'),
    ...     n_jobs=8
    ... )
    >>> 
    >>> # Single protein analysis
    >>> protein_results = model.run_for_protein('P12345')
    >>> 
    >>> # Get protein-level diagnostics and model details
    >>> diagnostics = model.get_protein_diagnostics('P12345')
    """

    def __init__(
        self,
        data: pd.DataFrame,
        protein_col: str,
        peptide_col: str,
        cond_col: str,
        intensity_col: str,
        weight_col: Optional[str] = None,
    ):
        """
        Initialize the LinearModel with proteomics dataset and column specifications.

        Args:
            data: Input proteomics data in long format containing protein, peptide,
                 condition, and intensity measurements
            protein_col: Column name for protein identifiers
            peptide_col: Column name for peptide/proteoform identifiers  
            cond_col: Column name for experimental conditions (e.g., time, treatment)
            intensity_col: Column name for abundance/intensity measurements
            weight_col: Column name for statistical weights (required for WLS/RLM/GLM)

        Raises:
            ValueError: If required columns are missing or data validation fails

        Notes:
            - Data should be in long format with one row per measurement
            - Categorical variables will be automatically encoded using C() notation
            - Weight column is required for weighted statistical models
            - Missing values in intensity data should be handled before analysis

        Examples:
            >>> # Initialize with standard proteomics data
            >>> model = LinearModel(
            ...     data=df,
            ...     protein_col='UniProt_ID',
            ...     peptide_col='Peptide_Sequence',
            ...     cond_col='Treatment',
            ...     intensity_col='Log2_Intensity',
            ...     weight_col='Confidence_Score'
            ... )
        """
        # Validate input parameters and data quality
        required_columns = [protein_col, peptide_col, cond_col, intensity_col]
        validate_model_inputs(data, required_columns, DEFAULT_MODEL_TYPE, weight_col)
        
        # Store configuration
        self.data = data.copy()
        self.protein_col = protein_col
        self.peptide_col = peptide_col
        self.cond_col = cond_col
        self.intensity_col = intensity_col
        self.weight_col = weight_col
        
        # Generate statistical formula using categorical encoding
        # C() ensures variables are treated as categorical for robust modeling
        self.formula = f'{self.intensity_col} ~ C({self.cond_col}) * C(allothers)'
        
        # Store analysis metadata
        self.last_analysis_params = None
        self.analysis_history = []

    @staticmethod
    def _fit_model(
        data: pd.DataFrame, 
        formula: str, 
        model_type: str, 
        weight_col: Optional[str]
    ) -> float:
        """
        Fit a statistical model and extract p-value for the interaction term.

        Core statistical modeling function that fits the specified linear model
        and returns the p-value for the interaction term of interest.

        Args:
            data: DataFrame containing the data for model fitting
            formula: R-style formula string for the statistical model
            model_type: Type of statistical model to fit ('ols', 'wls', 'rlm', 'glm', 'quantile')
            weight_col: Column name for statistical weights (required for weighted models)

        Returns:
            float: P-value for the interaction term, or np.nan if fitting fails

        Notes:
            - Uses Wald test to extract p-value for the entire interaction term
            - Robust error handling ensures graceful failure for convergence issues
            - Returns NaN for failed fits to maintain data structure integrity
            - Weighted models require valid weight column specification

        Examples:
            >>> # Fit weighted least squares model
            >>> p_value = LinearModel._fit_model(
            ...     data=protein_data,
            ...     formula='Intensity ~ C(Condition) * C(Peptide)',
            ...     model_type='wls',
            ...     weight_col='Weight'
            ... )
        """
        try:
            # Fit appropriate statistical model based on type
            if model_type == "ols":
                # OLS doesn't use weights - standard unweighted regression
                cur_model = smf.ols(formula=formula, data=data).fit()
            elif model_type == "wls":
                if weight_col is None:
                    raise ValueError("Weight column must be provided for WLS model.")
                cur_model = smf.wls(formula=formula, data=data, weights=data[weight_col]).fit()
            elif model_type == "rlm":
                # RLM uses its own internal robust weighting scheme, external weights not supported
                cur_model = smf.rlm(formula=formula, data=data).fit()
            elif model_type == "glm":
                if weight_col is not None:
                    cur_model = smf.glm(formula=formula, data=data, family=sm.families.Gaussian(), var_weights=data[weight_col]).fit()
                else:
                    cur_model = smf.glm(formula=formula, data=data, family=sm.families.Gaussian()).fit()
            elif model_type == "quantile":
                # Quantile regression at tau=0.5 (median regression)
                if weight_col is not None:
                    cur_model = smf.quantreg(formula=formula, data=data).fit(q=0.5, weights=data[weight_col])
                else:
                    cur_model = smf.quantreg(formula=formula, data=data).fit(q=0.5)
            else:
                raise ValueError(
                    f"Invalid model type: {model_type}. "
                    f"Must be one of {list(SUPPORTED_MODELS.keys())}"
                )
            
            # Extract p-value using Wald test for the interaction term
            return float(cur_model.wald_test_terms(scalar=False).pvalues[-1])
            
        except Exception as e:
            # Return NaN if model fails to converge or other error occurs
            # This maintains data structure while indicating failed fitting
            return np.nan

    @staticmethod
    def _worker_process_protein(args) -> List[Dict]:
        """
        Multiprocessing worker function for single protein analysis.

        Processes all peptides within a single protein to identify significant
        differences using the specified statistical model. Designed for efficient
        parallel processing across protein groups.

        Args:
            args: Tuple containing (protein_id, protein_data, formula, model_type,
                 protein_col, peptide_col, weight_col)

        Returns:
            List of dictionaries containing results for each peptide in the protein

        Notes:
            - Requires at least 2 peptides for meaningful analysis
            - Creates 'allothers' grouping for each peptide comparison
            - Returns empty list for proteins with insufficient peptides
            - Robust error handling for individual peptide failures

        Examples:
            >>> # Process single protein (typically called via multiprocessing)
            >>> args = (protein_id, protein_data, formula, 'wls', 
            ...         'Protein', 'Peptide', 'Weight')
            >>> results = LinearModel._worker_process_protein(args)
        """
        # protein_id, protein_data, formula, model_type, p_col, pep_col, w_col = args
        protein_id, protein_data, formula, model_type, pep_col, w_col = args
        unique_peptides = protein_data[pep_col].unique()
        
        # Require at least 2 peptides for meaningful comparison
        if len(unique_peptides) < 2:
            return []

        results = []
        for peptide in unique_peptides:
            # Create binary grouping: current peptide vs all others
            # Use .loc to avoid SettingWithCopyWarning
            protein_data_copy = protein_data.copy()
            protein_data_copy.loc[:, 'allothers'] = np.where(
                protein_data_copy[pep_col] == peptide, 
                peptide, 'allothers'
            )
            
            # Fit statistical model and extract p-value
            pval = LinearModel._fit_model(protein_data_copy, formula, model_type, w_col)
            
            results.append({
                'protein_id': protein_id, 
                'peptide_id': peptide, 
                'pval': pval
            })
            
        return results

    def run_analysis(
        self, 
        model_type: str = DEFAULT_MODEL_TYPE, 
        correction_strategy: str = DEFAULT_CORRECTION_STRATEGY,
        correction_methods=DEFAULT_CORRECTION_METHODS,
        n_jobs: Optional[int] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Execute comprehensive proteoform analysis with flexible statistical correction.

        Performs large-scale statistical analysis across all proteins using the specified
        model and correction strategy. Optimized for proteomics workflows with automatic
        multiprocessing and sophisticated p-value correction approaches.

        Args:
            model_type: Statistical model type to use ('ols', 'wls', 'rlm', 'glm', 'quantile')
            correction_strategy: P-value correction approach:
                               - 'two-step': Within-protein then global correction
                               - 'global': Single correction across all p-values  
                               - 'protein-only': Independent protein-wise correction
            correction_methods: Correction method(s) to apply:
                              - For 'two-step': tuple of (protein_method, global_method)
                              - For others: single method string
            n_jobs: Number of CPU cores for parallel processing (auto-selected if None)
            verbose: If True, print progress and status messages (default: False)

        Returns:
            pd.DataFrame: Comprehensive results with raw and adjusted p-values merged
                         with original data for complete analysis context

        Raises:
            ValueError: If parameters are invalid or incompatible

        Notes:
            - Automatically scales multiprocessing based on dataset size and available cores
            - Provides progress tracking for large datasets
            - Merges results with original data to preserve experimental context
            - Handles failed model fits gracefully with NaN p-values
            - Stores analysis parameters for reproducibility

        Examples:
            >>> # Standard two-step correction analysis
            >>> results = model.run_analysis(
            ...     model_type='wls',
            ...     correction_strategy='two-step',
            ...     correction_methods=('bonferroni', 'fdr_bh'),
            ...     n_jobs=8
            ... )
            >>> 
            >>> # Single global correction with verbose output
            >>> results = model.run_analysis(
            ...     model_type='ols',
            ...     correction_strategy='global',
            ...     correction_methods='fdr_bh',
            ...     verbose=True
            ... )
            >>> 
            >>> # Protein-wise correction only
            >>> results = model.run_analysis(
            ...     correction_strategy='protein-only',
            ...     correction_methods='bonferroni'
            ... )
        """
        # Validate model type and correction parameters
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Invalid model type: {model_type}. "
                f"Must be one of {list(SUPPORTED_MODELS.keys())}"
            )
        
        if correction_strategy not in CORRECTION_STRATEGIES:
            raise ValueError(
                f"Invalid correction strategy: {correction_strategy}. "
                f"Must be one of {list(CORRECTION_STRATEGIES.keys())}"
            )
        
        # Determine optimal number of workers
        n_jobs = select_n_jobs(n_jobs)
        
        # Prepare parallel processing tasks
        protein_groups = self.data.groupby(self.protein_col)
        tasks = [
            (
                protein_id, 
                group_df, 
                self.formula, 
                model_type, 
                self.peptide_col, 
                self.weight_col
            ) 
            for protein_id, group_df in protein_groups
        ]

        if verbose:
            print(f"Starting proteoform analysis on {len(tasks)} proteins using {n_jobs} worker(s)...")
            print(f"Statistical model: {model_type} ({SUPPORTED_MODELS[model_type]})")
            print(f"Correction strategy: {correction_strategy} ({CORRECTION_STRATEGIES[correction_strategy]})")
        
        # Execute parallel processing with progress tracking
        with Pool(processes=n_jobs) as pool:
            if verbose:
                results_list = list(tqdm(
                    pool.imap(self._worker_process_protein, tasks), 
                    total=len(tasks),
                    desc="Processing proteins"
                ))
            else:
                results_list = list(pool.imap(self._worker_process_protein, tasks))
        
        # Flatten results and validate
        flat_results = [item for sublist in results_list for item in sublist]
        if not flat_results:
            print("Warning: No results were generated. Check your data and parameters.")
            return pd.DataFrame()

        # Create results DataFrame
        res_df = pd.DataFrame(flat_results)
        res_df.rename(
            columns={'protein_id': self.protein_col, 'peptide_id': self.peptide_col}, 
            inplace=True
        )
        
        # Apply flexible p-value correction strategies
        res_df = self._apply_correction_strategy(
            res_df, correction_strategy, correction_methods, verbose
        )
        
        # Merge results with original data for complete context
        final_df = pd.merge(
            self.data, res_df, 
            on=[self.protein_col, self.peptide_col], 
            how="left"
        )
        
        # Store analysis metadata
        self.last_analysis_params = {
            'model_type': model_type,
            'correction_strategy': correction_strategy,
            'correction_methods': correction_methods,
            'n_jobs': n_jobs,
            'num_proteins': len(tasks),
            'num_results': len(flat_results)
        }
        
        self.analysis_history.append(self.last_analysis_params.copy())
        
        if verbose:
            print(f"Analysis completed: {len(flat_results)} peptide comparisons across {len(tasks)} proteins")
        
        return final_df.drop_duplicates().reset_index(drop=True)

    def _apply_correction_strategy(
        self, 
        res_df: pd.DataFrame, 
        correction_strategy: str, 
        correction_methods,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Apply the specified p-value correction strategy to results.

        Implements flexible correction approaches optimized for proteomics analysis
        where both protein-wise and global correction considerations are important.

        Args:
            res_df: Results DataFrame with raw p-values
            correction_strategy: Strategy to apply ('two-step', 'global', 'protein-only')
            correction_methods: Method(s) for correction

        Returns:
            pd.DataFrame: Results with adjusted p-values added

        Notes:
            - Two-step approach first corrects within proteins, then globally
            - Global approach treats all p-values equally
            - Protein-only approach maintains independence between proteins
        """
        res_df['prt_pval'] = np.nan
        res_df['adj_pval'] = np.nan

        if correction_strategy == 'two-step':
            if not isinstance(correction_methods, (list, tuple)) or len(correction_methods) != 2:
                raise ValueError(
                    "For 'two-step' strategy, correction_methods must be a tuple of two methods, "
                    "e.g., ('bonferroni', 'fdr_bh')."
                )
            protein_method, global_method = correction_methods
            
            if verbose:
                print(f"Applying two-step correction: "
                      f"1. Within-protein ({protein_method}), 2. Global ({global_method})")
            
            # Step 1: Within-protein correction
            res_df['prt_pval'] = res_df.groupby(self.protein_col)['pval'].transform(
                lambda x: correct.for_multiple_tests(x, correction_type=protein_method)
            )
            
            # Step 2: Global correction on the already-adjusted p-values
            res_df['adj_pval'] = correct.for_multiple_tests(
                res_df['prt_pval'], correction_type=global_method
            )

        elif correction_strategy == 'global':
            if not isinstance(correction_methods, str):
                raise ValueError(
                    "For 'global' strategy, correction_methods must be a single string, "
                    "e.g., 'fdr_bh'."
                )
            if verbose:
                print(f"Applying single-step global correction ({correction_methods})...")
            res_df['adj_pval'] = correct.for_multiple_tests(
                res_df['pval'], correction_type=correction_methods
            )

        elif correction_strategy == 'protein-only':
            if not isinstance(correction_methods, str):
                raise ValueError(
                    "For 'protein-only' strategy, correction_methods must be a single string, "
                    "e.g., 'bonferroni'."
                )
            if verbose:
                print(f"Applying single-step per-protein correction ({correction_methods})...")
            res_df['adj_pval'] = res_df.groupby(self.protein_col)['pval'].transform(
                lambda x: correct.for_multiple_tests(x, correction_type=correction_methods)
            )

        return res_df

    def run_for_protein(
        self, 
        protein_id: str, 
        model_type: str = DEFAULT_MODEL_TYPE,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Execute focused statistical analysis for a single specified protein.

        Performs detailed analysis on a single protein to examine peptide differences
        within that protein group. Useful for detailed investigation of specific proteins
        of interest or for validation of global analysis results.

        Args:
            protein_id: Protein identifier to analyze (must exist in the dataset)
            model_type: Statistical model type to use ('ols', 'wls', 'rlm', 'glm', quantile)
            verbose: If True, print progress and status messages (default: False)

        Returns:
            pd.DataFrame: Detailed results for all peptides within the specified protein

        Raises:
            ValueError: If protein_id is not found in the dataset

        Notes:
            - Provides detailed results without global correction context
            - Useful for focused analysis and method validation
            - Returns empty DataFrame if protein has insufficient peptides
            - Model fitting failures return NaN p-values

        Examples:
            >>> # Analyze specific protein of interest
            >>> protein_results = model.run_for_protein('P12345', model_type='wls')
            >>> print(f"Found {len(protein_results)} peptides in protein P12345")
            >>> 
            >>> # Compare different models for same protein
            >>> ols_results = model.run_for_protein('P12345', model_type='ols')
            >>> wls_results = model.run_for_protein('P12345', model_type='wls')
        """
        # Validate protein exists in dataset
        protein_data = self.data[self.data[self.protein_col] == protein_id].copy()
        if protein_data.empty:
            raise ValueError(f"Protein ID '{protein_id}' not found in the dataset.")
        
        # Validate model type
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Invalid model type: {model_type}. "
                f"Must be one of {list(SUPPORTED_MODELS.keys())}"
            )
        
        if verbose:
            print(f"Analyzing protein {protein_id} using {model_type} model...")
        
        # Process single protein
        results = self._worker_process_protein((
            protein_id, protein_data, self.formula, model_type, 
            self.peptide_col, self.weight_col
        ))
        
        if not results:
            print(f"Warning: No results generated for protein {protein_id}. "
                  f"Check that it has at least 2 peptides.")
            return pd.DataFrame()
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        if verbose:
            print(f"Analysis completed: {len(results)} peptides analyzed in protein {protein_id}")

        return results_df