#!/usr/bin/env python3
"""
High-Performance Integrated Distance Calculation and Clustering for Proteoform Analysis

A highly optimized framework that combines feature matrix generation, distance calculation,
and clustering in a single, efficient pipeline. Designed for proteomics workflows where
clustering is based on the Euclidean distance between peptide profiles across various
conditions, this script avoids intermediate data storage for maximum performance.

Version: 1.0.0
Date: 2025-09-11
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

import time
import warnings
from multiprocessing import Pool, cpu_count
from typing import Optional, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

# Import clustering functionality from the existing cluster module
try:
    from . import cluster
except ImportError:
    import cluster

# ======================================================================================
# Worker Function for Multiprocessing
# ======================================================================================

def _disluster_worker(args: Tuple) -> Tuple[str, Optional[Dict]]:
    """
    Perform distance matrix calculation and clustering for a single protein.
    
    This worker function encapsulates the entire process for one protein group:
    1. Generates a feature matrix of peptides vs. conditions.
    2. Directly calls the clustering module with the feature matrix.
    This avoids creating and storing an intermediate distance matrix.
    
    Args:
        args: Tuple containing processing parameters:
            - protein_name (str): Identifier for the current protein.
            - protein_df (pd.DataFrame): Data subset for this protein.
            - id_col (str): Column name for peptide identifiers.
            - cond_col (str): Column name for conditions.
            - quant_col (str): Column name for quantification values.
            - pdist_metric (str): The distance metric for `scipy.spatial.distance.pdist`.
            - clustering_params (dict): Parameters for the clustering algorithm.
            
    Returns:
        Tuple[str, Optional[Dict]]: Protein name and a results dictionary containing
        'peptides', 'labels', and 'n_clusters', or None if processing failed.
    """
    # Unpack arguments for clarity
    (protein_name, protein_df, id_col, cond_col, quant_col,
     pdist_metric, clustering_params) = args
    
    try:
        # Step 1: Create peptide profiles by averaging quant values
        peptide_profiles = protein_df.groupby([id_col, cond_col])[quant_col].median()

        # Step 2: Unstack to create the feature matrix (peptides x conditions)
        features = peptide_profiles.unstack(level=cond_col)
        
        # Fill any missing values that may result from unstacking
        features = features.fillna(0)
        
        n_peptides = features.shape[0]
        
        # Check if there are enough peptides to form a cluster
        if n_peptides < 2:
            warnings.warn(f"Protein '{protein_name}' has fewer than 2 unique peptides after processing, skipping.")
            return protein_name, None
        
        # If fewer peptides than the minimum cluster size, assign all to one cluster
        if n_peptides < clustering_params.get('min_clusters', 2):
            return protein_name, {
                'peptides': features.index.tolist(),
                'labels': np.ones(n_peptides, dtype=int),
                'n_clusters': 1
            }
            
        # Step 3: Perform clustering directly on the feature matrix
        try:
            cluster_labels = _cluster_feature_matrix(features.values, pdist_metric, clustering_params)
            
            return protein_name, {
                'peptides': features.index.tolist(),
                'labels': cluster_labels,
                'n_clusters': len(np.unique(cluster_labels))
            }
        
        except Exception as cluster_error:
            warnings.warn(f"Clustering failed for protein '{protein_name}': {cluster_error}")
            # As a fallback, assign all peptides to a single cluster
            return protein_name, {
                'peptides': features.index.tolist(),
                'labels': np.ones(n_peptides, dtype=int),
                'n_clusters': 1
            }

    except Exception as e:
        warnings.warn(f"Could not process protein '{protein_name}': {e}")
        return protein_name, None


def _cluster_feature_matrix(
    feature_matrix: np.ndarray,
    pdist_metric: str,
    clustering_params: Dict
) -> np.ndarray:
    """
    Perform clustering on a feature matrix using the ProteoformClustering class.
    
    This function correctly configures the clustering class to compute distances
    from the feature matrix internally, which is more efficient.
    
    Args:
        feature_matrix: A numpy array where rows are items and columns are features.
        pdist_metric: The distance metric to use (e.g., 'euclidean').
        clustering_params: A dictionary of parameters for clustering.
        
    Returns:
        An array of cluster labels.
    """
    cp = dict(clustering_params) if clustering_params is not None else {}
    n_items = feature_matrix.shape[0]

    # Sanitize min/max cluster counts to ensure they are valid
    cp_min = int(cp.get('min_clusters', 1))
    cp_max = cp.get('max_clusters', None)
    if cp_max is None: cp_max = n_items
    else: cp_max = int(cp_max)
    
    cp_min = max(1, cp_min)
    cp_max = max(cp_min, min(cp_max, n_items))

    # Instantiate the clustering class from the cluster module
    clusterer = cluster.ProteoformClustering(
        clustering_method=cp.get('clustering_method', 'hybrid_outlier_cut'),
        linkage_method=cp.get('linkage_method', 'complete'),
        distance_metric=pdist_metric,  # This metric will be used by pdist internally
        min_clusters=cp_min,
        max_clusters=cp_max,
        use_distance_matrix=False,  # Critical: tells the class to treat input as features
        verbose=False
    )
    
    # The fit_predict method will now compute pdist on the feature_matrix
    labels = clusterer.fit_predict(feature_matrix, method_params=cp)
    return labels

# ======================================================================================
# Main DisCluster Class
# ======================================================================================

class DisCluster:
    """
    High-performance integrated distance calculation and clustering framework.
    
    This class orchestrates the process of generating feature matrices from long-format
    data, calculating pairwise distances, and performing hierarchical clustering in a
    single, optimized pipeline accelerated by multiprocessing.
    
    Features:
    ---------
    - **Integrated Pipeline**: No intermediate data storage for high performance.
    - **Configurable Distance**: Supports all distance metrics from `scipy.spatial.distance.pdist`.
    - **Flexible Clustering**: Leverages all methods from the `cluster.py` module.
    - **Multiprocessing Ready**: Scales across all available CPU cores.
    - **Memory Efficient**: Processes data in chunks per protein group.

    Examples:
    ---------
    >>> # Initialize with default Euclidean distance
    >>> disluster = DisCluster(
    ...     protein_col='Protein',
    ...     peptide_col='Peptide', 
    ...     cond_col='Condition',
    ...     quant_col='Intensity'
    ... )
    >>> results_df = disluster.fit_transform(data)
    
    >>> # Use a different distance metric and custom clustering parameters
    >>> clustering_params = {'clustering_method': 'dynamic_cut', 'linkage_method': 'ward'}
    >>> disluster = DisCluster(
    ...     protein_col='Protein',
    ...     peptide_col='Peptide',
    ...     cond_col='Condition', 
    ...     quant_col='Abundance',
    ...     pdist_metric='cityblock',
    ...     clustering_params=clustering_params,
    ...     n_jobs=-1,
    ...     verbose=True
    ... )
    >>> results_df = disluster.fit_transform(data)
    """
    
    def __init__(
        self,
        protein_col: str,
        peptide_col: str,
        cond_col: str,
        quant_col: str,
        pdist_metric: str = 'euclidean',
        clustering_params: Optional[Dict] = None,
        n_jobs: int = -1,
        verbose: bool = False
    ):
        """
        Initialize the DisCluster framework.
        
        Args:
            protein_col: Column name for protein identifiers (grouping variable).
            peptide_col: Column name for peptide identifiers (items to be clustered).
            cond_col: Column name for conditions (features for distance calculation).
            quant_col: Column name for quantification values.
            pdist_metric: Distance metric for `pdist` (e.g., 'euclidean', 'cosine').
            clustering_params: Dictionary of parameters passed to the clustering module.
            n_jobs: Number of CPU cores for parallel processing (-1 for all).
            verbose: Enable detailed progress output.
        """
        self.protein_col = protein_col
        self.peptide_col = peptide_col
        self.cond_col = cond_col
        self.quant_col = quant_col
        self.pdist_metric = pdist_metric
        
        default_clustering = {
            'min_clusters': 1,
            'max_clusters': None,
            'clustering_method': 'hybrid_outlier_cut',
            'linkage_method': 'complete',
        }
        
        if clustering_params is None:
            self.clustering_params = default_clustering
        else:
            self.clustering_params = {**default_clustering, **clustering_params}
        
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self._last_computation_time = None
        self._last_protein_count = None

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform the integrated distance calculation and clustering analysis.
        
        Args:
            data: Input data in long format containing all required columns.
            
        Returns:
            A DataFrame with columns [protein_col, peptide_col, 'cluster_label'].
        """
        start_time = time.time() if self.verbose else None
        
        self._validate_data(data)
        
        if self.verbose:
            print("Starting DisCluster analysis...")
            print(f"Distance metric: {self.pdist_metric}")
            print(f"Dataset shape: {data.shape}")
        
        protein_groups = data.groupby(self.protein_col)
        n_proteins = len(protein_groups)
        
        if self.verbose:
            print(f"Found {n_proteins} proteins for processing.")
        
        tasks = [
            (
                protein_name, protein_df, self.peptide_col, self.cond_col,
                self.quant_col, self.pdist_metric, self.clustering_params
            )
            for protein_name, protein_df in protein_groups
        ]
        
        num_workers = cluster.select_n_jobs(self.n_jobs)
        
        if self.verbose:
            print(f"Using {num_workers} CPU cores for parallel processing.")
        
        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                results_list = pool.map(_disluster_worker, tasks)
        else:
            if self.verbose:
                print("Running sequentially (n_jobs=1).")
            results_list = [_disluster_worker(task) for task in tasks]
            
        result_rows = []
        successful_proteins = 0
        for protein_name, result in results_list:
            if result is not None:
                successful_proteins += 1
                for peptide, label in zip(result['peptides'], result['labels']):
                    result_rows.append({
                        self.protein_col: protein_name,
                        self.peptide_col: peptide,
                        'cluster_label': label
                    })
        
        if not result_rows:
            return pd.DataFrame(columns=[self.protein_col, self.peptide_col, 'cluster_label'])

        results_df = pd.DataFrame(result_rows)

        if self.verbose and start_time:
            elapsed = time.time() - start_time
            self._last_computation_time = elapsed
            self._last_protein_count = successful_proteins
            print(f"\nSuccessfully processed {successful_proteins}/{n_proteins} proteins.")
            print(f"Total computation time: {elapsed:.2f} seconds.")
            if successful_proteins > 0:
                print(f"Average time per protein: {elapsed/successful_proteins:.3f} seconds.")
        
        return results_df

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate the input DataFrame to ensure it has the required columns."""
        required_cols = [self.protein_col, self.peptide_col, self.cond_col, self.quant_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}.")
        if len(data) == 0:
            raise ValueError("Input DataFrame is empty.")
            
    def get_performance_stats(self) -> Dict[str, Union[float, int, None]]:
        """Get performance statistics from the last computation."""
        if self._last_computation_time is None:
            return {'computation_time': None, 'proteins_processed': None, 'avg_time_per_protein': None}
        
        avg_time = self._last_computation_time / self._last_protein_count if self._last_protein_count else 0
        
        return {
            'computation_time': self._last_computation_time,
            'proteins_processed': self._last_protein_count,
            'avg_time_per_protein': avg_time
        }

# ======================================================================================
# Convenience Function
# ======================================================================================

def distance_and_cluster(
    data: pd.DataFrame,
    protein_col: str,
    peptide_col: str,
    cond_col: str,
    quant_col: str,
    pdist_metric: str = 'euclidean',
    clustering_params: Optional[Dict] = None,
    n_jobs: int = -1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    A convenience function for a one-shot distance calculation and clustering analysis.
    
    Args:
        data: Input data in long format.
        protein_col: Protein identifier column name.
        peptide_col: Peptide identifier column name.
        cond_col: Condition column name.
        quant_col: Quantification column name.
        pdist_metric: Distance metric for pdist.
        clustering_params: Dictionary of clustering parameters.
        n_jobs: Number of CPU cores for parallel processing.
        verbose: Enable detailed output.
        
    Returns:
        A DataFrame with the clustering results.
    """
    disluster = DisCluster(
        protein_col=protein_col,
        peptide_col=peptide_col,
        cond_col=cond_col,
        quant_col=quant_col,
        pdist_metric=pdist_metric,
        clustering_params=clustering_params,
        n_jobs=n_jobs,
        verbose=verbose
    )
    return disluster.fit_transform(data)
