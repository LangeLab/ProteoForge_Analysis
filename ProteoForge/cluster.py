#!/usr/bin/env python3
"""
High-Performance Hierarchical Clustering for Proteoform Analysis

A comprehensive clustering framework designed specifically for proteoform correlation 
analysis with multiprocessing acceleration. Provides flexible clustering capabilities 
with multiple distance transformations, clustering algorithms, and quality assessment methods.

Version: 0.7.0
Date: 2025-07-01
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

import time
import warnings
from collections import Counter
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, List, Callable, Any, Tuple

import numpy as np
import pandas as pd

from sknetwork.hierarchy import cut_balanced
from dynamicTreeCut import cutreeHybrid
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_samples

# Suppress warnings from dynamicTreeCut
warnings.filterwarnings("ignore", category=UserWarning, module='dynamicTreeCut')

# ======================================================================================
# Performance and Multiprocessing Utilities
# ======================================================================================

def select_n_jobs(n_jobs: int) -> int:
    """
    Robustly determine the optimal number of parallel jobs for clustering operations.

    Automatically selects the best number of CPU cores to use based on system
    capabilities and user preferences, with intelligent defaults and safety checks.

    Args:
        n_jobs: Number of jobs to run in parallel:
               - None or 0: Use single-core (serial execution)
               - -1: Use all available CPU cores
               - -2: Use all cores except one
               - Positive integer: Use specified number of cores

    Returns:
        int: Optimal number of cores to use, never exceeding system capacity

    Notes:
        - Automatically caps at available CPU count to prevent oversubscription
        - Provides informative output about core usage decisions
        - Handles edge cases gracefully (negative values, excessive requests)

    Examples:
        >>> # Use all available cores
        >>> n_cores = select_n_jobs(-1)
        >>> print(f"Using {n_cores} cores")
        
        >>> # Use all but one core (recommended for interactive work)
        >>> n_cores = select_n_jobs(-2)
        
        >>> # Use specific number of cores
        >>> n_cores = select_n_jobs(4)
    """
    total_cores = cpu_count()
    
    if n_jobs is None or n_jobs == 0:
        return 1
        
    if n_jobs < 0:
        jobs = max(1, total_cores + 1 + n_jobs)
        if jobs < total_cores:
            print(f"Using {jobs} out of {total_cores} available CPU cores.")
        else:
            print(f"Using all {total_cores} CPU cores.")
        return jobs
        
    if n_jobs > total_cores:
        print(f"Warning: n_jobs ({n_jobs}) exceeds available cores ({total_cores}). Using all cores instead.")
        return total_cores
        
    return n_jobs

def pdist_to_squareform(distances: np.ndarray) -> np.ndarray:
    """
    Convert condensed distance matrix from pdist to square form.

    Utility function for converting the condensed distance matrix format
    returned by scipy.spatial.distance.pdist to a full square matrix format
    required by some clustering algorithms.

    Args:
        distances: Condensed distance matrix from scipy.spatial.distance.pdist

    Returns:
        np.ndarray: Full square distance matrix

    Notes:
        - Input should be the output of scipy.spatial.distance.pdist
        - Output is symmetric with zeros on the diagonal
        - Useful for clustering methods requiring full distance matrices

    Examples:
        >>> from scipy.spatial.distance import pdist
        >>> import numpy as np
        >>> 
        >>> # Create sample data and compute condensed distances
        >>> data = np.random.rand(5, 3)
        >>> condensed_dist = pdist(data)
        >>> 
        >>> # Convert to square form
        >>> square_dist = pdist_to_squareform(condensed_dist)
        >>> print(f"Square matrix shape: {square_dist.shape}")  # (5, 5)
    """
    return squareform(distances)

# ==============================================================================
# Main Clustering Class
# ==============================================================================

class ProteoformClustering:
    """
    Comprehensive hierarchical clustering framework for proteoform correlation analysis.

    Designed specifically for clustering proteoform correlation matrices with multiple
    distance transformations, clustering algorithms, and quality assessment methods.
    Optimized for proteomics workflows where correlation patterns need to be identified
    within protein groups.

    Features:
    ---------
    - **Multiple Distance Transformations**: Convert correlation to distance using
      various methods optimized for different correlation patterns
    - **Flexible Clustering Methods**: Support for balanced cut, dynamic tree cut,
      and fixed cut with automatic parameter optimization
    - **Quality Assessment**: Built-in metrics for evaluating clustering quality
    - **Extensible Design**: Easy addition of custom distance transforms and methods
    - **Robust Error Handling**: Graceful handling of edge cases and invalid data
    - **Performance Monitoring**: Built-in benchmarking and timing capabilities

    Clustering Methods:
    ------------------
    - **balanced_cut**: Ensures roughly equal cluster sizes (good for comparative analysis)
    - **dynamic_cut**: Adaptive cutting based on tree structure (best for natural groupings)
    - **fixed_cut**: Cut at specified height or number of clusters (user control)

    Distance Transforms:
    -------------------
    - **1-corr**: Standard correlation distance (linear transformation)
    - **1-abs(corr)**: Absolute correlation distance (considers negative correlations)
    - **sqrt(2*(1-corr))**: Euclidean-like distance (emphasizes strong correlations)
    - **sqrt(1-corr^2)**: Geometric distance (balanced approach)

    Examples:
    ---------
    >>> # Basic clustering with default parameters
    >>> clusterer = ProteoformClustering()
    >>> labels = clusterer.fit_predict(correlation_matrix)
    >>> print(f"Found {len(set(labels))} clusters")
    
    >>> # Advanced clustering with custom parameters
    >>> clusterer = ProteoformClustering(
    ...     distance_transform='sqrt(2*(1-corr))',
    ...     clustering_method='dynamic_cut',
    ...     linkage_method='ward',
    ...     auto_n_clusters=True,
    ...     min_clusters=2,
    ...     max_clusters=8
    ... )
    >>> labels = clusterer.fit_predict(correlation_matrix)
    
    >>> # Quality assessment
    >>> quality = clusterer.assess_quality(correlation_matrix, labels)
    >>> print(f"Clustering quality (silhouette): {quality:.3f}")
    """

    # Class-level dictionaries for extensible method definitions
    DISTANCE_TRANSFORMS = {
        # Standard transformations
        'corr': lambda x: x, 
        '1-corr': lambda x: 1 - x,
        '1-abs(corr)': lambda x: 1 - np.abs(x),
        
        # Advanced transformations
        '1-abs(corr)^2': lambda x: 1 - np.abs(x ** 2),
        '1-abs(sqrt(corr))': lambda x: 1 - np.sqrt(np.abs(x)),
        'sqrt(2*(1-corr))': lambda x: np.sqrt(2 * (1 - x)),  # Euclidean-like
        'sqrt(1-corr^2)': lambda x: np.sqrt(1 - x ** 2),  # Geometric distance
    }

    def __init__(
        self,
        distance_transform: str = '1-corr',
        clustering_method: str = 'hybrid_outlier_cut', 
        linkage_method: str = 'complete',
        distance_metric: str = 'euclidean',
        verbose: bool = False,
        min_clusters: int = 2,
        max_clusters: int = 10,
        use_distance_matrix: bool = True,
    ):
        """
        Initialize the ProteoformClustering framework.

        Args:
            distance_transform: Method for transforming correlation to distance.
                              Available: '1-corr', '1-abs(corr)', 'sqrt(2*(1-corr))', etc.
            clustering_method: Clustering algorithm to use.
                             Options: 'balanced_cut', 'dynamic_cut', 'fixed_cut'
            linkage_method: Hierarchical linkage method for dendrogram construction.
                          Options: 'complete', 'average', 'single', 'ward'
            distance_metric: Distance metric for pdist computation.
                           Options: 'euclidean', 'manhattan', 'cosine', 'correlation'
            verbose: Enable detailed output during clustering operations
            auto_n_clusters: Enable automatic cluster number selection (for fixed_cut)
            min_clusters: Minimum number of clusters for auto-selection
            max_clusters: Maximum number of clusters for auto-selection
            cut_func: Custom function for dendrogram cutting (overrides built-in methods)
            use_distance_matrix: Whether to compute and use distance matrix transformation

        Raises:
            ValueError: If distance_transform is not available

        Notes:
            - All parameters can be modified after initialization
            - Custom distance transforms can be added using add_distance_transform()
            - Clustering results are stored for quality assessment and analysis

        Examples:
            >>> # Conservative clustering for comparative analysis
            >>> clusterer = ProteoformClustering(
            ...     distance_transform='1-abs(corr)',
            ...     clustering_method='balanced_cut',
            ...     linkage_method='average'
            ... )
            
            >>> # Adaptive clustering for natural groupings
            >>> clusterer = ProteoformClustering(
            ...     distance_transform='sqrt(2*(1-corr))',
            ...     clustering_method='dynamic_cut',
            ...     verbose=True
            ... )
            
            >>> # Automatic optimization
            >>> clusterer = ProteoformClustering(
            ...     clustering_method='fixed_cut',
            ...     auto_n_clusters=True,
            ...     min_clusters=2,
            ...     max_clusters=6
            ... )
        """
        self.distance_transform = distance_transform
        self.clustering_method = clustering_method
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.verbose = verbose
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.use_distance_matrix = use_distance_matrix
        self._validate_methods()
        self.last_results_ = None
        self.last_correlation_matrix_ = None

        self._validate_methods()

        # Store results from last clustering
        self.last_results_ = None
        self.last_correlation_matrix_ = None

    def _validate_methods(self):
        if self.distance_transform not in self.DISTANCE_TRANSFORMS:
            raise ValueError(
                f"Distance transform '{self.distance_transform}' not available. "
                f"Available: {list(self.DISTANCE_TRANSFORMS.keys())}"
            )
        
        # Check clustering method - warn but don't error for invalid methods
        valid_methods = [
            'hybrid_outlier_cut', 'dendrogram_cut', 'balanced_cut', 
            'dynamic_cut', 'fixed_cut'
        ]
        if self.clustering_method not in valid_methods and not callable(self.clustering_method):
            print(f"Warning: Unknown clustering method '{self.clustering_method}'. Will use 'fixed_cut' as fallback.")
            self.clustering_method = 'fixed_cut'
        
    def _calc_distance_matrix(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Transform correlation matrix to distance matrix using selected method.

        Args:
            corr_matrix: Input correlation matrix

        Returns:
            Transformed distance matrix with NaN/inf values handled
        """
        transform_func = self.DISTANCE_TRANSFORMS[self.distance_transform]
        dist = transform_func(corr_matrix)
        return np.nan_to_num(dist, nan=0.0, posinf=1.0, neginf=-1.0)

    def _prepare_correlation_matrix(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Prepare and validate correlation matrix for clustering.
        
        Args:
            corr_matrix: Input correlation matrix
            
        Returns:
            Cleaned correlation matrix with finite values
        """
        corr_matrix = np.asarray(corr_matrix)
        if not np.isfinite(corr_matrix).all():
            if self.verbose:
                print("Warning: Correlation matrix contains NaN or infinite values, replacing them.")
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        return corr_matrix

    @classmethod
    def add_distance_transform(cls, name: str, transform_func: Callable):
        """
        Add a custom distance transformation method to the class.

        Extends the available distance transformations with user-defined functions,
        allowing for specialized distance metrics tailored to specific applications.

        Args:
            name: Unique name for the new transformation
            transform_func: Function that takes a correlation matrix and returns distances

        Notes:
            - Transform function should accept numpy arrays and return numpy arrays
            - Function should handle NaN values appropriately
            - Added transforms are available to all instances of the class

        Examples:
            >>> # Add a custom logarithmic transform
            >>> def log_transform(corr_matrix):
            ...     return -np.log(np.abs(corr_matrix) + 1e-10)
            >>> 
            >>> ProteoformClustering.add_distance_transform('log_distance', log_transform)
            >>> 
            >>> # Use the new transform
            >>> clusterer = ProteoformClustering(distance_transform='log_distance')
            
            >>> # Add a threshold-based transform
            >>> def threshold_transform(corr_matrix, threshold=0.5):
            ...     return np.where(np.abs(corr_matrix) > threshold, 0, 1)
            >>> 
            >>> ProteoformClustering.add_distance_transform('threshold_0.5', 
            ...                                           lambda x: threshold_transform(x, 0.5))
        """
        cls.DISTANCE_TRANSFORMS[name] = transform_func

    def get_available_methods(self) -> Dict[str, List[str]]:
        return {
            'distance_transforms': list(self.DISTANCE_TRANSFORMS.keys()),
            'clustering_methods': [
                'hybrid_outlier_cut',
                'dendrogram_cut', 
                'balanced_cut', 
                'dynamic_cut', 
                'fixed_cut'
            ],
            'linkage_methods': ['complete', 'average', 'single', 'ward', 'centroid', 'median'],
            'distance_metrics': ['euclidean', 'manhattan', 'cosine', 'correlation']
        }

    def _cluster_balanced_cut(
        self,
        linkage_matrix: np.ndarray,
        method_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Perform balanced cut clustering ensuring roughly equal cluster sizes.

        Args:
            linkage_matrix: Hierarchical linkage matrix
            method_params: Parameters including max_size_pct and absolute_max_size

        Returns:
            Cluster labels array (1-based indexing)
        """
        max_size_pct = method_params.get('max_size_pct', 0.75)
        absolute_max_size = method_params.get('absolute_max_size', 3)
        n_samples = linkage_matrix.shape[0] + 1
        max_cluster_size = max(absolute_max_size, int(n_samples * max_size_pct))
        if self.verbose: print(f" - Using balanced cut with max_cluster_size = {max_cluster_size}")
        labels = cut_balanced(linkage_matrix, max_cluster_size=max_cluster_size)
        # Convert from 0-based to 1-based indexing to match other methods
        return labels + 1


    def _cluster_dynamic_cut(
        self,
        linkage_matrix: np.ndarray,
        distances: np.ndarray,
        method_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Perform dynamic tree cut clustering with adaptive height selection.

        Args:
            linkage_matrix: Hierarchical linkage matrix
            distances: Pairwise distances for dynamic cutting
            method_params: Parameters including min_size_pct

        Returns:
            Cluster labels array
        """
        min_size_pct = method_params.get('min_size_pct', 0.20)
        n_samples = linkage_matrix.shape[0] + 1
        min_cluster_size = max(1, int(n_samples * min_size_pct))
        if self.verbose: print(f" - Using dynamic cut with min_cluster_size = {min_cluster_size}")
        distance_matrix = pdist_to_squareform(distances)
        result = cutreeHybrid(linkage_matrix, distance_matrix, minClusterSize=min_cluster_size, verbose=self.verbose)
        return result['labels']

    def _cluster_fixed_cut(
        self, 
        linkage_matrix: np.ndarray, 
        method_params: Dict[str, Any], 
        **kwargs
    ) -> np.ndarray:
        """
        Perform fixed cut clustering at specified height or number of clusters.

        Args:
            linkage_matrix: Hierarchical linkage matrix
            method_params: Parameters including n_clusters and cut_height
            distances: Optional distances for auto-selection

        Returns:
            Cluster labels array
        """
        n_clusters = method_params.get('n_clusters', self.min_clusters)
        if self.verbose: print(f" - Using fixed cut with n_clusters = {n_clusters}")
        return fcluster(linkage_matrix, n_clusters, criterion='maxclust')


    def _cluster_dendrogram_cut(
        self, 
        linkage_matrix: np.ndarray, 
        **kwargs
    ) -> np.ndarray:
        """Highly efficient cut based on dendrogram structure."""
        n_samples = linkage_matrix.shape[0] + 1
        merge_distances = linkage_matrix[:, 2]
        distance_acceleration = np.diff(np.diff(merge_distances))

        if distance_acceleration.size > 0:
            best_k_index = np.argmax(distance_acceleration) + 2
            best_n_clusters = n_samples - best_k_index
        else:
            best_n_clusters = self.min_clusters

        # Clamp the result within the user-defined min/max bounds
        best_n_clusters = max(self.min_clusters, min(self.max_clusters, best_n_clusters))
        
        if self.verbose:
            print(f" - Auto-selected {best_n_clusters} clusters via dendrogram analysis.")
        
        return fcluster(linkage_matrix, best_n_clusters, criterion='maxclust')

    def _cluster_hybrid_outlier_cut(
        self, 
        linkage_matrix: np.ndarray, 
        distances: np.ndarray, 
        method_params: Dict[str, Any], 
        **kwargs
    ) -> np.ndarray:
        """
        Performs a hybrid cut: finds main clusters via dendrogram analysis,
        then isolates poorly-fit outliers using silhouette scores.
        """
        if self.verbose:
            print(" - Using hybrid cut with outlier detection.")
            
        # --- Step 1: Perform the initial, fast dendrogram cut ---
        initial_labels = self._cluster_dendrogram_cut(linkage_matrix)
        # initial_labels = self._cluster_auto_quality_cut(linkage_matrix, distances)
        
        n_clusters_initial = len(np.unique(initial_labels))
        
        # If we only found one cluster, there's nothing to compare, so we can't get scores.
        if n_clusters_initial < 2:
            return initial_labels
            
        # --- Step 2: Calculate silhouette scores for each sample ---
        # silhouette_samples requires a square distance matrix
        dist_matrix_sq = pdist_to_squareform(distances)
        try:
            # This function calculates the silhouette score for each individual point
            per_sample_scores = silhouette_samples(dist_matrix_sq, initial_labels, metric='precomputed')
        except ValueError:
            # This can happen if a cluster has only one member initially
            return initial_labels

        # --- Step 3: Identify and re-label outliers ---
        outlier_threshold = method_params.get('outlier_threshold', 0.05)
        final_labels = np.copy(initial_labels)
        
        # Find points with scores below the threshold
        outlier_indices = np.where(per_sample_scores < outlier_threshold)[0]
        
        if self.verbose and len(outlier_indices) > 0:
            print(f"   - Identified {len(outlier_indices)} potential outlier(s) with silhouette score < {outlier_threshold}.")

        # Assign each outlier to a new, unique cluster ID
        if len(outlier_indices) > 0:
            next_cluster_id = np.max(final_labels) + 1
            for idx in outlier_indices:
                # Only re-label if it's not already a single-member cluster
                if np.sum(final_labels == final_labels[idx]) > 1:
                    final_labels[idx] = next_cluster_id
                    next_cluster_id += 1
            
        return final_labels    

    def _cluster_auto_quality_cut(
        self, 
        linkage_matrix: np.ndarray, 
        distances: np.ndarray
    ) -> np.ndarray:
        """
        Automatically determines the best number of clusters by optimizing the silhouette score.
        This method is data-driven and requires no manual parameter tuning.
        """
        n_samples = linkage_matrix.shape[0] + 1
        
        # Define the range of cluster numbers to test
        # We can't have more clusters than samples
        max_k = min(self.max_clusters, n_samples - 1)
        min_k = max(self.min_clusters, 2)

        # If the range is invalid (e.g., only 2 peptides), return a single cluster
        if max_k < min_k:
            return np.ones(n_samples, dtype=int)

        k_values = range(min_k, max_k + 1)
        silhouette_scores = []
        
        # We need the square distance matrix for silhouette score calculation
        dist_matrix_sq = pdist_to_squareform(distances)

        for k in k_values:
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            # Silhouette score is only valid for 2 or more clusters
            if len(np.unique(labels)) < 2:
                score = -1.0 # Assign a poor score if clustering fails
            else:
                score = silhouette_score(dist_matrix_sq, labels, metric='precomputed')
            silhouette_scores.append(score)
        
        # Find the number of clusters that resulted in the best silhouette score
        best_k = k_values[np.argmax(silhouette_scores)]
        
        if self.verbose:
            print(f" - Auto-selected {best_k} clusters via silhouette optimization.")
        
        # Return the cluster labels for the optimal k
        return fcluster(linkage_matrix, best_k, criterion='maxclust')

    def _calculate_validation_scores(
            self, 
            linkage_matrix: np.ndarray, 
            distances: np.ndarray
    ) -> Tuple[Dict[str, list], List[int]]:
        
        """Helper to calculate scores for a range of k values."""
        n_samples = linkage_matrix.shape[0] + 1
        max_k = min(self.max_clusters, n_samples - 1)
        min_k = max(self.min_clusters, 2)

        if max_k < min_k:
            return {}, []

        k_values = list(range(min_k, max_k + 1))
        scores = {'silhouette': [], 'davies_bouldin': [], 'calinski_harabasz': []}
        dist_matrix_sq = pdist_to_squareform(distances)

        for k in k_values:
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            if len(set(labels)) < 2:
                scores['silhouette'].append(-1)
                scores['davies_bouldin'].append(np.inf)
                scores['calinski_harabasz'].append(-1)
                continue
            try:
                scores['silhouette'].append(silhouette_score(dist_matrix_sq, labels, metric='precomputed'))
                scores['davies_bouldin'].append(davies_bouldin_score(dist_matrix_sq, labels))
                scores['calinski_harabasz'].append(calinski_harabasz_score(dist_matrix_sq, labels))
            except (ValueError, ZeroDivisionError):
                scores['silhouette'].append(-1)
                scores['davies_bouldin'].append(np.inf)
                scores['calinski_harabasz'].append(-1)
        
        return scores, k_values

    def fit_predict(
        self,
        corr_matrix: np.ndarray,
        method_params: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Perform complete clustering pipeline on correlation matrix.

        Primary method for executing the clustering workflow, from correlation
        matrix input to final cluster label assignment.

        Args:
            corr_matrix: Correlation matrix to cluster (square, symmetric)
            method_params: Optional parameters specific to clustering method

        Returns:
            np.ndarray: Cluster labels for each element in the correlation matrix

        Raises:
            ValueError: If clustering_method is unknown or matrix is invalid

        Notes:
            - Correlation matrix should be square and symmetric
            - Missing values (NaN) are automatically handled
            - Results are stored for subsequent quality assessment

        Examples:
            >>> import numpy as np
            >>> 
            >>> # Create sample correlation matrix
            >>> corr_matrix = np.array([[1.0, 0.8, 0.2],
            ...                        [0.8, 1.0, 0.3],
            ...                        [0.2, 0.3, 1.0]])
            >>> 
            >>> # Basic clustering
            >>> clusterer = ProteoformClustering()
            >>> labels = clusterer.fit_predict(corr_matrix)
            >>> print(f"Cluster labels: {labels}")
            
            >>> # Clustering with method-specific parameters
            >>> clusterer = ProteoformClustering(clustering_method='balanced_cut')
            >>> labels = clusterer.fit_predict(corr_matrix, 
            ...                               method_params={'max_size_pct': 0.6})
            
            >>> # Fixed cut with specific number of clusters
            >>> clusterer = ProteoformClustering(clustering_method='fixed_cut')
            >>> labels = clusterer.fit_predict(corr_matrix,
            ...                               method_params={'n_clusters': 2})
        """

        if method_params is None: method_params = {}
        
        corr_matrix = self._prepare_correlation_matrix(corr_matrix)
        
        # Define a map from method name to function
        method_map = {
            'hybrid_outlier_cut': self._cluster_hybrid_outlier_cut, 
            'dendrogram_cut': self._cluster_dendrogram_cut,
            'balanced_cut': self._cluster_balanced_cut,
            'dynamic_cut': self._cluster_dynamic_cut,
            'fixed_cut': self._cluster_fixed_cut,
            # 'affinity_propagation': self._cluster_affinity_propagation,
        }
        # Handle Affinity Propagation separately as it doesn't use linkage
        if self.clustering_method == 'affinity_propagation':
            clusters = self._cluster_affinity_propagation(corr_matrix, method_params)
        
        # All other methods are hierarchical and require linkage matrix
        else:
            if self.use_distance_matrix:
                # _calc_distance_matrix returns a square pairwise distance matrix
                # Convert the square distance matrix to condensed form for linkage
                distance_matrix = self._calc_distance_matrix(corr_matrix)
                # squareform will convert a square (n,n) distance matrix to
                # the condensed (n*(n-1)/2,) vector expected by linkage
                distances = squareform(distance_matrix, checks=False)
            else:
                # When not using a precomputed distance matrix, treat corr_matrix
                # as feature-like and compute pairwise distances with pdist
                distances = pdist(corr_matrix, metric=self.distance_metric)

            # Compute hierarchical linkage from condensed distances
            linkage_matrix = linkage(distances, method=self.linkage_method)
            
            # Get the appropriate clustering function
            cluster_func = method_map.get(self.clustering_method)
            
            if cluster_func:
                # Call the function with method-specific arguments
                if self.clustering_method == 'balanced_cut':
                    clusters = cluster_func(linkage_matrix, method_params)
                elif self.clustering_method == 'dynamic_cut':
                    clusters = cluster_func(linkage_matrix, distances, method_params)
                elif self.clustering_method == 'dendrogram_cut':
                    clusters = cluster_func(linkage_matrix)
                elif self.clustering_method == 'fixed_cut':
                    clusters = cluster_func(linkage_matrix, method_params)
                elif self.clustering_method == 'hybrid_outlier_cut':
                    clusters = cluster_func(linkage_matrix, distances, method_params)
                else:
                    # Fallback for any other methods - try with all arguments
                    clusters = cluster_func(
                        linkage_matrix=linkage_matrix, 
                        distances=distances,
                        corr_matrix=corr_matrix,
                        method_params=method_params
                    )
            elif callable(self.clustering_method):
                 clusters = self.clustering_method(
                    linkage_matrix=linkage_matrix, 
                    distances=distances,
                    corr_matrix=corr_matrix,
                    method_params=method_params
                )

        self.last_results_ = clusters
        self.last_correlation_matrix_ = corr_matrix
        return clusters

    def assess_quality(
        self, 
        corr_matrix: np.ndarray, 
        labels: np.ndarray, 
        metric: str = 'silhouette', 
        use_distance_matrix: bool = True
    ) -> float:
        """
        Assess clustering quality using standard validation metrics.

        Evaluates the quality of clustering results using established metrics
        to help determine optimal clustering parameters and compare methods.

        Args:
            corr_matrix: Original correlation matrix used for clustering
            labels: Cluster labels from clustering algorithm
            metric: Quality metric to use for assessment.
                   Options: 'silhouette', 'davies_bouldin', 'calinski_harabasz'
            use_distance_matrix: Whether to transform correlation to distance first

        Returns:
            float: Quality score (higher is better for silhouette/calinski_harabasz,
                  lower is better for davies_bouldin)

        Raises:
            ValueError: If metric is not recognized

        Notes:
            - Silhouette score: [-1, 1], higher is better, >0.5 is good
            - Davies-Bouldin score: [0, inf), lower is better, <1 is good  
            - Calinski-Harabasz score: [0, inf), higher is better

        Examples:
            >>> # Assess clustering quality
            >>> clusterer = ProteoformClustering()
            >>> labels = clusterer.fit_predict(corr_matrix)
            >>> 
            >>> # Silhouette analysis
            >>> silhouette = clusterer.assess_quality(corr_matrix, labels, 'silhouette')
            >>> print(f"Silhouette score: {silhouette:.3f}")
            >>> 
            >>> # Compare multiple metrics
            >>> for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
            ...     score = clusterer.assess_quality(corr_matrix, labels, metric)
            ...     print(f"{metric}: {score:.3f}")
        """
        if use_distance_matrix:
            dist_matrix = self._calc_distance_matrix(corr_matrix)
        else:
            dist_matrix = corr_matrix
            
        if metric == 'silhouette':
            return silhouette_score(
                dist_matrix if use_distance_matrix else corr_matrix, 
                labels, 
                metric='precomputed' if use_distance_matrix else 'euclidean'
            )
        elif metric == 'davies_bouldin':
            return davies_bouldin_score(dist_matrix, labels)
        elif metric == 'calinski_harabasz':
            return calinski_harabasz_score(dist_matrix, labels)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def benchmark(
        self, 
        corr_matrix: np.ndarray, 
        methods: List[str] = None, 
        metrics: List[str] = None, 
        n_repeats: int = 1
    ) -> pd.DataFrame:
        """
        Comprehensive benchmarking of clustering methods for quality and performance.

        Compares multiple clustering methods across quality metrics and timing
        to help select optimal parameters for specific datasets.

        Args:
            corr_matrix: Correlation matrix for benchmarking
            methods: List of clustering methods to compare.
                    Default: ['balanced_cut', 'dynamic_cut', 'fixed_cut']
            metrics: List of quality metrics to evaluate.
                    Default: ['silhouette']
            n_repeats: Number of timing repetitions for statistical reliability

        Returns:
            pd.DataFrame: Benchmark results with columns:
                         method, metric, score, time_sec

        Notes:
            - Higher repetitions provide more reliable timing estimates
            - Results are sorted by quality score for easy comparison
            - Failed methods return NaN scores for quality metrics

        Examples:
            >>> # Quick benchmark with defaults
            >>> clusterer = ProteoformClustering()
            >>> results = clusterer.benchmark(corr_matrix)
            >>> print(results.sort_values('score', ascending=False))
            
            >>> # Comprehensive benchmark
            >>> methods = ['balanced_cut', 'dynamic_cut', 'fixed_cut']
            >>> metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
            >>> results = clusterer.benchmark(corr_matrix, methods, metrics, n_repeats=5)
            >>> 
            >>> # Find best method for each metric
            >>> for metric in metrics:
            ...     best = results[results['metric'] == metric].iloc[0]
            ...     print(f"Best {metric}: {best['method']} ({best['score']:.3f})")
        """
        if methods is None:
            methods = ['balanced_cut', 'dynamic_cut', 'fixed_cut']
        if metrics is None:
            metrics = ['silhouette']
            
        results = []
        
        for method in methods:
            # Temporarily change method
            original_method = self.clustering_method
            self.clustering_method = method
            
            # Determine distance matrix usage
            use_dist = method not in ['mean_threshold', 'std_quantile']
            original_use_dist = self.use_distance_matrix
            self.use_distance_matrix = use_dist
            
            # Time the clustering
            times = []
            for _ in range(n_repeats):
                start_time = time.time()
                labels = self.fit_predict(corr_matrix)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Assess quality for each metric
            for metric in metrics:
                try:
                    score = self.assess_quality(
                        corr_matrix, labels, metric=metric, use_distance_matrix=use_dist
                    )
                except Exception:
                    score = np.nan
                    
                results.append({
                    'method': method,
                    'metric': metric,
                    'score': score,
                    'time_sec': np.mean(times),
                })
            
            # Restore original settings
            self.clustering_method = original_method
            self.use_distance_matrix = original_use_dist
        
        return pd.DataFrame(results)

# ==============================================================================
# Batch Processing Utilities
# ==============================================================================

def _for_single_protein(
    protein: str, 
    corr_matrix: pd.DataFrame, 
    min_clusters: int,
    max_clusters: Optional[int] = None,
    distance_transform: str = '1-corr',
    clustering_method: str = 'fixed_cut',
    linkage_method: str = 'complete',
    distance_metric: str = 'correlation',
    verbose: bool = False
) -> Tuple[str, Optional[np.ndarray]]:
    """
    Process clustering for a single protein's correlation matrix.

    Worker function for parallel processing of multiple protein correlation matrices.
    Handles all aspects of single-protein clustering with comprehensive error handling.

    Args:
        protein: Protein identifier
        corr_matrix: Correlation matrix for the protein
        min_clusters: Minimum number of clusters to form
        max_clusters: Maximum number of clusters (defaults to n-1 peptides)
        distance_transform: Distance transformation method
        clustering_method: Clustering algorithm to use
        linkage_method: Hierarchical linkage method
        distance_metric: Distance metric for pdist
        auto_n_clusters: Enable automatic cluster number selection
        use_distance_matrix: Whether to use distance matrix transformation
        verbose: Enable detailed output

    Returns:
        Tuple of (protein_name, cluster_labels) or (protein_name, None) if failed

    Notes:
        - Requires at least 4 peptides for meaningful clustering
        - Automatically adjusts max_clusters if it exceeds data constraints
        - Returns None for failed clustering attempts with error logging
    """
    try:
        n = corr_matrix.shape[0]
        if n < 4:
            return (protein, None)  # Not enough peptides to cluster
            
        if max_clusters is None:
            max_clusters = n - 1    # Default to n-1 if not specified
            
        # Check if max_clusters is less than min_clusters
        if max_clusters < min_clusters:
            raise ValueError(
                f"max_clusters ({max_clusters}) cannot be less than min_clusters ({min_clusters})."
            )   
            
        if max_clusters > n - 1:
            max_clusters = n - 1
        
        clusterer = ProteoformClustering(
            distance_transform=distance_transform,
            clustering_method=clustering_method,
            linkage_method=linkage_method,
            distance_metric=distance_metric,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            verbose=verbose, 
        )
        
        labels = clusterer.fit_predict(corr_matrix.values)
        return (protein, labels)
        
    except Exception as e:
        if verbose: print(f"An error occurred while processing protein '{protein}': {e}, returning 0s")
        return (protein, np.zeros(corr_matrix.shape[0], dtype=int) if n > 0 else None)


def all_proteins(
    corr_dict: Dict[str, pd.DataFrame],
    min_clusters: int = 1,
    max_clusters: Optional[int] = None,
    distance_transform: str = '1-corr',
    clustering_method: str = 'fixed_cut',
    linkage_method: str = 'complete',
    distance_metric: str = 'correlation',
    verbose: bool = False,
    n_jobs: int = -1
) -> Dict[str, Optional[np.ndarray]]:
    """
    Perform high-performance batch clustering across multiple protein correlation matrices.

    Efficiently processes large collections of protein correlation matrices using
    multiprocessing acceleration. Optimized for proteomics workflows where hundreds
    or thousands of proteins need independent clustering analysis.

    Args:
        corr_dict: Dictionary mapping protein names to correlation matrices
        min_clusters: Minimum number of clusters to form for each protein
        max_clusters: Maximum number of clusters (auto-determined if None)
        distance_transform: Distance transformation method to apply
        clustering_method: Clustering algorithm for all proteins
        linkage_method: Hierarchical linkage method
        distance_metric: Distance metric for pdist computation
        auto_n_clusters: Enable automatic cluster number optimization
        use_distance_matrix: Whether to apply distance matrix transformation
        verbose: Enable progress reporting and detailed output
        n_jobs: Number of CPU cores to use (-1 for all available)

    Returns:
        Dict mapping protein names to cluster labels (or None for failed clustering)

    Notes:
        - Automatically scales multiprocessing based on available CPU cores
        - Gracefully handles clustering failures for individual proteins
        - Optimized for memory efficiency with large protein collections
        - Serial fallback available when n_jobs=1

    Examples:
        >>> # Basic batch clustering
        >>> results = all_proteins(correlation_dict, n_jobs=4)
        >>> successful = {k: v for k, v in results.items() if v is not None}
        >>> print(f"Successfully clustered {len(successful)} proteins")
        
        >>> # Advanced batch processing with custom parameters
        >>> results = all_proteins(
        ...     correlation_dict,
        ...     min_clusters=2,
        ...     max_clusters=6,
        ...     distance_transform='sqrt(2*(1-corr))',
        ...     clustering_method='dynamic_cut',
        ...     auto_n_clusters=True,
        ...     verbose=True,
        ...     n_jobs=-1
        ... )
        
        >>> # Serial processing for debugging
        >>> results = all_proteins(correlation_dict, verbose=True, n_jobs=1)
    """
    num_workers = select_n_jobs(n_jobs)

    if num_workers == 1:
        # Serial processing
        if verbose:
            print("Running in serial mode.")
        results = {}
        for protein, corr_matrix in corr_dict.items():
            _, labels = _for_single_protein(
                protein, corr_matrix, min_clusters,
                max_clusters=max_clusters,
                distance_transform=distance_transform,
                clustering_method=clustering_method,
                linkage_method=linkage_method,
                distance_metric=distance_metric,
                verbose=verbose
            )
            results[protein] = labels
        return results

    # Parallel processing
    results = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_protein = {
            executor.submit(
                _for_single_protein,
                protein, corr_matrix, min_clusters,
                max_clusters,
                distance_transform,
                clustering_method,
                linkage_method,
                distance_metric,

                verbose
            ): protein
            for protein, corr_matrix in corr_dict.items()
        }

        if verbose:
            print(f"Submitted {len(future_to_protein)} clustering jobs to {num_workers} workers.")
            
        for future in as_completed(future_to_protein):
            protein, labels = future.result()
            if protein: 
                results[protein] = labels

    return results


def process_results(
    clustering_results: Dict[str, Optional[np.ndarray]],
    protein_col: str = 'Protein',
    peptide_id_col: str = 'PeptideID',
    cluster_col: str = 'Cluster',
) -> pd.DataFrame:
    """
    Convert clustering results dictionary to structured DataFrame format.

    Transforms the output of batch clustering operations into a tidy DataFrame
    suitable for downstream analysis and visualization.

    Args:
        clustering_results: Dictionary from all_proteins() function
        protein_col: Column name for protein identifiers
        peptide_id_col: Column name for peptide/proteoform identifiers
        cluster_col: Column name for cluster assignments

    Returns:
        pd.DataFrame: Structured data with protein, peptide, and cluster columns

    Notes:
        - Automatically handles failed clustering attempts (None values)
        - Peptide IDs are adjusted to start from 1 (not 0)
        - Missing cluster assignments are dropped from output

    Examples:
        >>> # Process batch clustering results
        >>> clustering_dict = all_proteins(correlation_matrices)
        >>> df = process_results(clustering_dict)
        >>> print(df.head())
        
        >>> # Custom column names
        >>> df = process_results(
        ...     clustering_dict,
        ...     protein_col='ProteinID',
        ...     peptide_id_col='ProteoformID',
        ...     cluster_col='ClusterGroup'
        ... )
    """
    data = pd.DataFrame.from_dict(
        clustering_results, 
        orient='index',
    ).reset_index().rename(
        columns={'index': protein_col}
    ).melt(
        id_vars=protein_col, 
        var_name=peptide_id_col, 
        value_name=cluster_col,
        ignore_index=True
    ).dropna(subset=[cluster_col])
    
    # Adjust PeptideID to start from 1
    data[peptide_id_col] = data[peptide_id_col].astype(int) + 1  
    return data


def dict_to_data(
    test_data: pd.DataFrame, 
    clustering_results: Dict[str, Optional[np.ndarray]],
    protein_col: str = 'Protein',
    peptide_id_col: str = 'PeptideID',
    cluster_col: str = 'ClusterID',
) -> pd.DataFrame:
    """
    Merge clustering results with original dataset for comprehensive analysis.

    Combines clustering assignments with the original dataset to enable
    integrated analysis of clustering results with protein/peptide metadata.

    Args:
        test_data: Original dataset containing protein and peptide information
        clustering_results: Dictionary from all_proteins() function
        protein_col: Column name for protein identifiers
        peptide_id_col: Column name for peptide identifiers  
        cluster_col: Column name for cluster assignments

    Returns:
        pd.DataFrame: Merged dataset with original data plus cluster assignments

    Raises:
        ValueError: If required columns are not found in test_data

    Notes:
        - Automatically detects whether identifiers are in columns or index
        - Preserves all original data columns
        - Cluster IDs are converted to nullable integer type
        - Left join preserves all original data, even without cluster assignments

    Examples:
        >>> # Merge with original proteomic data
        >>> merged_data = dict_to_data(original_dataset, clustering_results)
        >>> print(f"Added clustering data to {len(merged_data)} rows")
        
        >>> # Analyze clustering by protein properties
        >>> cluster_summary = merged_data.groupby(['Protein', 'ClusterID']).size()
        >>> print(cluster_summary.head())
    """
    # Convert clustering results to DataFrame
    clustering_df = process_results(
        clustering_results,
        protein_col=protein_col,
        peptide_id_col=peptide_id_col,
        cluster_col=cluster_col,
    )
    
    # Check if required columns exist in test_data
    if protein_col not in test_data.columns or peptide_id_col not in test_data.columns:
        # Check if they are in the index
        if protein_col in test_data.index.names and peptide_id_col in test_data.index.names:
            # Reset the index to make them columns
            test_data = test_data.reset_index()
        else:
            # Raise an error if they are not found
            raise ValueError(
                f"Columns {protein_col} and {peptide_id_col} must exist in the test_data DataFrame."
            )
        
    # Set up indices for efficient merging
    test_data_indexed = test_data.set_index([protein_col, peptide_id_col])
    clustering_df_indexed = clustering_df.set_index([protein_col, peptide_id_col])
    
    # Merge datasets
    merged_data = test_data_indexed.merge(
        clustering_df_indexed,
        left_index=True,
        right_index=True,
        how='left'
    )
    
    # Reset index and rename cluster column
    merged_data = merged_data.reset_index()
    merged_data = merged_data.rename(columns={cluster_col: 'ClusterID'})
    
    # Ensure cluster column is nullable integer type
    merged_data['ClusterID'] = merged_data['ClusterID'].astype('Int64')
    
    return merged_data

# ==============================================================================
# Benchmarking and Analysis Utilities  
# ==============================================================================

def grid_search_benchmark(
    corr_matrix: np.ndarray,
    param_grid: List[Dict[str, Any]],
    metrics: List[str] = None,
    n_repeats: int = 1
) -> pd.DataFrame:
    """
    Comprehensive parameter grid search for optimal clustering configuration.

    Systematically evaluates all combinations of clustering parameters to identify
    the best configuration for a given dataset. Essential for parameter optimization
    and method comparison.

    Args:
        corr_matrix: Representative correlation matrix for parameter optimization
        param_grid: List of parameter dictionaries to evaluate
        metrics: Quality metrics for evaluation (default: ['silhouette'])
        n_repeats: Number of timing repetitions for statistical reliability

    Returns:
        pd.DataFrame: Complete benchmark results with parameters and scores

    Notes:
        - Each parameter combination is evaluated independently
        - Failed parameter combinations return NaN scores
        - Results include both quality scores and timing information
        - Useful for identifying parameter sensitivity and optimal ranges

    Examples:
        >>> # Define parameter grid
        >>> param_grid = [
        ...     {
        ...         'distance_transform': '1-corr',
        ...         'clustering_method': 'balanced_cut',
        ...         'linkage_method': 'complete',
        ...         'max_size_pct': 0.5
        ...     },
        ...     {
        ...         'distance_transform': 'sqrt(2*(1-corr))',
        ...         'clustering_method': 'dynamic_cut', 
        ...         'linkage_method': 'ward',
        ...         'min_size_pct': 0.2
        ...     }
        ... ]
        >>> 
        >>> # Run grid search
        >>> results = grid_search_benchmark(
        ...     corr_matrix, 
        ...     param_grid,
        ...     metrics=['silhouette', 'davies_bouldin'],
        ...     n_repeats=3
        ... )
        >>> 
        >>> # Find best parameters
        >>> best_config = results.loc[results['score'].idxmax()]
        >>> print(f"Best configuration: {best_config['method']}")
    """
    if metrics is None:
        metrics = ['silhouette']
        
    results = []
    
    # Define which parameters belong to the class vs method_params
    class_params_set = {
        'distance_transform', 'clustering_method', 'linkage_method', 'distance_metric',
        'verbose', 'auto_n_clusters', 'min_clusters', 'max_clusters', 'cut_func', 
        'use_distance_matrix'
    }
    
    for params in param_grid:
        times = []
        
        # Separate class parameters from method parameters
        class_params = {k: v for k, v in params.items() if k in class_params_set}
        method_params = {k: v for k, v in params.items() 
                        if k not in class_params_set and v is not None}
        
        # Time the clustering multiple times
        for _ in range(n_repeats):
            clusterer = ProteoformClustering(**class_params)
            start_time = time.time()
            labels = clusterer.fit_predict(corr_matrix, method_params=method_params)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Evaluate each quality metric
        for metric in metrics:
            try:
                clusterer = ProteoformClustering(**class_params)
                score = clusterer.assess_quality(
                    corr_matrix, labels, 
                    metric=metric, 
                    use_distance_matrix=class_params.get('use_distance_matrix', True)
                )
            except Exception:
                score = np.nan
                
            results.append({
                **params,
                'method': params.get('clustering_method', ''),
                'metric': metric,
                'score': score,
                'time_sec': float(np.mean(times))
            })
    
    return pd.DataFrame(results)

