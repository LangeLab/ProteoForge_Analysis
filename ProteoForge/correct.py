#!/usr/bin/env python3
"""
Multiple Testing Correction for P-values

A comprehensive implementation of multiple testing correction methods with performance
optimizations and compatibility with statistical standards. Provides methods equivalent
to R's p.adjust function plus additional modern approaches like Storey q-values.

Version: 0.7.0
Date: 2025-07-01
Author: Enes K. Ergin
License: CC BY-NC 4.0
"""

import numpy as np
from scipy import interpolate


# ======================================================================================
# Constants and Configuration
# ======================================================================================

# Supported correction methods with their aliases
SUPPORTED_METHODS = {
    'bonferroni': 'bonferroni',
    'holm': 'holm', 
    'hochberg': 'hochberg',
    'bh': 'benjamini_hochberg',
    'fdr': 'benjamini_hochberg',
    'fdr_bh': 'benjamini_hochberg',
    'by': 'benjamini_yekutieli',
    'qvalue': 'storey_qvalue',
    'none': 'none'
}

# Statsmodels method mapping for validation
STATSMODELS_MAP = {
    'bonferroni': 'bonferroni',
    'holm': 'holm',
    'hochberg': 'simes-hochberg',
    'bh': 'fdr_bh',
    'by': 'fdr_by'
}

# ======================================================================================
# Core Multiple Testing Correction Functions
# ======================================================================================

def qEstimate(
        pv: np.ndarray,
        m: int = None,
        verbose: bool = False,
        lowmem: bool = False,
        pi0: float = None
    ) -> np.ndarray:
    """
    Estimate q-values from p-values using the Storey and Tibshirani (2003) method.

    This is a Python adaptation of the R/qvalue algorithm for controlling the
    positive False Discovery Rate (pFDR) in multiple hypothesis testing scenarios.
    Particularly useful for genomics and high-throughput screening applications.

    Args:
        pv: Array of p-values to estimate q-values for. Can be 1D or 2D.
        m: Number of tests. If not specified, uses pv.size.
        verbose: If True, print verbose messages about pi0 estimation.
        lowmem: Use a memory-efficient in-place algorithm (slower, but less memory).
        pi0: Proportion of true null hypotheses. If None, estimated automatically.

    Returns:
        np.ndarray: Array of q-values of the same shape as input p-values.

    Raises:
        AssertionError: If p-values are not between 0 and 1, or if pi0 is not in [0, 1].

    Notes:
        This function is intended for genomics-scale multiple testing, where the number
        of hypotheses is large and the Storey q-value method is appropriate. For other
        correction methods, see `for_multiple_tests`.

    References:
        Storey JD, Tibshirani R. Statistical significance for genomewide studies. 
        Proc Natl Acad Sci U S A. 2003 Aug 5;100(16):9440-5.
        
    Examples:
        >>> import numpy as np
        >>> # Basic q-value estimation
        >>> pvals = np.array([0.001, 0.01, 0.04, 0.1, 0.5])
        >>> qvals = qEstimate(pvals)
        >>> print(f"Q-values: {qvals}")
        
        >>> # For large datasets with memory constraints
        >>> qvals = qEstimate(large_pvals, lowmem=True)
        
        >>> # With known pi0 (proportion of nulls)
        >>> qvals = qEstimate(pvals, pi0=0.8, verbose=True)
    """
    assert(pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

    original_shape = pv.shape
    pv = pv.ravel()  # flattens the array in place, more efficient than flatten()

    if m is None:
        m = float(len(pv))
    else:
        # the user has supplied an m
        m *= 1.0

    # if the number of hypotheses is small, just set pi0 to 1
    if len(pv) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = np.arange(0, 0.90, 0.01)
        counts = np.array([(pv > i).sum() for i in np.arange(0, 0.9, 0.01)])
        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = np.array(pi0)

        # fit natural cubic spline
        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)
        if verbose:
            print("qvalues pi0=%.3f, estimated proportion of null features " % pi0)

        if pi0 > 1:
            if verbose:
                print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
            pi0 = 1.0

    assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

    if lowmem:
        # low memory version, only uses 1 pv and 1 qv matrices
        qv = np.zeros((len(pv),))
        last_pv = pv.argmax()
        qv[last_pv] = (pi0*pv[last_pv]*m)/float(m)
        pv[last_pv] = -np.inf
        prev_qv = last_pv
        for i in range(int(len(pv))-2, -1, -1):
            cur_max = pv.argmax()
            qv_i = (pi0*m*pv[cur_max]/float(i+1))
            pv[cur_max] = -np.inf
            qv_i1 = prev_qv
            qv[cur_max] = min(qv_i, qv_i1)
            prev_qv = qv[cur_max]

    else:
        p_ordered = np.argsort(pv)
        pv = pv[p_ordered]
        qv = pi0 * m/len(pv) * pv
        qv[-1] = min(qv[-1], 1.0)

        for i in range(len(pv)-2, -1, -1):
            qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

        # reorder qvalues
        qv_temp = qv.copy()
        qv = np.zeros_like(qv)
        qv[p_ordered] = qv_temp

    # reshape qvalues
    qv = qv.reshape(original_shape)

    return qv

def for_multiple_tests(
        pvalues: np.ndarray,
        correction_type: str = "bonferroni",
    ) -> np.ndarray:
    """
    Perform multiple testing correction on p-values using standard statistical methods.

    Comprehensive implementation of multiple testing correction methods equivalent 
    to R's p.adjust function, with additional support for modern FDR methods and
    Storey q-values. All methods are order-invariant and produce reproducible results.

    Args:
        pvalues: Array-like of p-values to correct. Can be 1D or 2D; output will match input shape.
        correction_type: Type of correction to perform. Supported values:
            - "bonferroni": Bonferroni correction (most conservative)
            - "holm": Holm-Bonferroni step-down method
            - "hochberg": Hochberg step-up procedure  
            - "bh", "fdr", "fdr_bh": Benjamini-Hochberg FDR control
            - "by": Benjamini-Yekutieli FDR (independent/positive dependence)
            - "qvalue": Storey q-value method for genomics applications
            - "none": No correction, returns input unchanged

    Returns:
        np.ndarray: Array of adjusted p-values (q-values), same shape as input.

    Raises:
        ValueError: If an unknown correction_type is provided or input is invalid.

    Notes:
        - All methods except "none" and "qvalue" are validated against statsmodels
        - Bonferroni is most conservative; use for small numbers of tests
        - Benjamini-Hochberg (bh/fdr) recommended for most applications
        - Storey q-values optimal for genomics with large numbers of tests
        - All methods handle edge cases (ties, extreme values) appropriately

    Examples:
        >>> import numpy as np
        >>> 
        >>> # Basic usage with different methods
        >>> pvals = np.array([0.01, 0.04, 0.03, 0.02])
        >>> 
        >>> # Bonferroni correction (conservative)
        >>> bonf = for_multiple_tests(pvals, "bonferroni")
        >>> print(f"Bonferroni: {bonf}")
        >>> 
        >>> # Benjamini-Hochberg FDR (recommended)
        >>> bh = for_multiple_tests(pvals, "bh") 
        >>> print(f"BH FDR: {bh}")
        >>> 
        >>> # Holm step-down method
        >>> holm = for_multiple_tests(pvals, "holm")
        >>> print(f"Holm: {holm}")
        >>> 
        >>> # Storey q-values for genomics
        >>> qvals = for_multiple_tests(pvals, "qvalue")
        >>> print(f"Q-values: {qvals}")
        >>> 
        >>> # Works with 2D arrays
        >>> pvals_2d = np.array([[0.01, 0.04], [0.03, 0.02]])
        >>> corrected_2d = for_multiple_tests(pvals_2d, "bh")
        >>> print(f"2D corrected: {corrected_2d}")
        >>> 
        >>> # No correction
        >>> unchanged = for_multiple_tests(pvals, "none")
        >>> assert np.array_equal(unchanged, pvals)
    """
    if correction_type is None or correction_type.lower() == "none":
        return np.array(pvalues)

    pvalues = np.asarray(pvalues)
    original_shape = pvalues.shape
    pvalues = pvalues.ravel()
    n = pvalues.size

    method = correction_type.lower()
    qvalues = np.empty_like(pvalues, dtype=float)

    if method == "bonferroni":
        qvalues = np.minimum(pvalues * n, 1.0)

    elif method == "holm":
        order = np.argsort(pvalues)
        sorted_p = pvalues[order]
        n = len(pvalues)
        holm = np.empty(n, dtype=float)
        for i in range(n):
            holm[i] = (n - i) * sorted_p[i]
        holm = np.minimum(holm, 1.0)
        # Step-down: cumulative maximum from left to right
        for i in range(1, n):
            holm[i] = max(holm[i], holm[i-1])
        # Map back to original order
        qvalues[order] = holm

    elif method == "hochberg":
        order = np.argsort(pvalues)[::-1]
        ranked = np.empty_like(order)
        ranked[order] = np.arange(len(pvalues))
        hoch = (n - ranked) * pvalues[order]
        hoch = np.minimum.accumulate(hoch)
        hoch = np.minimum(hoch, 1.0)
        qvalues[order] = hoch

    elif method in ("bh", "fdr", "fdr_bh"):
        order = np.argsort(pvalues)
        sorted_p = pvalues[order]
        n = float(len(pvalues))
        bh = sorted_p * n / (np.arange(1, len(pvalues) + 1))
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        bh = np.minimum(bh, 1.0)
        qvalues[order] = bh

    elif method == "by":
        order = np.argsort(pvalues)
        sorted_p = pvalues[order]
        n = float(len(pvalues))
        q = np.sum(1.0 / (np.arange(1, int(n) + 1)))
        by = sorted_p * n * q / (np.arange(1, len(pvalues) + 1))
        by = np.minimum.accumulate(by[::-1])[::-1]
        by = np.minimum(by, 1.0)
        qvalues[order] = by

    elif method == "qvalue":
        qvalues = qEstimate(pvalues, m=n)
        qvalues = qvalues.ravel()

    else:
        raise ValueError(
            f"Unknown correction_type: {correction_type}. "
            f"Supported methods: {list(SUPPORTED_METHODS.keys())}"
        )

    return qvalues.reshape(original_shape)
