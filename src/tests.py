import os
import feather
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
import pandas as pd 

import pingouin as pg
from scipy import interpolate
import scipy.special as special
import statsmodels.formula.api as smf

from src import utils

# GLOBAL VARIABLES
COLS = [
    "N1", "N2", "log2FC", 
    "df_p", "df_adjp", 
    "eq_lp", "eq_ladjp", 
    "eq_up", "eq_uadjp",
    "eq_p", "eq_adjp",
    "comb_p", "comb_adjp", 
    "log10(pval)", "log10(adj_pval)",
    "Status"
]

def printParams(
        combs_list: list[tuple[str, str]],
        pThr: float=0.05,
        dfThr: float=1,
        eqThr: float=0.5, 
        cvThr: float=75,
        is_log2: bool=False,
        equalVar: bool=True, 
        correction: str='fdr',
        is_paired: bool=False,
         
    ):
    """
        Prints the test parameters for the given test configuration.
    """
    print("Test Parameters:")
    print(f"  - p-value threshold: {pThr}")
    print(f"  - Equal variance: {equalVar}")
    print(f"  - Use Paired test: {is_paired}")
    print(f"  - Data in log2 Scale: {is_log2}")
    print(f"  - Correction method: {correction}")
    print(f"  - Filter higher than CV%: {cvThr}")
    print(f"  - Difference lfc boundary: (-{dfThr}, {dfThr})")
    print(f"  - Equivalence lfc boundary: (-{eqThr}, {eqThr})")
    print("")
    if len(combs_list) == 1:
        print(f"Single test for {combs_list[0][0]} vs {combs_list[0][1]}")
    elif len(combs_list) > 1 and len(combs_list) < 10:
        print(f"Multiple tests for:")
        for i, comb in enumerate(combs_list):
            print(f"  - {i+1}) {comb[0]} vs {comb[1]}")
    else:
        print(f"Multiple tests for {len(combs_list)} combinations")    

def make_protein_selection_indicator(
      cv_arr: np.ndarray,
      cv_thr: float = 0.15
    ):
    """
        Make the protein selection indicator array
        -1: CV > cv_thr (Unreliably Quantified)
        0: NaN (Not Quantified)
        1: CV <= cv_thr (Robustly Quantified)

        Parameters
        ----------
        cv_arr: np.ndarray
            Array of CV values
        cv_thr: float
            CV threshold

        Returns
        -------
        arr: np.ndarray
            Array of protein selection indicator            
    """

    # invalid cv values
    cv_arr[cv_arr < 0] = np.nan

    # find index for NaN
    nan_idx = np.where(
        np.isnan(cv_arr)
    )[0]
    # find index for CV > cv_thr
    cv_idx = np.where(cv_arr > cv_thr)[0]
    # find index for others
    comp_idx = np.where(
        (cv_arr <= cv_thr) & (~np.isnan(cv_arr))
    )[0]

    # Build the array
    arr = np.zeros(len(cv_arr))
    arr[cv_idx] = -1
    arr[nan_idx] = 0
    arr[comp_idx] = 1

    # Return the array
    return arr    

def _ttest_finish(
        df: np.ndarray, 
        t: np.ndarray, 
        alternative: str
    ):
    """
        Common code between all 3 t-test functions.
    """

    # Calculate p-value based on alternative hypothesis
    if alternative == 'less':
        pval = special.stdtr(df, t)
    elif alternative == 'greater':
        pval = special.stdtr(df, -t)
    elif alternative == 'two-sided':
        pval = special.stdtr(df, -np.abs(t))*2
    else:
        raise ValueError(
            "alternative must be "
            "'less', 'greater' or 'two-sided'"
        )
    # Return t-statistics and p-value
    if t.ndim == 0:
        t = t[()]
    if pval.ndim == 0:
        pval = pval[()]

    return t, pval

def _ttest_CI(
        df: np.ndarray,
        t: np.ndarray,
    ):
    """
        Calculate the confidence intervals for tests.
        Trying to improve the performance (WIP)
    """
    pass

def ttest_ind_with_na(
        m1: np.ndarray,
        m2: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray, 
        equal_var: bool=True,
        alternative: str='two-sided'
    ):
    
    """
        The ttest_ind function from scipy.stats 
        but with the ability to handle missing values.
    """
    # If the samples have equal variances
    if equal_var:
        # Calculate the degrees of freedom
        df = ((n1 + n2) - 2)
        # Calculate the pooled variance
        svar = ((n1-1)*v1 + (n2-1)*v2) / df
        # Calculate the denominator
        denom = np.sqrt(svar*(1.0/n1 + 1.0/n2))
    # If the variances are not equal
    else:
        # Calculate the degrees of freedom
        vn1 = v1/n1
        vn2 = v2/n2
        with np.errstate(divide='ignore', invalid='ignore'):
            df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))
            
        # If df is undefined, variances are zero.
        # It doesn't matter what df is as long as it is not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = np.sqrt(vn1 + vn2)
    
    # Calculate t-statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (m1-m2) / denom
        
    return _ttest_finish(df, t, alternative)

def run_unpaired(
        S1_arr: np.ndarray,
        S2_arr: np.ndarray,
        pThr: float=0.05,
        dfThr: float=1,
        eqThr: float=0.5, 
        equalVar: bool=True, 
        correction: str='fdr', 
    ):
    """
        Running combined t-test and tost for the 
        given 2 samples on an unpaired configuration.
    """  

    # Calculate sample sizes for each protein in each sample
    n1 = S1_arr.shape[1] - np.sum(np.isnan(S1_arr), axis=1)
    n2 = S2_arr.shape[1] - np.sum(np.isnan(S2_arr), axis=1)

    # Check if less than 2 replicates
    if np.any(n1 < 2) or np.any(n2 < 2):
        raise ValueError("Some proteins have less than 2 replicates!")
    # Check if n1 and n2 are equal
    if np.any(n1 != n2):
        if equalVar:
            raise ValueError(
                """Number of replicates from both samples
                are not equal, should you use equalVar=False?"""
            )

    # Calculate useful statistics to use in tests
    m1, m2 = np.nanmean(S1_arr, axis=1), np.nanmean(S2_arr, axis=1)
    v1, v2 = np.nanvar(S1_arr, axis=1, ddof=1), np.nanvar(S2_arr, axis=1, ddof=1)

    # Calculate fold-change (Assumes the data is log2 transformed)
    log2fc = m1 - m2

    # Find the index of proteins to be considered for equivalence
    is_test_eq = np.abs(log2fc) < eqThr

    # Calculate the t-test p-values
    ttest_pval = ttest_ind_with_na(
        m1, 
        m2, 
        v1, 
        v2, 
        n1, 
        n2, 
        equal_var=equalVar,
        alternative='two-sided'
    )[1]
    # Apply multiple testing correction
    ttest_pval_corr = multiple_testing_correction(
        ttest_pval,
        correction_type=correction,
        sample_size=None
    )

    # Calculate the tost p-values
    # Test against upper equivalence bound
    p_greater = ttest_ind_with_na(
        m1 + eqThr,
        m2,
        v1,
        v2,
        n1,
        n2,
        equal_var=True,
        alternative="greater"
    )[1]

    # Test against lower equivalence bound
    p_less = ttest_ind_with_na(
        m1 - eqThr,
        m2,
        v1,
        v2,
        n1,
        n2,
        equal_var=True,
        alternative="less"
    )[1]

    # Corect the greater and less individually
    p_greater_corr = multiple_testing_correction(
        p_greater,
        correction_type=correction,
        sample_size=None
    )
    p_less_corr = multiple_testing_correction(
        p_less,
        correction_type=correction,
        sample_size=None
    )
    # Combine the two p-values
    tost_pval = np.maximum(p_greater, p_less)
    tost_pval_corr = np.maximum(p_greater_corr, p_less_corr)

    # Create combination p and q value arrays 
    comb_pval = np.where(
        is_test_eq, 
        tost_pval, 
        ttest_pval
    )
    comb_pval_corr = np.where(
        is_test_eq,
        tost_pval_corr,
        ttest_pval_corr
    )

    # ttest and tost specific significance conditions
    ttest_signf = (ttest_pval_corr < pThr) & (np.abs(log2fc) > dfThr)
    tost_signf = (tost_pval_corr < pThr) & (np.abs(log2fc) < eqThr)

    # Record boolean for significant p values
    is_signf = np.where(
        is_test_eq,
        tost_signf,
        ttest_signf
    )

    # Create test based logging -1 or 1 for p value
    tmp = np.log10(comb_pval)
    logp = np.where(is_test_eq, tmp, -tmp)
    # Create test based logging -1 or 1 for q value
    tmp = np.log10(comb_pval_corr)
    logq = np.where(is_test_eq, tmp, -tmp)

    # Calculate protein status based on significance
    prot_status = np.where(
        is_signf,               # If significant
        np.where(
            is_test_eq,         # If equivalence test
            1.,                 # Eq = 1
            -1.,                # Df = -1
        ),
        0.                      # Not significant = 0
    )

    # Return the results as numpy ndarrays
    return np.stack(
        (
            n1, n2, log2fc, 
            ttest_pval, ttest_pval_corr,
            p_less, p_less_corr,
            p_greater, p_greater_corr,
            tost_pval, tost_pval_corr,
            comb_pval, comb_pval_corr,
            logp, logq, prot_status
        ), 
        axis=1
    )

def ttest_rel_with_na(
        d: np.ndarray,
        n: np.ndarray,
        alternative: str='two-sided'
    ):

    """
        The ttest_rel function from scipy.stats 
        but with the ability to handle missing values.
    """
    # Calculate the degrees of freedom
    df = n - 1
    # Mean difference
    dm = d.mean(axis=1)
    # Calculate the variance of the difference
    v = d.var(axis=1, ddof=1)
    # Calculate the denominator
    denom = np.sqrt(v / n)
    # Calculate t-statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        t = dm / denom
    # Calculate p-value based on alternative hypothesis
    return _ttest_finish(df, t, alternative)

def run_paired(
        S1_arr: np.ndarray,
        S2_arr: np.ndarray,
        pThr: float=0.05,
        dfThr: float=1,
        eqThr: float=0.5, 
        correction: str='fdr', 
    ):
    """
        Run paired t-test on two sets for the 
        given two samples on a paired configuration.
    """    

    # Calculate sample sizes for each protein in each sample
    n1 = S1_arr.shape[1] - np.sum(np.isnan(S1_arr), axis=1)
    n2 = S2_arr.shape[1] - np.sum(np.isnan(S2_arr), axis=1) 
    
    if not np.array_equal(n1, n2):
        raise ValueError("Paired t-test requires n1 and n2 to be the same!")
    # Save the single sample size
    n = n1
    # Calculate the difference between the two samples
    d = (S1_arr - S2_arr).astype('d')
    # logfold change
    log2fc = np.nanmean(d, axis=1)
    
    # Calculate the t-test p-values
    ttest_pval = ttest_rel_with_na(
        d, 
        n, 
        alternative='two-sided'
    )[1]

    # Apply multiple testing correction
    ttest_pval_corr = multiple_testing_correction(
        ttest_pval,
        correction_type=correction,
        sample_size=None
    )

    # Calculate the tost p-values
    # Test against the upper equivalence boundary
    p_greater = ttest_rel_with_na(
        d + eqThr,
        n,
        alternative='greater'
    )[1]
    # Test against the lower equivalence boundary
    p_less = ttest_rel_with_na(
        d - eqThr,
        n,
        alternative='less'
    )[1]
    # Combine the two p-values
    tost_pval = np.maximum(p_greater, p_less)

    # Corect the greater and less individually
    p_greater_corr = multiple_testing_correction(
        p_greater,
        correction_type=correction,
        sample_size=None
    )
    p_less_corr = multiple_testing_correction(
        p_less,
        correction_type=correction,
        sample_size=None
    )

    # Combine the two p-values
    tost_pval = np.maximum(p_greater, p_less)
    tost_pval_corr = np.maximum(p_greater_corr, p_less_corr)
        
    # Check if the difference is within the equivalence boundary
    is_test_eq = np.abs(log2fc) < eqThr

    # Create combination p and q value arrays 
    comb_pval = np.where(
        is_test_eq, 
        tost_pval, 
        ttest_pval
    )
    comb_pval_corr = np.where(
        is_test_eq,
        tost_pval_corr,
        ttest_pval_corr
    )

    # ttest and tost specific significance conditions
    ttest_signf = (ttest_pval_corr < pThr) & (np.abs(log2fc) > dfThr)
    tost_signf = (tost_pval_corr < pThr) & (np.abs(log2fc) < eqThr)

    # Record boolean for significant p values
    is_signf = np.where(
        is_test_eq,
        tost_signf,
        ttest_signf
    )
    
    # Create test based logging -1 or 1 for p value
    tmp = np.log10(comb_pval)
    logp = np.where(is_test_eq, tmp, -tmp)
    # Create test based logging -1 or 1 for q value
    tmp = np.log10(comb_pval_corr)
    logq = np.where(is_test_eq, tmp, -tmp)

    # Calculate protein status based on significance
    prot_status = np.where(
        is_signf,               # If significant
        np.where(
            is_test_eq,         # If equivalence test
            1.,                 # Eq = 1
            -1.,                # Df = -1
        ),
        0.                      # Not significant = 0
    )

    # Return the results as numpy ndarrays
    return np.stack(
        (
            n1, n2, log2fc, 
            ttest_pval, ttest_pval_corr,
            p_less, p_less_corr,
            p_greater, p_greater_corr,
            tost_pval, tost_pval_corr,
            comb_pval, comb_pval_corr,
            logp, logq, prot_status
        ), 
        axis=1
    )

def qEstimate(
        pv, 
        m=None, 
        verbose=False, 
        lowmem=False, 
        pi0=None
    ):
    
    """
    Estimates q-values from p-values
    source: # https://github.com/nfusi/qvalue
    Args
    =====
    m: number of tests. If not specified m = pv.size
    verbose: print verbose messages? (default False)
    lowmem: use memory-efficient in-place algorithm
    pi0: if None, it's estimated as suggested in Storey and Tibshirani, 2003.
         For most GWAS this is not necessary, since pi0 is extremely likely to be
         1
    Returns
    =====
    qvalues: array of q-values of same size as p-values
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


# TODO: add more correction methods
def multiple_testing_correction(
        pvalues: np.ndarray, 
        correction_type: str="bonferroni", 
        sample_size: int=None
    ):

    """
    Performs multiple testing correction on p-values 
        using p.adjust methods from R
    Args
    =====
    pvalues: array of p-values
    correction_type: type of correction to perform
    sample_size: number of tests performed, 
        if None, it is set to the length of pvalues
    Returns
    =====
    qvalues: array of q-values of same size as p-values
    """

    if correction_type == None:
        return pvalues
    else:
        pvalues = np.array(pvalues)
        if sample_size is None:
            sample_size = pvalues.shape[0]
        qvalues = np.empty(sample_size)

        if correction_type == "bonferroni":
            qvalues = sample_size * pvalues

        elif correction_type == "holm":
            values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
            values.sort()
            for rank, vals in enumerate(values):
                pvalue, i = vals
                qvalues[i] = (sample_size-rank) * pvalue

        elif correction_type == "fdr" or correction_type == "fdr_bh":
            by_descend = pvalues.argsort()[::-1]
            by_orig = by_descend.argsort()
            steps = float(len(pvalues)) / np.arange(len(pvalues), 0, -1)
            q = np.minimum(1, np.minimum.accumulate(steps * pvalues[by_descend]))
            qvalues = q[by_orig]
            
        elif correction_type == "qvalue":
            qvalues = qEstimate(pvalues, m=sample_size)

        return qvalues
  
def run_questvar(
        # two samples
        S1_arr: np.ndarray,
        S2_arr: np.ndarray,
        is_log2: bool = False,
        # Thresholds
        cv_thr: float=0.15,
        p_thr: float=0.05,
        df_thr: float=1,
        eq_thr: float=0.5,
        var_equal: bool=False,
        is_paired: bool=False,
        correction: str='fdr',
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
       Run's QuEStVar's testing of a given pair from directly two np.ndarrays, applies the framework and 
       returns the result dataframes. This is used mainly for benchmarking and testing purposes.
    """

    ## Logical checks
    # Logically check the input parameters
    if df_thr < eq_thr:
        raise ValueError(
            """The equivalence boundary must be smaller 
            than the difference boundary (logFC cutoff)!"""
        )
    # Check if the passed correction is valid
    if correction not in ['bonferroni', 'holm','fdr', 'qvalue', None]:
        raise ValueError(
            """Invalid correction method, 
            must be one of 'bonferroni', 'holm','fdr', 'qvalue'!"""
        )
    # Check if the variables passed for variance and paired are logical
    if is_paired and (not var_equal):
        raise ValueError(
            """Paired test cannot be done without equal variance!"""
        )
    # Ensure both arrays have enough replicates
    if S1_arr.shape[1] < 2 or S2_arr.shape[1] < 2:
        raise ValueError(
            """Both samples must have at least 2 replicates!"""
        )

    # Get index to keep track of proteins
    proteins = np.arange(S1_arr.shape[0])

    # Get the coefficient of variation for each protein
    S1_arr_cv = utils.cv_numpy(S1_arr, axis=1, format='ratio')
    S2_arr_cv = utils.cv_numpy(S2_arr, axis=1, format='ratio')

    # Make the protein selection indicator array
    S1_arr_ps = make_protein_selection_indicator(S1_arr_cv, cv_thr)
    S2_arr_ps = make_protein_selection_indicator(S2_arr_cv, cv_thr)

    # Sum S1 and S2 indicator arrays for selection
    # -2, -1, 0 <- wont be selected
    # TODO: Adding simple low-value imputation
    # 1 <- S1 or S2 can be imputed - select if imputation allowed
    # 2 <- No processing necessary - selected
    S_arr_ps = S1_arr_ps + S2_arr_ps

    # Get the index for 1 (Ms+Qn) and 2s (Qn+Qn)
    subidx = np.where(S_arr_ps >= 2)[0]

    # Subset the dataframes to the shared index
    S1_arr_ready = S1_arr[subidx]
    S2_arr_ready = S2_arr[subidx]
    
    # Log2 transform the data if necessary
    if not is_log2:
        S1_arr_ready = np.log2(S1_arr_ready)
        S2_arr_ready = np.log2(S2_arr_ready)

    # Run the tests
    if is_paired:
        # If the test is paired, then run the paired test
        res = run_paired(
            S1_arr_ready,
            S2_arr_ready,
            pThr=p_thr,
            dfThr=df_thr,
            eqThr=eq_thr,
            correction=correction
        )
    else:
        # If the test is unpaired, then run the unpaired test
        res = run_unpaired(
            S1_arr_ready,
            S2_arr_ready,
            pThr=p_thr,
            dfThr=df_thr,
            eqThr=eq_thr,
            equalVar=var_equal,
            correction=correction
        )

    status_all = np.zeros(len(proteins)) * np.nan
    status_all[subidx] = res[:, -1]

    # Create a dataframe from the results
    res_df = pd.DataFrame(
        res,
        columns=COLS,
        index=proteins[subidx]
    )

    # Create a info dataframe
    info_df = pd.DataFrame(
        {
            "Protein": proteins,
            "S1_Status": S1_arr_ps,
            "S2_Status": S2_arr_ps, 
            "Status": status_all, 
        }
    )

    return res_df, info_df

# Mean Distribution - Normal D
def normal_distribution(
        mu: float,            # Mean
        sd: float,            # Standard deviation
        size: int = 1000,     # Number of values
        is_log2: bool = True, # Is the mean and sd at log2 scale
        seed: int = None,     # Random seed
    ):
    """
        Create a normal distribution with a 
        given mean and standard deviation.
    """
    if seed is not None:
        np.random.seed(seed)
    # Create a normal distribution
    arr = np.random.normal(mu, sd, size)
    # Return to raw values
    if is_log2:
        arr = np.power(2, arr)
    # Return the array
    return arr

# CV Distribution - right skewed
def skewed_distribution(
        mu: float,        # mean
        k: float = 2,     # shape
        theta: float = 1, # scale
        size: int = 1000, # number of values
        seed: int = None, # rng seed
    ):
    """
        Create a skewed distribution with a gamma distribution
        to simulate a CV distribution of technical 
        replicates in a proteomics experiment.
    """
    # If seed is provided
    if seed is not None:
        np.random.seed(seed)

    # Right skewed distribution
    arr = np.random.gamma(
        k,      # shape
        theta,  # scale
        size    # number of values
    )
    # Scale the distribution with the mean
    arr = arr * mu / np.mean(arr)
    # Ensure the min value is 1
    arr = arr - np.min(arr) + 1
    # Return the array
    return arr

# Main function to simulate data
def simulate_data(
        # Data shape
        nPrts: int = 5000,
        nReps: int = 50,
        
        ## Mean intensity distribution
        int_mu: float = 18,      # mean value at log scale
        int_sd: float = 1,       # standard deviation range
        int_log2: bool = True,   # is provided mean and sd at log2 scale
        
        ## Protein CV distribution
        cv_mu: float = 27.5,     # mean value for cv as percentage
        cv_k: float = 2,         # shape for cv
        cv_theta: float = 0.5,   # scale for cv
        cv_pct: bool = True,     # is provided cv_mu at percentage

        # Return as a dataframe
        as_dataframe: bool = True,
        # Random seed
        seed: int = None,
    ):
    """
        Simulate a data in the shape of a proteomics experiment with normally 
        distributed mean values for proteins (rows) and 
        right-skewed CVs calculated from each protein using replicates (columns).
        The replicates are simulated from a log-normal distribution using the mean and the cv.
    """

    # Calculates an intensity distribution representing protein means 
    mean_dist = normal_distribution(
        mu = int_mu,
        sd = int_sd,
        size = nPrts,
        is_log2 = int_log2,
        seed = seed
    )
    # Reshape with len, 1
    mean_dist = (mean_dist[:, np.newaxis])

    # Calculates a CV distribution representing protein CVs
    cv_dist = skewed_distribution(
        mu = cv_mu,
        k = cv_k,
        theta = cv_theta,
        size = nPrts,
        seed = seed
    )
    # Convert the cv_dist to ratio if it is in percentage
    if cv_pct:
        cv_dist = cv_dist / 100
    
    # Reshape with len, 1
    cv_dist = (cv_dist[:, np.newaxis])

    if np.any(mean_dist == 0):
        raise ValueError("Mean values cannot be zero for log-normal distribution.")
    
    # Calculate standard deviations from means and CVs
    sd_dist = mean_dist * cv_dist
    # Calculate parameters for the log-normal distribution
    mu = np.log(mean_dist**2 / np.sqrt(sd_dist**2 + mean_dist**2))
    sigma = np.sqrt(np.log1p(cv_dist**2))
    # Generate log-normal random numbers
    log_normal_random = np.random.lognormal(
        mu, 
        sigma, 
        (nPrts, nReps)
    )

    if as_dataframe:
        sim_data = pd.DataFrame(log_normal_random)
        sim_data.columns = [f"Replicate {i+1}" for i in range(nReps)]
        sim_data.index = [f"Protein {i+1}" for i in range(nPrts)]
        return sim_data
    else:
        return log_normal_random

def calculate_power(
        simulated_sei: float,
        target_sei: float,
    ):
    """
    Calculates power as the difference between target and simulated SEI,
    capped at 1 to show the power from 0 to 1.

    Args:
        simulated_sei: The simulated Sample Equivalence Index (SEI).
        target_sei: The target SEI (adjusted for technical variation).

    Returns:
        The power, ranging from 0 (no detectability) to 1 (full detectability).
    """
    # Error handling
    ## Both should be between 0 and 1
    if not 0 <= simulated_sei <= 1:
        raise ValueError(f"Simulated SEI should be between 0 and 1., got: {simulated_sei}\nmake sure to provide SEI as ratio, not percentage")
    if not 0 <= target_sei <= 1:
        raise ValueError(f"Target SEI should be between 0 and 1., got: {target_sei}\nmake sure to provide SEI as ratio, not percentage")
       
    # Calculate difference, ensuring it's positive
    difference = max(0, target_sei - simulated_sei) 
    
    # Cap the power at 1
    power = min(1, 1 - difference)  

    return power

def simulate_single_run(
        iterations: tuple
    ) -> dict:
    """
        Runs the test for a single iteration for setup parameters.
        to use with multiprocessing.
    """
    cvMean, eqThr, nPrts, nRep, pThr, dfThr, cvThr, corr, i, int_mu, int_sd, int_log2, cv_k, cv_theta = iterations

    # Simulate a data
    sim_data = simulate_data(
        nPrts = nPrts,
        # Times to for two groups each with nRep size
        nReps = nRep*2, 
        int_mu = int_mu,
        int_sd = int_sd,
        int_log2 = int_log2,
        cv_mu = cvMean,
        cv_k = cv_k,
        cv_theta = cv_theta,
        cv_pct = True,
        as_dataframe = False,
        seed = None
    )
    # Randomly split the data to two groups
    sim1, sim2 = np.split(sim_data, 2, axis=1)
    # Run the test
    res_df, info_df = run_questvar(
        S1_arr = sim1,
        S2_arr = sim2,
        is_log2 = False,
        cv_thr = cvThr,
        p_thr = pThr,
        df_thr = dfThr,
        eq_thr = eqThr,
        var_equal = True,
        is_paired = False,
        correction = corr,
    )
    # Calculate the SEI
    calc_SEI = res_df["Status"].value_counts(
        normalize=True
    ).mul(1).reindex(
        [-1, 0, 1,], 
        fill_value=0
    ).loc[1]

    # Create a dictionary with extra information
    return {
        "cvMean": cvMean,
        "eqThr": eqThr,
        "nPrts": nPrts,
        "nRep": nRep,
        "pThr": pThr,
        "dfThr": dfThr,
        "cvThr": cvThr,
        "corr": corr,
        "iteration": i,
        "calc_SEI": calc_SEI
    }

def multiprocess_simulation(
        iterations: list,
        nCores: int = None,
    ) -> pd.DataFrame:
    """
        Run the simulation in parallel with multiprocessing 
        combines the results and returns a DataFrame.

        Args:
            iterations: List of tuples containing the parameters for each iteration.
            nCores: Number of cores to use for multiprocessing. (default: None, uses all available cores)

        Returns:
            pd.DataFrame: DataFrame containing the results of the simulation.
    """
    if nCores is None:
        nCores = os.cpu_count()
    else: 
        if nCores > os.cpu_count():
            nCores = os.cpu_count()
        elif nCores < 1:
            nCores = 1
    
    with Pool(nCores) as pool:
        results = pool.map(simulate_single_run, iterations)

    return pd.DataFrame(results)

def print_power_analysis_results(
    results_df: pd.DataFrame,           # DataFrame containing simulation results
    power: float,                       # Target power for the analysis
    ## Main parameters searched in the simulation
    eqThr: float = None,                # Single Equivalence threshold
    eqBoundaries: np.ndarray = None,    # Array of tested equivalence boundaries
    nRep: int = None,                   # Number of replicates per group
    nReps: np.ndarray = None,           # Array of replicates per group
    cvMean: float = None,               # Mean coefficient of variation
    cvMeans: np.ndarray = None,         # Array of mean coefficients of variation
    ## Other parameters relevant to the simulation
    nPrts: int = None,                  # Number of proteins
    pThr: float = None,                 # Significance threshold
    corr: str = None,                   # Correction method
    dfThr: float = None,                # Threshold for statistical difference
    cvThr: float = None,                # CV threshold
    nRepeat: int = None                 # Number of simulation repeats
):
    """
    Prints the results of the power analysis in a structured format.

    Args:
        results_df: DataFrame containing simulation results (eqThr and calc_power).
        power: Target power for the analysis.
    """

    print("\nPower Analysis Results:")
    print("-----------------------")

    # If the results DataFrame is empty, print a message and return
    if results_df.empty:
        print("No boundaries were tested in the simulation.")
        return

    # Input Parameters
    print("\nInput Parameters:")
    print(f"  - Target power: {power:.2f}")
    if eqThr is not None:
        print(f"  - Symmetrical equivalence boundary: {eqThr:.2f}")
    else:
        if eqBoundaries is None:
            raise ValueError("eqBoundaries must be provided if eqThr is not.")
    # print(f"  - Equivalence Boundaries: {eq_boundaries}")
    if nRep is not None:
        print(f"  - Sample size (rep) per Group: {nRep}")
    else:
        if nReps is None:
            raise ValueError("nReps must be provided if nRep is not.")
    if cvMean is not None:
        print(f"  - Mean Intra-sample CV: {cvMean:.2f}%")
    else:
        if cvMeans is None:
            raise ValueError("cvMeans must be provided if cvMean is not.")
    if nPrts is not None:
        print(f"  - Number of Proteins: {nPrts}")
    if pThr is not None:
        print(f"  - Significance threshold (p-value): {pThr:.2f}")
    if corr is not None:
        print(f"  - P-value correction method: {corr}")
    if dfThr is not None:
        print(f"  - Threshold for statistical difference: {dfThr:.2f}")
    if cvThr is not None:
        print(f"  - CV-based filtering threshold: {cvThr:.2f}")
    if nRepeat is not None:
        print(f"  - Simulation Repeats: {nRepeat}")

    # Simulation Results
    print("\nSimulation Results:")
    if eqThr is None and eqBoundaries is not None:
        print("Symmetrical equivalence boundaries tested =", eqBoundaries)
        if results_df[results_df["calc_power"] >= power].empty:
            print("  - No boundary achieved the desired power.")
        else:
            optimal_eqBoundary = results_df["eqThr"][results_df["calc_power"] >= power].min()
            print(f"  - The minimum symmetrical equivalence boundary for target: {optimal_eqBoundary:.2f}")

    if nRep is None and nReps is not None:
        print("Sample sizes (rep) per group tested =", nReps)
        if results_df[results_df["calc_power"] >= power].empty:
            print("  - No sample size tested achieved the desired power.")
        else:
            optimal_nRep = results_df["nRep"][results_df["calc_power"] >= power].min()
            print(f"  - The minimum sample size per group for target: {optimal_nRep}")

    if cvMean is None and cvMeans is not None:
        print("Mean intra-sample CVs tested =", cvMeans)
        if results_df[results_df["calc_power"] >= power].empty:
            print("  - No mean intra-sample CV tested achieved the desired power.")
        else:
            optimal_cvMean = results_df["cvMean"][results_df["calc_power"] >= power].max()
            print(f"  - The maximum mean intra-sample CV for target: {optimal_cvMean:.2f}")



############################################ ProteoForge ############################################

def _fit_model_and_get_pvalue(
        data: pd.DataFrame,
        formula: str,
        model_type: str,
        weight_col: str = None
    ) -> float:
    """
        Fits a statistical model and returns the p-value for the interaction term.

        Args:
            data (pd.DataFrame): The data to fit the model to.
            formula (str): The formula to use for the model.
            cond_col (str): The name of the condition column.
            model_type (str): The type of model to fit ['ols', 'mqr', 'rlm', 'glm', 'wls'].
                - ols: Ordinary Least Squares
                - mqr: Quantile Regression (Median)
                - rlm: Robust Linear Model
                - glm: Generalized Linear Model
                - wls: Weighted Least Squares
            weight_col (str): The name of the column to use as weights for WLS model.

        Returns:
            float: The p-value for the interaction term.
    """
    if model_type == "ols":
        model = smf.ols(formula=formula, data=data).fit()
    elif model_type == "mqr":
        model = smf.quantreg(formula=formula, data=data, weights=data[weight_col]).fit(q=0.5)
    elif model_type == "rlm":
        model = smf.rlm(formula=formula, data=data, weights=data[weight_col]).fit()
    elif model_type == "glm":
        model = smf.glm(formula=formula, data=data, weights=data[weight_col]).fit()
    elif model_type == "wls":
        if weight_col is None:
            raise ValueError("Weight column should be provided for WLS model.")
        model = smf.wls(formula=formula, data=data, weights=data[weight_col]).fit()
    else:
        raise ValueError("Invalid model type. Must be one of ['ols', 'mqr', 'rlm', 'glm', 'wls']")
    # Extract the interaction term and its p-value
    return float(model.wald_test_terms().pvalues[-1])

def _run_model_single_protein(
        protein_data: pd.DataFrame,
        formula: str,
        peptide_col: str,
        model_type: str,
        weight_col: str = None
    ) -> dict:
    """
        Runs the model for a single protein.

        Args:
            protein_data (pd.DataFrame): Data for a single protein.
            formula (str): The formula to use for the model.
            cond_col (str): The name of the condition column.
            peptide_col (str): The name of the peptide column.
            model_type (str): The type of model to fit ('ols', 'mqr').

        Returns:
            dict: A dictionary of p-values for each peptide in the protein.
    """
    # Get the unique peptides
    unique_peptides = protein_data.index.unique()
    pvalues = {} # Initialize the dictionary to store the p-values
    for peptide in unique_peptides:
        sub_data = protein_data.copy()
        sub_data['allothers'] = 'allothers'
        sub_data.loc[peptide, "allothers"] = peptide
        pval = _fit_model_and_get_pvalue(sub_data, formula, model_type, weight_col)
        pvalues[peptide] = pval
    return pvalues

def run_model(
        long_data: pd.DataFrame,
        cond_col: str = "day",
        intensity_col: str = "ms1adj",
        protein_col: str = "protein_id",
        peptide_col: str = "peptide_id",
        correction_type = "fdr_bh",
        model_type: str = "wls",  # mqr, ols, rlm, glm, wls
        weight_col: str = None
    ) -> pd.DataFrame:
    
    """
        Run a linear model on the data to test the interaction between the condition 
            and the other peptides in protein. The model is fitted for each protein,  
            peptide combination in the data.

        Args:
            long_data (pd.DataFrame): The input data in long format
            cond_col (str): The column with the condition information
            intensity_col (str): The column with the intensity information
            protein_col (str): The column with the protein information
            peptide_col (str): The column with the peptide information
            correction_type (str): The multiple testing correction to use
            model_type (str): The type of model to use (mqr, ols, rlm, glm, wls)
                - mqr: Quantile Regression (Median)
                - ols: Ordinary Least Squares
                - rlm: Robust Linear Model
                - glm: Generalized Linear Model
                - wls: Weighted Least Squares
            weight_col (str): The column to use as weights for WLS model
            
        Returns:
            pd.DataFrame: The results of the model

        Raises:
            ValueError: If model_type is not one of mqr, ols, rlm
    """

    # Create a copy of the input data
    input_data = long_data.copy()
    unique_proteins = input_data[protein_col].unique()
    # Set protein_id as index
    input_data = input_data.set_index(protein_col)

    # Specific formula for the model
    formula = f'{intensity_col} ~ {cond_col} * allothers'

    pdict = {}
    # Iterate over each protein
    for cur_prot in tqdm(unique_proteins, total=len(unique_proteins)):
        # Get the data for the current protein
        protein_data = input_data.loc[cur_prot].set_index(peptide_col).copy()
        # Run the model for the protein
        pvalues = _run_model_single_protein(
            protein_data, 
            formula,  
            peptide_col, 
            model_type, 
            weight_col
        )
        # Store the results in the dictionary
        for peptide, pval in pvalues.items():
            pdict[(cur_prot, peptide)] = pval

    # Expand the dictionary to a DataFrame
    res_df = pd.DataFrame(
        [(protein, peptide, pval) for (protein, peptide), pval in pdict.items()], 
        columns=[protein_col, peptide_col, "pval"]
    )
    # Merge the results with the input data
    res_df = pd.merge(
        input_data.reset_index(), res_df, on=[protein_col, peptide_col], how="left"
    )

    # Create correct for peptide number then overall correction - adj. p-values
    res_df["pval"] = res_df["pval"].astype(float).fillna(1)
    res_df.groupby(protein_col)['pval'].transform(
        # Corrected with unique n of peptides since it is done for each peptide against other
        lambda x: x * (len(x.unique()))
    )
    res_df["pval"] = res_df["pval"].clip(upper=1)
    res_df["adj_pval"] = multiple_testing_correction(
        res_df["pval"], correction_type=correction_type
    )
    # Return the results
    return res_df