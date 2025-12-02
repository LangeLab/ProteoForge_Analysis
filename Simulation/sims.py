import numpy as np
import pandas as pd 

# ======================================================================================
# Global Variables and Settings
# ======================================================================================



# ======================================================================================
# Data Generation Functions
# ======================================================================================

def normal_distribution_with_outliers(
        mu: float,                          
        sd: float,                          
        size: int = 1000,                   
        is_log2: bool = True,               
        outlier_fraction: float = 0.1,      
        outlier_sd_multiplier: float = 2,   
        seed: int = None,                   
    ) -> np.array:
    """
        Create a normal distribution with the provided mean and 
        standard deviation, with additional outliers.

        Args:
            mu (float): Mean of the distribution
            sd (float): Standard deviation of the distribution
            size (int): Number of values to generate
            is_log2 (bool): Is the mean and sd at log2 scale
            outlier_fraction (float): Fraction of outliers
            outlier_sd_multiplier (float): Multiplier for outlier sd
            seed (int): Random seed for reproducibility 
                (default: None, -1 for random)

        Returns:
            np.array: Array of values (shape: (size, ))
    """
    # Set the seed for reproducibility (if needed)
    if seed not in [None, -1]: np.random.seed(seed)
    elif seed == -1: np.random.seed()
    
    # Calculate the number of outliers
    n_outliers = int(size * outlier_fraction)
    n_normal = size - n_outliers
    
    # Create the main normal distribution
    normal_values = np.random.normal(mu, sd, n_normal)
    
    # Create the outlier distribution with a larger standard deviation
    outlier_values = np.random.normal(mu, sd * outlier_sd_multiplier, n_outliers)
    
    # Combine the two distributions
    combined_values = np.concatenate([normal_values, outlier_values])
    
    # Shuffle the combined values to mix outliers with normal values
    np.random.shuffle(combined_values)
    
    # Return to raw values if is_log2 is True
    if is_log2: combined_values = np.power(2, combined_values)
    
    # Return the array
    return combined_values

def lognormal_distribution(
        mu: float,        
        med: float,       
        size: int = 1000, 
        seed: int = None, 
    ):
    """
        Create a lognormal distribution with the provided mean and median. 
        The distribution is scaled to the mean. Another way to create a
        CV distribution observed in proteomics intra-sample CVs. 
        This one allows to setup both mean and median, which can be preferred
        over the gamma distribution in cases where the median is known.

        Args:
            mu (float): Mean of the distribution
            med (float): Median of the distribution
            size (int): Number of values to generate
            seed (int): Random seed for reproducibility
                (default: None, -1 for random)

        Returns:
            np.array: Array of values (shape: (size, ))
    """
    # Set the seed for reproducibility (if needed)
    if seed not in [None, -1]: np.random.seed(seed)
    elif seed == -1: np.random.seed()

    # Calculate sigma (std dev of underlying normal)
    sigma = np.sqrt(np.log(mu / med) * 2)

    # Calculate mu (mean of underlying normal)
    mu = np.log(med)

    # Generate lognormal samples
    return np.random.lognormal(mu, sigma, size)

def generate_peptide_counts(
        n_proteins: int,
        min_peptides: int,
        max_peptides: int,
        alpha: float = 1.5,
        beta: float = 5,
        seed: int = None,
    ) -> dict:
    """
        Generating peptide number for number of proteins based on a beta distribution 
            with the provided alpha and beta parameters, and determined minimum and 
            maximum peptide values.

        Args:
            n_proteins (int): Number of proteins
            min_peptides (int): Minimum number of peptides
            max_peptides (int): Maximum number of peptides
            alpha (float): Alpha parameter of the beta distribution
            beta (float): Beta parameter of the beta distribution

        Returns:
            dict: Dictionary of protein: number of peptides
    """
    # Set the seed for reproducibility (if needed)
    if seed not in [None, -1]: np.random.seed(seed)
    elif seed == -1: np.random.seed()

    # Generate the beta distribution
    beta_values = np.random.beta(alpha, beta, n_proteins)
    
    # Scale the beta values to the range [min_peptides, max_peptides]
    scaled_values = min_peptides + (max_peptides - min_peptides) * beta_values
    
    # Clip the values to ensure they are within the range and convert to integers
    peptide_counts = np.clip(scaled_values, min_peptides, max_peptides).astype(int)
    
    # Create the dictionary of protein: peptide counts
    pep_mapping = {
        i: peptide_counts[i] for i in range(n_proteins)
    }
    
    return pep_mapping

def generate_replicates(
        mean_arr: np.array,         
        cv_arr: np.array,           
        nReps: int,                 
        meanScale: str = "raw",         
        cvType: str = "percent",    
        randomizeCV: bool = False,  
        as_dataframe: bool = False, 
        seed: int = None            
    ) -> np.ndarray:
    """
        Generates technical or biological replicates for given proteins based on the
            provided mean and CV values. The mean values can be at different scales
            (raw, log2, log10, log) and the CV values can be in percentage or ratio.
            The CV values can also be randomized before applying to the mean values,
            to simulate different scenarios.
        
        Args:
            mean_arr (np.array): Mean of proteins of the sample as vector
            cv_arr (np.array): Coefficient of Variation of proteins as vector
            nReps (int): Number of technical replicates to simulate
            meanScale (str): Scale of the mean values 
                (default: raw, [log2, log10, log, raw])
            cvType (str): Type of the CV values [percent, ratio]
            randomizeCV (bool): Randomize the CV values before applying
            as_dataframe (bool): Return as a dataframe
            seed (int): Random seed for reproducibility
                (default: None, -1 for random)

        Returns:
            np.array: Array of values (shape: (nProteins, nReps)) 
            or
            pd.DataFrame: DataFrame of values (shape: (nProteins, nReps))

        Raises:
            ValueError: If the meanScale is not one of the supported scales
            ValueError: If the cvType is not one of the supported types
            ValueError: If the mean and cv arrays do not have the same shape
            ValueError: If the mean values are zero
    
    """
    # Set the seed for reproducibility (if needed)
    if seed not in [None, -1]: np.random.seed(seed)
    elif seed == -1: np.random.seed()
    
    scale_dict = { "raw": 1, "log2": 2, "log10": 10, "log": np.e, }
    # Check the meanScale
    if meanScale not in scale_dict.keys():
        raise ValueError("meanScale should be one of: raw, log2, log10, log")
    elif meanScale == "raw": mean_arr = mean_arr
    else: # If the mean values not (raw), convert them to raw
        mean_arr = np.power(scale_dict[meanScale], mean_arr)

    # Check the cvType
    if cvType not in ["percent", "ratio"]:
        raise ValueError("cvType should be one of: percent, ratio")

    # Check the shape of the mean and cv arrays
    if mean_arr.shape != cv_arr.shape:
        raise ValueError("Mean and CV arrays should have the same shape.")    
    else:
        nPrts = mean_arr.shape[0]

    # Reshape the mean array
    mean_arr = mean_arr.reshape(-1, 1)
    # if cvType is percent, convert it to ratio
    if cvType == "percent": cv_arr = cv_arr / 100
    # Reshape the cv array
    cv_arr = cv_arr.reshape(-1, 1)
    
    # Check if the mean values are zero
    if np.any(mean_arr == 0):
        raise ValueError("Mean values cannot be zero for log-normal distribution.")
    
    # If randomizeCV is True, randomize the CV values
    if randomizeCV: cv_arr = np.random.permutation(cv_arr)

    # Calculate standard deviations from means and CVs
    sd_dist = mean_arr * cv_arr

    # Calculate parameters for the log-normal distribution
    mu = np.log(mean_arr**2 / np.sqrt(sd_dist**2 + mean_arr**2))
    sigma = np.sqrt(np.log1p(cv_arr**2))

    log_normal_random = np.random.lognormal( mu, sigma, (nPrts, nReps) )

    # Return the log-normal random values as df or matrix
    if as_dataframe: return pd.DataFrame(log_normal_random)
    else: return log_normal_random

def generate_peptide_level(
        data: np.ndarray,
        pepN_cnts: dict,
        is_log2: bool = False,

        ## Simulate Replicates
        repStd: tuple = (0.1, 0.5),
        ## Outliers
        outlier_fraction: float = 0.01,
        outlier_multiplier: float = 0.5,
        ## Noise
        add_noise: bool = True,
        noise_sd: float = 0.01,
        ## DF Building
        as_dataframe: bool = True,
        protein_col: str = "Protein",
        peptide_col: str = "Peptide",
        pep_str: str = "pep_",
        rep_str: str = "rep_",
        pro_str: str = "pro_",
        seed: int = None,
    )-> pd.DataFrame:
    """
        Generate peptide level data for the given a protein data matrix 
            and protein:peptide count mapping. The peptide level data is
            generated by simulating technical replicates for each protein
            and then aggregating the peptide level data based on the
            provided protein:peptide count mapping. Outliers and noise
            can also be added to the data.

        Args:
            data (np.ndarray): Protein data matrix
            pepN_cnts (dict): Protein:Peptide count mapping
            is_log2 (bool): Is the data in log2 scale
            repStd (tuple): Standard deviation of replicates
            outlier_fraction (float): Fraction of outliers
            outlier_multiplier (float): Multiplier for outlier values
            add_noise (bool): Add noise to the data
            noise_sd (float): Standard deviation of the noise
            as_dataframe (bool): Return as a dataframe
            protein_col (str): Column name for protein names
            peptide_col (str): Column name for peptide names
            pep_str (str): Prefix for peptide names
            rep_str (str): Prefix for replicate names
            pro_str (str): Prefix for protein names
            seed (int): Random seed for reproducibility
                (default: None, -1 for random)

        Returns:
            pd.DataFrame: Peptide level data (shape: (nPeptides, nReplicates))
            or
            np.ndarray: Peptide level data (shape: (nPeptides, nReplicates))

        Raises:
            ValueError: If the input data is not a non-empty DataFrame
            ValueError: If the peptide count mapping is not a non-empty dictionary
            ValueError: If the repStd values are not a tuple of two values
            ValueError: If the repStd values are not integers or floats
            ValueError: If the repStd values are not in increasing order
            ValueError: If the outlier_fraction is not an integer or float
            ValueError: If the outlier_fraction is not between 0 and 1
            ValueError: If the outlier_multiplier is not an integer or float
            ValueError: If the noise_sd is not an integer or float
    """

    ## Validations
    # Check the input data
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Input data should be a non-empty DataFrame.")
    if not isinstance(pepN_cnts, dict) or not pepN_cnts:
        raise ValueError("Peptide count mapping should be a non-empty dictionary.")

    # std deviation of replicates
    if not isinstance(repStd, tuple) or len(repStd) != 2:
        raise ValueError("repStd should be a tuple of two values.")
    if not all([isinstance(v, (int, float)) for v in repStd]):
        raise ValueError("repStd values should be integers or floats.")
    if repStd[0] >= repStd[1]:
        raise ValueError("repStd values should be in increasing order.")
    
    # outlier fraction
    if not isinstance(outlier_fraction, (int, float)):
        raise ValueError("outlier_fraction should be an integer or float.")
    if not 0 <= outlier_fraction <= 1:
        raise ValueError("outlier_fraction should be between 0 and 1.")
    
    # outlier multiplier
    if not isinstance(outlier_multiplier, (int, float)):
        raise ValueError("outlier_multiplier should be an integer or float.")
    
    # noise standard deviation
    if not isinstance(noise_sd, (int, float)):
        raise ValueError("noise_sd should be an integer or float.")
    
    ## Data Generation
    # Set the seed for reproducibility (if needed)
    if seed not in [None, -1]: np.random.seed(seed)
    elif seed == -1: np.random.seed()

    # Convert the data to log2 scale if needed
    if not is_log2:
        input_data = np.log2(data).values
    else:
        input_data = data.values
    
    # Get the number of peptides per protein and the number of replicates
    n_proteins = len(pepN_cnts)
    n_replicates = input_data.shape[1]
    n_peptides = sum(pepN_cnts.values())

    # Generate replicate means and standard deviations for all proteins
    repMeans = np.array([input_data[c] for c in pepN_cnts.keys()])
    repStds = np.random.uniform(repStd[0], repStd[1], size=(n_proteins, n_replicates))

    # Generate a normal distribution of peptide values
    pep_data_protein = np.random.normal(
        loc=repMeans.repeat(list(pepN_cnts.values()), axis=0),
        scale=repStds.repeat(list(pepN_cnts.values()), axis=0),
        size=(n_peptides, n_replicates)
    )

    post_fix = False
    # Introduce outliers
    if outlier_fraction > 0:
        n_outliers = int(outlier_fraction * n_peptides * n_replicates)
        outlier_indices = np.random.choice(
            pep_data_protein.size, n_outliers, replace=False
        )
        np.put(
            pep_data_protein,  outlier_indices, 
            np.take(pep_data_protein, outlier_indices) + outlier_multiplier
        )
        post_fix = True

    # Add noise to the data
    if add_noise:
        pep_data_protein += np.random.normal(0, noise_sd, size=pep_data_protein.shape)
        post_fix = True

    # Post-adjustment to ensure the mean remains close to the protein mean
    if post_fix:
        column_means = pep_data_protein.mean(axis=0)
        adjustment_factors = repMeans.mean(axis=0) / column_means
        pep_data_protein *= adjustment_factors

    if not is_log2:
        pep_data_protein = np.power(2, pep_data_protein)
        
    if as_dataframe: # Build the dataframe and return
        col_names = [f"{rep_str}{i+1}" for i in range(n_replicates)]
        protein_names = [f"{pro_str}{i+1}" for i in range(n_proteins)]
        peptide_names = np.concatenate([
            [
                f"{pep_str}{i+1}" for i in range(pepN_cnts[protein])
            ] for protein in pepN_cnts.keys()
        ])
        pep_data = pd.DataFrame(pep_data_protein, columns=col_names)
        pep_data[protein_col] = np.repeat(protein_names, list(pepN_cnts.values()))
        pep_data[peptide_col] = peptide_names
        return pep_data
    
    else: # Return as a matrix (styling can be done later)
        return pep_data_protein
    
def generate_condition_mappers(
        n_condition: int,
        n_replicates: int,
        condition_shifts: list,
        # 
        control_name: str = "control",
        condition_suffix: str = "cond-",
    ):
    """
    Build condition names, their shifts, sample map, and palettes.

    Args:
        n_condition (int): Number of conditions (excluding control).
        n_replicates (int): Number of replicates per condition.
        condition_shifts (list): List of shifts for each condition (excluding control).

    Returns:
        dict: condition_sample_map
        dict: condition_shifts
    """
    ## Validations

    # Check if the number of conditions and shifts match
    if n_condition != len(condition_shifts):
        raise ValueError("Number of conditions and shifts do not match!")
    # Check if the number of replicates is greater than 0
    if n_replicates < 1:
        raise ValueError("Number of replicates should be greater than 0!")

    # Build condition names and their shifts
    conditions = [control_name] + [
        f"{condition_suffix}{i}" for i in range(1, n_condition + 1)
    ]
    condition_shifts = [0] + condition_shifts
    condition_shifts = {
        condition: shift for condition, shift in zip(conditions, condition_shifts)
    }

    condition_sample_map = {
        control_name: [f'{control_name}-{i}' for i in range(1, n_replicates + 1)],
    }

    for condition in conditions[1:]:
        condition_sample_map[condition] = [
            f'{condition}-{i}' for i in range(1, n_replicates + 1)
        ]
        # condition_pal[condition] = tmp.pop()


    return ( condition_sample_map, condition_shifts )

def generate_complete_data(
        data: pd.DataFrame,          
        condition_shifts: dict,      
        condition_sample_map: dict,  
        shift_scale: float = .5,     
        is_log2: bool = False,       
        add_noise: bool = True,      
        noise_sd: float = 0.1,       

        protein_col: str = "Protein",
        peptide_col: str = "Peptide",
       
        seed: int = None, 
    ):
    """
        Generate complete data based on the provided reference data 
            and condition shifts for conditions. The data is generated
            by adding the shifts to the reference data and then adding
            noise to the data. The data is then returned as a DataFrame.

        Args:
            data (pd.DataFrame): Reference data for generating the data
            condition_shifts (dict): Condition shifts for the data
            condition_sample_map (dict): Condition: [Samples] mapping
            shift_scale (float): Std for the shifts distribution
            is_log2 (bool): Is the data in log2 scale
            add_noise (bool): Add noise to the data
            noise_sd (float): Standard deviation of the noise
            protein_col (str): Column name for protein names
            peptide_col (str): Column name for peptide names
            seed (int): Random seed for reproducibility
                (default: None, -1 for random)
        
        Returns:
            pd.DataFrame: Complete data 
                (shape: (nProteins, nSamples=nConditions*nReplicates))

    """
    # Set the seed for reproducibility (if needed)
    if seed not in [None, -1]: np.random.seed(seed)
    elif seed == -1: np.random.seed()

    index_data = data[[protein_col, peptide_col]]
    data = data.drop(columns=[protein_col, peptide_col])
    # Convert the data to log2 scale if needed
    if not is_log2:
        input_data = np.log2(data).values
    else:
        input_data = data.values

    # Create a shift matrix  shifts x data.shape[0]
    # Uses the shift as mean and shift_scale as std to generate the shifts
    shift_matrix = np.random.normal(
        loc=np.array(list(condition_shifts.values())), 
        scale=shift_scale, 
        size=(input_data.shape[0], len(condition_shifts))
    )

    # Create the complete data matrix
    complete_data = np.zeros(
        (input_data.shape[0], len(condition_sample_map) * input_data.shape[1])
    )
    colnames = []
    for i, (k,v) in enumerate(condition_sample_map.items()):
        # Get copy and place the shift and update the data for the condition
        tmp_data = input_data + shift_matrix[:, i].reshape(-1, 1)
        colnames += v
        if add_noise:
            tmp_data += np.random.normal(0, noise_sd, size=tmp_data.shape)
        # Update the complete data matrix
        complete_data[:, i*input_data.shape[1]:(i+1)*input_data.shape[1]] = tmp_data

    if not is_log2:
        complete_data = np.power(2, complete_data)    

    # Convert the complete data to a DataFrame
    complete_data = pd.DataFrame(
        complete_data, columns=colnames
    )
    complete_data = pd.concat(
        [ index_data, complete_data ], 
        axis=1
    ).set_index([protein_col, peptide_col])
    
    return complete_data

def proteomics_data(
        n_proteins: int,
        n_peptides: tuple,
        n_conditions: int,
        n_replicates: int,
        condition_shifts: list,
        mu_mean: float = 20,
        sd_mean: float = 2,
        mu_cv: float = 10,
        med_cv: float = 8,
        repStd: tuple = (0.01, 0.05),
        outlier_fraction: float = 0.0001,
        outlier_multiplier: float = 0.01,
        add_noise: bool = True,
        noise_sd: float = 0.1,
        shift_scale: float = 0.1,
        is_log2: bool = False,
        control_name: str = "control",
        condition_suffix: str = "cond",
        seed: int = None
    ) -> pd.DataFrame:
    """
    Simulate a complete proteomics dataset with multiple conditions and replicates.

    Returns:
        pd.DataFrame: Simulated peptide-level data with all conditions and replicates.
    """
    # 1. Protein mean values
    mean_values = normal_distribution_with_outliers(
        mu=mu_mean,
        sd=sd_mean,
        size=n_proteins,
        is_log2=False,
        outlier_fraction=0.10,
        outlier_sd_multiplier=1.0,
        seed=seed
    )
    mean_values = 2 ** mean_values

    # 2. Protein CV values
    cv_values = lognormal_distribution(
        mu=mu_cv,
        med=med_cv,
        size=n_proteins,
        seed=seed
    )

    # 3. Generate protein-level replicates (reference/control)
    control_data = generate_replicates(
        mean_values,
        cv_values,
        meanScale="raw",
        cvType="percent",
        nReps=n_replicates,
        randomizeCV=True,
        as_dataframe=True,
        seed=seed
    )

    # 4. Generate number of peptides per protein
    pepN_cnts = generate_peptide_counts(
        n_proteins=n_proteins,
        min_peptides=n_peptides[0],
        max_peptides=n_peptides[1],
        alpha=0.5,
        beta=3,
        seed=seed
    )

    # 5. Generate peptide-level data for the control
    pep_data = generate_peptide_level(
        control_data,
        pepN_cnts,
        is_log2=is_log2,
        repStd=repStd,
        outlier_fraction=outlier_fraction,
        outlier_multiplier=outlier_multiplier,
        add_noise=add_noise,
        noise_sd=noise_sd,
        seed=seed
    )

    # 6. Generate condition/sample/color mappers
    condition_sample_map, cond_shifts = generate_condition_mappers(
        n_condition=n_conditions - 1,
        n_replicates=n_replicates,
        condition_shifts=condition_shifts[1:],
        control_name=control_name,
        condition_suffix=condition_suffix,
    )

    # 7. Add the control shift back to the dict for complete_data
    cond_shifts = {
        control_name: 0, 
        **{k: v for k, v in zip(
            list(cond_shifts.keys())[1:], list(cond_shifts.values())[1:]
        )}
    }

    # 8. Generate the complete peptide-level data with all conditions
    complete_data = generate_complete_data(
        pep_data,
        condition_shifts=cond_shifts,
        condition_sample_map=condition_sample_map,
        shift_scale=shift_scale,
        is_log2=is_log2,
        add_noise=False,
        noise_sd=noise_sd,
        seed=seed
    )

    return complete_data, condition_sample_map, cond_shifts

# ======================================================================================
# Perturbation Functions
# ======================================================================================

def generate_perturb_values(
        nPro_to_perturb: int, 
        nCond: int,
        pertMag_range: tuple,
        direction: str ='random'
    ):
    """
        Generate perturbation values with specified direction variations.

        Args:
            nPro_to_perturb (int): Number of proteins to perturb.
            nCond (int): Number of conditions.
            pertMag_range (tuple): Range of magnitudes for perturbation.
            direction (str): Direction variation type
                (default: 'random', [same, different, random])
        Returns:
            np.ndarray: Perturbation values.
    """
    # Generate random perturbation magnitudes
    perturb_values = np.random.uniform(
        pertMag_range[0], pertMag_range[1], 
        size=(nPro_to_perturb, nCond)
    )

    if direction == 'random':
        # Randomize the direction of the perturbation (positive or negative)
        perturb_values = np.where(
            np.random.rand(nPro_to_perturb, 1) > 0.5, 
            perturb_values, -perturb_values
        )
    elif direction == 'same':
        # Apply the same direction for all conditions
        direction_sign = np.where(
            np.random.rand(nPro_to_perturb, 1) > 0.5, 1, -1
        )
        perturb_values *= direction_sign
    elif direction == 'different':
        # Apply different directions for each condition
        signs = np.ones((nPro_to_perturb, nCond))
        half_cond = nCond // 2
        signs[:, :half_cond] = -1
        for row in signs:
            np.random.shuffle(row)
        perturb_values *= signs
    else:
        raise ValueError(
            "Invalid direction type. Choose from 'random', 'same', or 'different'."
        )

    return perturb_values

def perturb_data(
        data: pd.DataFrame,
        perturb_conds: list,
        pertMag_range: tuple,
        nPro_to_perturb: int,
        nPep_to_perturb: float,
        proteins_to_perturb: list,
        condition_sample_map: dict,
        perturb_name: str = 'randomPep', 
        perturb_dir_setup: str = 'same',
        seed: int = None,
    ):
    """ 
        Main function to add perturbations to the data based on the provided
        perturbation settings. The perturbations can be of different magnitudes, 
        types, and distributions, and can be applied to single, multiple 
        condiitons across single or multiple peptides. The additional noise can 
        be added to the data to simulate the real-world complexities.
    """
    # Set the seed for reproducibility (if needed)
    if seed not in [None, -1]: np.random.seed(seed)
    elif seed == -1: np.random.seed()
    # Setup the data
    tmp = np.log2(data).copy()
    original_index = data.index
    perturb_data = tmp.loc[pd.IndexSlice[proteins_to_perturb, :]]
    unchanged_data = tmp.drop(proteins_to_perturb)

    perturbation_records = []

     # The 'twoPep' case has a unique way of generating perturbation values
    perturb_values_fixed = None
    if nPep_to_perturb >= 2 and nPep_to_perturb == int(nPep_to_perturb):
        perturb_values_fixed = generate_perturb_values(
            nPro_to_perturb=nPro_to_perturb,
            nCond=int(nPep_to_perturb),
            pertMag_range=pertMag_range,
            direction=perturb_dir_setup,
        )
        perturb_conditions_fixed = np.random.choice(
            perturb_conds, nPro_to_perturb
        )

    # This loop is necessary because the number of peptides to change ('n')
    # can be different for each protein.
    for i, protein in enumerate(proteins_to_perturb):
        cur_data = perturb_data.loc[protein]
        total_peptides_in_protein = len(cur_data)
        
        # Determine the number of peptides (n) to perturb for this protein
        if nPep_to_perturb >= 2 and nPep_to_perturb == int(nPep_to_perturb):
            n = int(nPep_to_perturb)
        elif 0 < nPep_to_perturb < 1:
            n = int(
                np.ceil(nPep_to_perturb * total_peptides_in_protein)
            ) if perturb_name == "halfPlusPep" else int(
                np.floor(nPep_to_perturb * total_peptides_in_protein)
            )
        elif nPep_to_perturb == -1:
            n = int(np.random.uniform(0.1, 0.5) * total_peptides_in_protein)
        else:
            continue # Skip if nPep_to_perturb is invalid

        n = max(2, n) # Ensure at least 2 peptides are perturbed
        n = min(n, total_peptides_in_protein) # Cannot exceed available peptides

        pep_indices = np.random.choice(cur_data.index, n, replace=False)

        # Determine perturbation details (magnitude, direction, conditions)
        if perturb_values_fixed is not None:
            pert_shifts = perturb_values_fixed[i]
            conds = np.repeat(perturb_conditions_fixed[i], n)
        else:
            pert_mag = np.random.uniform(pertMag_range[0], pertMag_range[1], n)
            pert_dir = np.repeat(
                np.random.choice([-1, 1]), n
            ) if perturb_name != "randomPep" else np.random.choice(
                [-1, 1], n, replace=True
            )
            pert_shifts = pert_mag * pert_dir
            conds = np.random.choice(perturb_conds, n, replace=True)

        # Add this protein's perturbation plan to our list of records
        for j in range(n):
            perturbation_records.append({
                "Protein": protein,
                "Peptide": pep_indices[j],
                "pertCondition": conds[j],
                "pertShift": pert_shifts[j],
            })

    # Create the final perturbation map from the records
    pert_map_df = pd.DataFrame(perturbation_records)

    # --- 3. Apply All Perturbations Using Vectorized Operations ---
    if not pert_map_df.empty:
        # To vectorize, we "un-pivot" the data from wide to long format.
        original_cols = perturb_data.columns
        perturb_data_long = perturb_data.reset_index().melt(
            id_vars=['Protein', 'Peptide'], 
            var_name='Sample', 
            value_name='Intensity'
        )

        # Create a reverse map from Sample to Condition for merging
        sample_to_cond_map = {
            sample: cond for cond, samples in condition_sample_map.items() for sample in samples
        }
        perturb_data_long['pertCondition'] = perturb_data_long['Sample'].map(sample_to_cond_map)

        # Merge the long data with our perturbation plan.
        # This aligns the correct shift with the correct peptide and condition.
        merged_df = pd.merge(
            perturb_data_long,
            pert_map_df[['Protein', 'Peptide', 'pertCondition', 'pertShift']],
            on=['Protein', 'Peptide', 'pertCondition'],
            how='left'
        )

        # Apply the shift. NaNs (where no perturbation was planned) become 0.
        merged_df['pertShift'].fillna(0, inplace=True)
        merged_df['Intensity'] += merged_df['pertShift']

        # Pivot the data back to its original wide format
        perturb_data = merged_df.pivot_table(
            index=['Protein', 'Peptide'], columns='Sample', values='Intensity'
        )[original_cols] # Restore original column order

    # --- 4. Finalize and Store Results ---
    perturbed_data = pd.concat(
        [perturb_data, unchanged_data], axis=0
    ).reindex(original_index)
    perturbed_data = np.power(2, perturbed_data)

    return perturbed_data, pert_map_df

# ======================================================================================
# Amputation and Imputation Functions
# ======================================================================================

def amputation(
        data: pd.DataFrame,
        unique_proteins: np.ndarray,
        proteins_to_perturb: np.ndarray,
        condition_shifts: dict,
        condition_sample_map: dict,
        n_amputate_1: int = 50,
        n_amputate_2: int = 50,
        n_amputate_3: int = 50,
        missing_rate: float = 0.30,
        seed: int = None
    ) -> pd.DataFrame:
    """
        Introduce missing values into the simulated data.

        Args:
            data (pd.DataFrame): The simulated data.
            unique_proteins (np.ndarray): Array of unique proteins.
            proteins_to_perturb (np.ndarray): Array of proteins to perturb.
            condition_shifts (dict): Dictionary of condition shifts.
            condition_sample_map (dict): Dictionary mapping conditions to samples.
            seed (int): Random seed for reproducibility.
            n_amputate_1 (int): Number of non-perturbed proteins to amputate in the first step.
            n_amputate_2 (int): Number of non-perturbed proteins to amputate in the second step.
            n_amputate_3 (int): Number of perturbed proteins to amputate.
            missing_rate (float): Rate of missing values to introduce.

        Returns:
            pd.DataFrame: DataFrame with missing values introduced.
    """
     # Set the seed for reproducibility (if needed)
    if seed not in [None, -1]: np.random.seed(seed)
    elif seed == -1: np.random.seed()
    missing_data = data.copy()
    non_perturbed_proteins = np.setdiff1d(unique_proteins, proteins_to_perturb)
    
    # Step 1: Select random proteins to amputate and introduce missing values
    proteins_to_amputate = np.random.choice(
        non_perturbed_proteins, n_amputate_1, replace=False
    )
    missing_data.loc[proteins_to_amputate] = missing_data.loc[proteins_to_amputate].mask(
        np.random.rand(*missing_data.loc[proteins_to_amputate].shape) < missing_rate
    )
    
    # Step 2: Select another set of non-perturbed proteins and remove values in one condition
    proteins_to_amputate = np.setdiff1d(non_perturbed_proteins, proteins_to_amputate)
    proteins_to_amputate = np.random.choice(
        proteins_to_amputate, n_amputate_2, replace=False
    )
    conds_to_amputate = np.random.choice(
        list(condition_shifts.keys()), n_amputate_2, replace=True
    )
    for i, protein in enumerate(proteins_to_amputate):
        cur_data = missing_data.loc[protein]
        cur_peptide = np.random.choice(cur_data.index, 1)[0]
        samples = condition_sample_map[conds_to_amputate[i]]
        missing_data.loc[(protein, cur_peptide), samples] = np.nan
    
    # Step 3: Select perturbed proteins and introduce missing values
    proteins_to_amputate = np.random.choice(
        proteins_to_perturb, n_amputate_3, replace=False
    )
    missing_data.loc[proteins_to_amputate] = missing_data.loc[proteins_to_amputate].mask(
        np.random.rand(*missing_data.loc[proteins_to_amputate].shape) < missing_rate
    )
    
    return missing_data

def calculate_downshift(
        data: pd.DataFrame, 
        shiftMag: float = 1, 
        lowPct: float = 0.1
    ) -> np.ndarray:
    """
        Calculate the downshifted values for the given data based on the provided
            shift magnitude and low percentile. The downshifted values are calculated
            by shifting the low values below the low percentile by the shift magnitude.

        Args:
            data (pd.DataFrame): Input data
            shiftMag (float): Magnitude of the downshift
            lowPct (float): Percentile to consider as low values

        Returns:
            np.ndarray: Downshifted values as a matrix
            
    """
    # Flatten the data and remove NaN values
    flatten_data = data.values.flatten()
    flatten_data = flatten_data[~np.isnan(flatten_data)]
    
    # Find the low value threshold
    lowVal = np.percentile(flatten_data, lowPct)
    
    # Get the values below the low value threshold
    lowDist = flatten_data[flatten_data < lowVal]
    
    # Return the downshifted values
    return (lowDist - shiftMag)

def downshifted_imputation(
        data: pd.DataFrame,
        condition_sample_map: dict,
        is_log2: bool = False,
        shiftMag: float = 1,
        lowPct: float = 0.1,
        minValue: float = None,
        impute_all: bool = True,
        seed: int = None,
    ) -> pd.DataFrame:
    """
        Perform downshifted imputation on the given data based on the provided
            condition sample map. The imputation is done by calculating the mean
            of the downshifted values below a certain percentile and using it to
            impute the missing values.

        Args:
            data (pd.DataFrame): Input data with missing values
            condition_sample_map (dict): Condition: [Samples] mapping
            is_log2 (bool): Is the data in log2 scale
            shiftMag (float): Magnitude of the downshift
            lowPct (float): Percentile to consider as low values
            minValue (float): Minimum value for the data if needed a whole data shift 
                (default: None)
            impute_all (bool): Impute all missing values at once
                (default: False) imputes all missing or only complete missing in condition
            seed (int): Random seed for reproducibility
                (default: None, -1 for random)

        Returns:
            pd.DataFrame: Data with imputed values
    """
    # Set the seed for reproducibility (if needed)
    if seed not in [None, -1]: np.random.seed(seed)
    elif seed == -1: np.random.seed()

    # Convert the data to log2 scale if needed
    if not is_log2:
        imputed_data = np.log2(data).copy()

    # Imputation loop (Updated with vectorization)
    for k, v in condition_sample_map.items():
        isMiss = imputed_data[v].isna()
        
        sd = imputed_data[v].dropna(how="all").std(axis=1)
        
        if impute_all:
            # Impute all missing values
            anyMiss = isMiss.sum(axis=1) > 0
            sdDist = np.random.choice(sd, size=anyMiss.sum(), replace=True)
            means = calculate_downshift(imputed_data[v], shiftMag, lowPct).mean()
            imputed_values = np.random.normal(means, sdDist[:, np.newaxis], (anyMiss.sum(), len(v)))
            imputed_data.loc[anyMiss, v] = imputed_values
        else:
            # Impute only completely missing values
            fullMiss = isMiss.sum(axis=1) == len(v)
            sdDist = np.random.choice(sd, size=fullMiss.sum(), replace=True)
            means = calculate_downshift(imputed_data[v], shiftMag, lowPct).mean()
            imputed_values = np.random.normal(means, sdDist[:, np.newaxis], (fullMiss.sum(), len(v)))
            imputed_data.loc[fullMiss, v] = imputed_values

    if not is_log2:
        imputed_data = np.power(2, imputed_data)

    if minValue is not None:
        imputed_data = imputed_data + (minValue - imputed_data.min().min())

    return imputed_data


# ======================================================================================
# Generate Input Data Functions
# ======================================================================================

def assign_groups_based_on_perturbation(
        data: pd.DataFrame, 
        protein_col: str = 'Protein', 
        peptide_col: str = 'Peptide', 
        condition_col: str = 'pertCondition', 
        shift_col: str = 'pertShift', 
        pertPFG_col: str = 'pertPFG'
    ) -> pd.DataFrame:
    """

    """
    unique_proteins = data[protein_col].unique()
    
    # Initialize pertPFG_col with -1
    data[pertPFG_col] = -1
    
    def ensure_list(x):
        if isinstance(x, list):
            return x
        else:
            return [x]

    for protein in unique_proteins:
        cur_data = data[data[protein_col] == protein].copy()
        if len(cur_data) == 1: 
            continue  # Skip if only one peptide is perturbed
        else:
            # Ensure that pertCondition and pertShift are always treated as lists
            cur_data[condition_col] = cur_data[condition_col].apply(ensure_list)
            cur_data[shift_col] = cur_data[shift_col].apply(ensure_list)
            
            # Flatten the lists and get unique conditions and shifts
            unique_conds = np.unique([item for sublist in cur_data[condition_col] for item in sublist])
            unique_shifts = np.unique([item for sublist in cur_data[shift_col] for item in sublist])
            
            # If all conditions and shifts are the same
            if len(unique_conds) == 1 and len(unique_shifts) == 1:
                cur_data.loc[cur_data.index, pertPFG_col] = 0
            else:
                shift_signs = np.sign(
                    [item for sublist in cur_data[shift_col] for item in sublist]
                )
                if len(np.unique(shift_signs)) == 1:
                    cur_data.loc[cur_data.index, pertPFG_col] = 1
                else:
                    group_counter = 0
                    for idx, row in cur_data.iterrows():
                        if cur_data.loc[idx, pertPFG_col] != -1:
                            continue
                        shift_signs = np.sign(row[shift_col])
                        group_found = False
                        for g_idx, g_row in cur_data.iterrows():
                            if g_idx == idx or cur_data.loc[g_idx, pertPFG_col] != -1:
                                continue
                            g_shift_signs = np.sign(g_row[shift_col])
                            if np.array_equal(shift_signs, g_shift_signs):
                                cur_data.loc[[idx, g_idx], pertPFG_col] = group_counter
                                group_found = True
                        if group_found:
                            group_counter += 1
        # Update the data
        data.loc[cur_data.index, pertPFG_col] = cur_data[pertPFG_col]

    return data.set_index([protein_col, peptide_col])

def build_test_data(
        data,
        condition_sample_map,
        perturbation_map,
        proteins_to_perturb,
        missing_data = None,
    ):   
    
    ### Build the Test Data
    # Create a normalized version of the data
    log2_data = np.log2(data)
    # Center and scale the data
    centered_data = (log2_data - log2_data.mean()) / log2_data.std()
    # Calculate the mean of each peptide across the day1 samples
    cntrPepMean = centered_data[condition_sample_map["control"]].mean(axis=1)
    # Substract cntrPepMean from each sample row-wise in centered_data
    adjusted_dat = centered_data.subtract(
        cntrPepMean, 
        axis=0
    ).replace(np.nan, np.mean(cntrPepMean))

    # Adjusted data
    test_data = adjusted_dat.reset_index().melt(
        id_vars=["Protein", "Peptide"],
        var_name="Sample",
        value_name="adjIntensity"
    )
    # Add the raw data
    test_data['Intensity'] = data.reset_index().melt(
        id_vars=["Protein", "Peptide"],
        var_name="Sample",
        value_name="ms1"
    )['ms1']
    test_data['log10Intensity'] = np.log10(test_data['Intensity'])
    
    # Add the condition
    test_data['Condition'] = test_data['Sample'].apply(
        lambda x: "-".join(x.split("-")[:-1])
    )
    # Add isReal checker if missing_data is provided
    if missing_data is not None:
        test_data['isReal'] = missing_data.reset_index().melt(
            id_vars=["Protein", "Peptide"],
            var_name="Sample",
            value_name="Intensity"
        )['Intensity'].notna().astype(int)
    else: 
        # Add isReal checker
        test_data['isReal'] = 1
    # Add the missing values
    test_data['isImputed'] = test_data['isReal'].apply(lambda x: 1 - x)
    # Add checker for complete missing values (for all replicates)
    test_data['isCompMiss'] = test_data.groupby([
        'Protein', 'Peptide', 'Condition'
    ])['isImputed'].transform('all').astype(int)
    # Remove the isImputed column
    test_data.drop(columns=['isImputed'], inplace=True)

    ## The Imputation Values (Real, Sparse, Complete)
    value_def = {"Real": 1, "Sparse": 10**-10, "Complete": .75}
    test_data['imputeWeight'] = np.where(
        test_data['isReal'] == 1, value_def["Real"], 
        np.where(
            test_data['isCompMiss'] == 1, 
            value_def["Complete"], 
            value_def["Sparse"]
        )
    )

    # Data to be Build for the perturbation
    info_data = pd.DataFrame(perturbation_map).T
    info_data = assign_groups_based_on_perturbation(
        info_data,
        protein_col="Protein",
        peptide_col="Peptide",
        condition_col="pertCondition",
        shift_col="pertShift",
        pertPFG_col="pertPFG"
    ).reindex(data.index).reset_index()
    info_data['pertPFG'] = info_data['pertPFG'].fillna(-1).astype(int)
    info_data['pertPeptide'] = info_data['pertShift'].notnull()
    info_data['pertProtein'] = info_data['Protein'].isin(proteins_to_perturb)
    info_data['Reason'] = "Perturbation"

    # Combine the data for the perturbation
    test_data = test_data.merge(
        info_data, 
        on=['Protein', 'Peptide'], 
        how='left'
    )

    test_data.insert(2, 'PeptideID', test_data['Peptide'].str.split('_').str[-1].astype(int))

    if missing_data is not None:
        # Add small value for other tools since they won't work with any missing values
        test_data['Intensity'] = test_data['Intensity'].fillna(10) 
        test_data['log10Intensity'] = np.log10(test_data['Intensity'])

    return test_data