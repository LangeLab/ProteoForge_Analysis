import numpy as np
import pandas as pd 
from multiprocessing import Pool
from src import utils, tests, plots
import statsmodels.formula.api as smf


############################### Data Generation Functions ###############################

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
        Create a normal distribution with the provided mean and standard deviation,
        with additional outliers.

        Args:
            mu (float): Mean of the distribution
            sd (float): Standard deviation of the distribution
            size (int): Number of values to generate
            is_log2 (bool): Is the mean and sd at log2 scale
            outlier_fraction (float): Fraction of outliers
            outlier_sd_multiplier (float): Multiplier for outlier standard deviation
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
        control_color: str = "#8d99ae",
        condition_suffix: str = "cond-",
        condition_palette: str = "Blues",
        
    ):
    """
    Build condition names, their shifts, sample map, and palettes.

    Args:
        n_condition (int): Number of conditions (excluding control).
        n_replicates (int): Number of replicates per condition.
        condition_shifts (list): List of shifts for each condition (excluding control).
        base_palette (str): Base color palette for conditions.
        control_color (str): Color for the control condition.

    Returns:
        dict: condition_sample_map
        dict: condition_palette
        dict: sample_palette
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
    condition_pal = {control_name: control_color}
    tmp = plots.sns.color_palette(condition_palette, (n_condition)).as_hex()[::-1]
    for condition in conditions[1:]:
        condition_sample_map[condition] = [
            f'{condition}-{i}' for i in range(1, n_replicates + 1)
        ]
        condition_pal[condition] = tmp.pop()

    # Define the sample-palette mapping
    sample_palette = {}
    for condition, samples in condition_sample_map.items():
        for sample in samples:
            sample_palette[sample] = condition_pal[condition]

    return (
        condition_sample_map, condition_pal, 
        sample_palette, condition_shifts
    )

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
            pd.DataFrame: Complete data (shape: (nProteins, nSamples=nConditions*nReplicates))

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

############################### Perturbation Functions ###############################

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
        direction_sign = np.where(np.random.rand(nPro_to_perturb, 1) > 0.5, 1, -1)
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
        raise ValueError("Invalid direction type. Choose from 'random', 'same', or 'different'.")

    return perturb_values






def perturb_data(
        data: pd.DataFrame,
        perturb_settings: dict,
        seed: int = None,
    ):
    """ 
        Main function to add perturbations to the data based on the provided
        perturbation settings. The perturbations can be of different magnitudes, 
        types, and distributions, and can be applied to single, multiple 
        condiitons across single or multiple peptides. The additional noise can 
        be added to the data to simulate the real-world complexities.
    """
    pass

################################ (A/I)mputation Functions #################################

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
    proteins_to_amputate = np.random.choice(non_perturbed_proteins, n_amputate_1, replace=False)
    missing_data.loc[proteins_to_amputate] = missing_data.loc[proteins_to_amputate].mask(
        np.random.rand(*missing_data.loc[proteins_to_amputate].shape) < missing_rate
    )
    
    # Step 2: Select another set of non-perturbed proteins and remove values in one condition
    proteins_to_amputate = np.setdiff1d(non_perturbed_proteins, proteins_to_amputate)
    proteins_to_amputate = np.random.choice(proteins_to_amputate, n_amputate_2, replace=False)
    conds_to_amputate = np.random.choice(list(condition_shifts.keys()), n_amputate_2, replace=True)
    for i, protein in enumerate(proteins_to_amputate):
        cur_data = missing_data.loc[protein]
        cur_peptide = np.random.choice(cur_data.index, 1)[0]
        samples = condition_sample_map[conds_to_amputate[i]]
        missing_data.loc[(protein, cur_peptide), samples] = np.nan
    
    # Step 3: Select perturbed proteins and introduce missing values
    proteins_to_amputate = np.random.choice(proteins_to_perturb, n_amputate_3, replace=False)
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




################################# Benchmark Functions #################################

# Useful for ensuring the values are between 0 and 1
def min_max_scale(x): return (x - x.min()) / (x.max() - x.min())

def build_proteoform_groups_mp(protein, test_data):
    return utils.build_proteoform_groups(
        cur_protein=protein,
        data=test_data,
        sample_col="Sample",
        peptide_col="Peptide",
        protein_col="Protein",
        quant_col='adjIntensity',
        minSizePct=0.50, 
        absoluteMaxSize=3,
        corrMethod="kendall",
    )

def calculate_clusters(test_data, n_processes=28):
    with Pool(n_processes) as p:
        clusters = p.starmap(
            build_proteoform_groups_mp, [
                (protein, test_data) for protein in test_data['Protein'].unique()
            ]
        )
    clusters = pd.concat(clusters, axis=0, ignore_index=True)
    return clusters

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

    ### Add cluster_ids
    
    # Calculate the clustering
    clusters = calculate_clusters(test_data)

    # Create colormap for Clusters
    uniqClusters = clusters["cluster_id"].sort_values().unique()
    cluster_palette = plots.sns.color_palette("Set2", len(uniqClusters)).as_hex()
    cluster_palette = dict(zip(uniqClusters, cluster_palette))

    # Add the cluster palette to the test data
    test_data = test_data.merge(
        clusters,
        on=["Protein", "Peptide"],
        how="left"
    )

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

    ## Technical Replicate Variance (1 = Low, 0 = High)
    # Calculate tec variance per protein-peptide-condition
    test_data['TechVar'] = test_data.groupby([ 
        'Protein', 'Peptide', 'Condition'
    ])['Intensity'].transform('var')
    # Min-max scaling of the technical variance and inverse
    test_data['TechVar'] = 1 - min_max_scale(test_data['TechVar'])

    test_data['Weight'] = (
        (test_data['imputeWeight'] * 0.90) +
        (test_data['TechVar'] * 0.10)
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
    # Update the pertPeptide based on complete missing values 
    #   (if all missing then a peptide becomes pertPeptide)
    test_data['pertPeptide'] = np.where(
        test_data['isCompMiss'] == 1, 1, test_data['pertPeptide']
    ).astype(bool)

    test_data['Reason'] = np.where(
        test_data['isCompMiss'] == 1, "Biological Absence", test_data['Reason']
    )

    if missing_data is not None:
        # Add small value for other tools since they won't work with any missing values
        test_data['Intensity'] = test_data['Intensity'].fillna(10) 
        test_data['log10Intensity'] = np.log10(test_data['Intensity'])

    return test_data

def apply_ProteoForge(
        data: pd.DataFrame,
        formula: str,
        protein_col: str = 'Protein',
        peptide_col: str = 'Peptide',
        weight_col: str = 'Weight',
        comparison_col: str = 'allothers',
        correction_method: str = 'fdr'
    ):
    """
        Apply the ProteoForge method to the given data using the provided formula.
        The method is applied for each protein and peptide combination and the 
        p-values are calculated for the comparison_col.

        Args:
            data (pd.DataFrame): Input data
            formula (str): Formula to use for the model
            protein_col (str): Column name for protein names
            peptide_col (str): Column name for peptide names
            weight_col (str): Column name for weights
            comparison_col (str): Column name for comparison

        Returns:
            pd.DataFrame: Data with p-values and adjusted p-values
    """
    results = []
    unique_proteins = data[protein_col].unique()
    test_data = data.set_index([protein_col])
    for cur_prot in unique_proteins:
        protein_data = test_data.loc[cur_prot].copy()
        protein_data[comparison_col] = comparison_col
        unique_peptides = protein_data[peptide_col].unique()
        pvalues = {}
        for cur_pep in unique_peptides:
            peptide_data = protein_data.set_index(peptide_col).copy()
            peptide_data.loc[cur_pep, comparison_col] = cur_pep
            model = smf.wls(formula, data=peptide_data, weights=peptide_data[weight_col]).fit()
            pvalues[cur_pep] = float(model.wald_test_terms().pvalues[-1])
            model = None
        protein_data["pval"] = protein_data[peptide_col].map(pvalues)
        protein_data['pval'] = (protein_data['pval'].fillna(1)) * len(pvalues) 
        protein_data['pval'] = protein_data['pval'].clip(upper=1)
        protein_data["adj.pval"] = tests.multiple_testing_correction(
            protein_data["pval"], correction_type = correction_method
        )
        protein_data["Method"] = 'ProteoForge'
        # Update the test_data
        results.append(protein_data)

    # Combine the results
    results = pd.concat(results, axis=0)
    return results

def benchmark_ProteoForge_grouping(
        data: pd.DataFrame,
        thresholds: list,
        ## Existing Names
        pvalue_col: str = 'adj.pval',
        protein_col: str = 'Protein',
        cluster_col: str = 'cluster_id',
        perturbation_col: str = 'pertPFG',
        ## New Column Names to be Added
        predPFG_col: str = 'predPFG',
        PFGTrueLabel_col: str = 'PFGTrueLabel',
        PFGPredLabel_col: str = 'PFGPredLabel',
        significance_col: str = 'isSignificant',
        threshold_col: str = 'threshold',
        method_col: str = 'Method',
    ) -> pd.DataFrame :
    """
        Benchmark the ProteoForge method's Grouping based on the provided thresholds and
        return the metrics for each threshold.

        Args:
            data (pd.DataFrame): Input data
            thresholds (list): List of thresholds to use
            pvalue_col (str): Column name for p-values
            protein_col (str): Column name for protein names
            cluster_col (str): Column name for cluster ids
            perturbation_col (str): Column name for perturbation groups
            predPFG_col (str): Column name for predicted PFG
            PFGTrueLabel_col (str): Column name for true PFG labels
            PFGPredLabel_col (str): Column name for predicted PFG labels
            significance_col (str): Column name for significance
            threshold_col (str): Column name for threshold
            method_col (str): Column name for method

        Returns:
            pd.DataFrame: Metrics data for each threshold
    """

    metrics_data = []
    tmp = data.copy()
    for thr in thresholds:
        # Determine the significant peptides
        tmp[significance_col] = tmp[pvalue_col] < thr
        # Check if any peptide in the cluster is significant
        any_significant = tmp.groupby([
            protein_col, cluster_col
        ])[significance_col].transform('any')
        # Check if there is more than one significant peptide in the cluster
        multiple_significant = tmp.groupby([
            protein_col, cluster_col
        ])[significance_col].transform('sum') > 1
        # Calculate the predicted PFG
        tmp[predPFG_col] = np.where(
            any_significant & multiple_significant, 
            tmp[cluster_col], -1
        )
        tmp[PFGTrueLabel_col] = tmp[perturbation_col] > -1
        tmp[PFGPredLabel_col] = tmp[predPFG_col] > -1
        # Calculate the metrics
        metrics = utils.calculate_metrics(
            true_labels=tmp[PFGTrueLabel_col],
            pred_labels=tmp[PFGPredLabel_col],
            verbose=False, return_metrics=True
        )
        metrics = pd.DataFrame(metrics, index=[0])
        metrics[threshold_col] = thr
        metrics[method_col] = 'ProteoForge'
        metrics_data.append(metrics)

    metrics_data = pd.concat(metrics_data, axis=0, ignore_index=True)

    return metrics_data

def benchmark_COPF_grouping(
        data: pd.DataFrame,
        thresholds: list,
        proteoformScore_cutoff: float = None,
        ## Existing Names
        pvalue_col: str = 'adj.pval',
        protein_col: str = 'Protein',
        cluster_col: str = 'cluster_id',
        perturbation_col: str = 'pertPFG',
        proteoformID_col: str = 'proteoform_id',
        proteoformScore_col: str = 'proteoform_score',
    ):

    metrics_data = []
    tmp = data.copy()
    for thr in thresholds:
        tmp = utils.update_proteoform_grouping_in_COPF(
            tmp, 
            score_thr = proteoformScore_cutoff,
            pval_thr = thr,
            protein_col = protein_col,
            cluster_col = cluster_col,
            pval_col= pvalue_col,
            score_col = proteoformScore_col,
            sep='-'
        )
        tmp['Predicted_groupid'] = (tmp[proteoformID_col].str.split("-").str[-1]).fillna('-1').astype(float)
        tmp['PFGPredLabel'] = tmp['Predicted_groupid'] > 0
        tmp['PFGTrueLabel'] = tmp[perturbation_col] > 0

        metrics = utils.calculate_metrics(
            true_labels=tmp['PFGTrueLabel'],
            pred_labels=tmp['PFGPredLabel'],
            verbose=False, return_metrics=True
        )
        metrics = pd.DataFrame(metrics, index=[0])
        metrics['threshold'] = thr
        metrics['Method'] = 'COPF'
        metrics_data.append(metrics)

    metrics_data = pd.concat(metrics_data, axis=0, ignore_index=True)

    return metrics_data