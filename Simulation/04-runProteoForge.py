# Initialize overall start time
import os
import sys
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Start timing
overall_start_time = time.time()

# Import required libraries
import numpy as np
import pandas as pd
import multiprocessing as mp

# Set working directory (similar to R script)
cur_wd = os.getcwd()
# if cur_wd has /Simulation at the end, remove it
if cur_wd.endswith("/Simulation"):
    os.chdir(os.path.dirname(cur_wd))

# Add project root directory to path for custom modules
project_root = os.getcwd()
sys.path.insert(0, project_root)

# Main Computational Components of ProteoForge
from ProteoForge import disluster
from ProteoForge import weight, model


# Set back to Benchmark directory for data paths
os.chdir(os.path.join(project_root, "Simulation"))

# Globals and Constants
cur_method = 'ProteoForge'
data_ids = ['Sim1', 'Sim2', 'Sim3', 'Sim4']
# data_ids = ['Sim4']  # For testing, only process Sim4


## Configuration for the analysis
# Model type to use:
# - 'quantile' for median quantile regression (auto-weights)
# - 'rlm' for robust linear model (auto-weights - robust)
# - 'ols' for ordinary least squares (no weights)
# - 'wls' for weighted least squares (user-configured weights)
# - 'glm' for generalized linear model (user-configured weights)
# Notes: rlm and quantile best for non-imputed data,
#        wls and glm best for imputed data with user-defined weights
model_to_use = 'wls' #
# Multiple Testing Correction strategy and methods
# - 'global' correction done after all peptide p-values are calculated
# - 'protein-only' correction done for peptides within each protein
# - 'two-step' protein-only followed by global correction
# Methods can be:
# - 'bonferroni' strict best done for within protein correction
# - 'holm' (Holm-Bonferroni) 
# - 'fdr_bh' (Benjamini-Hochberg) general purpose
# - 'fdr_by' (Benjamini-Yekutieli) dependence correction
# - 'qvalue' (q-value estimation) omics oriented similar to fdr_bh
# Default option is two-step with fdr_bh for both steps, 
# can be bonferroni+fdr_bh but this effects larger proteins more negatively
# protein-only or global can be an option but more false positives expected...
correction = {
    # 'strategy': 'global',  # Options: 'global', 'protein-only', 'two-step'
    # 'methods': 'fdr_bh'  # Options: 'bonferroni', 'holm', 'fdr_bh', 'fdr_by', 'qvalue'
    'strategy': 'two-step',  # Options: 'global', 'protein-only', 'two-step'
    'methods': ('bonferroni', 'fdr_bh')  # Options first protein-level, then global
}
# Weight Component and Importance 

# Automatically detect CPU cores and use 4/5 of them
total_cores = mp.cpu_count()
# Use half of the available cores to avoid overloading
n_jobs = max(1, total_cores // 2)

# Function to format time difference
def format_time_diff(time_diff):
    if time_diff < 60:
        return f"{time_diff:.2f} seconds"
    elif time_diff < 3600:
        return f"{time_diff / 60:.2f} minutes"
    else:
        return f"{time_diff / 3600:.2f} hours"

def min_max_scale(x): return (x - x.min()) / (x.max() - x.min())

# Print header
print("=" * 80)
print("                        PROTEOFORGE ALGORITHM BENCHMARK                        ")
print("=" * 80)
print("Starting ProteoForge analysis on multiple datasets...")
print(f"Data IDs to process: {', '.join(data_ids)}")
print(f"CPU cores available: {total_cores}, using: {n_jobs} cores")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time))}")
print("=" * 80)
print()


# Initialize results tracking
results_summary = {}

# Step 0: Create condition-run dictionary (done once for all datasets)
print("Step 0: Creating condition-run dictionary...")
step_start = time.time()

# Process each data ID
for i, data_id in enumerate(data_ids, 1):
    # Initialize time for this dataset
    sim_start_time = time.time()

    print(f"\n[{i}/{len(data_ids)}] Processing dataset: {data_id}")
    print("-" * 80)
    # Define data path for the simulation
    data_path = f"./data/{data_id}/"
    # Find ones ends with _InputData.feather
    input_files = [f for f in os.listdir(data_path) if f.endswith('_InputData.feather')]
    if not input_files:
        print(f"Error: No input data file found for {data_id} in {data_path}")
        continue

    print(f"Found {len(input_files)} input files for dataset {data_id}.")
    
    # Looping through input files (usually one per dataset)
    for input_file in input_files:
        file_start_time = time.time()
        step_start = time.time()
        print(f" Step 1: Loading data from {input_file}...")
        input_data_path = os.path.join(data_path, input_file)
        # Set output file name and path
        output_file = input_file.replace('_InputData.feather', '_ProteoForge_ResultData.feather')
        output_data_path = os.path.join(data_path, output_file)
    
        # Check if file exists
        if not os.path.exists(input_data_path):
            raise FileNotFoundError(f"File not found: {input_data_path}")
    
        # Load data
        test_data = pd.read_feather(input_data_path)
        print(f"  Data read from: {input_data_path}")
        print(f"  Dimensions: {test_data.shape[0]} rows x {test_data.shape[1]} columns")
        print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
        print()

        # Step 2: Weight calculation
        step_start = time.time()
        print("Step 2: Calculating peptide weights...")

        weights_data = weight.generate_weights_data(
            test_data,
            sample_cols=['Sample'],
            log_intensity_col='log10Intensity',
            adj_intensity_col='adjIntensity',
            control_condition='control',
            condition_col='Condition',
            protein_col='Protein',
            peptide_col='Peptide',
            is_real_col='isReal',
            is_comp_miss_col='isCompMiss',
            sparse_imputed_val=1e-10,
            dense_imputed_val=0.75,
            verbose=False,
        )
        
        test_data['Weight'] = (
            (weights_data['W_Impute'] * 0.90) + 
            (weights_data['W_RevTechVar'] * 0.10)
        )

        print(f"  Calculated optimal weights for the dataset")
        print(f"  Added peptide index column")
        print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
        print()

        # (for complex permutations have np.array for pertCondition and pertShift)
        # Convert numpy arrays to strings before running the model 
        # To ensure drop_duplicates works correctly 
        test_data['pertCondition'] = test_data['pertCondition'].apply(
            lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x)
        )
        test_data['pertShift'] = test_data['pertShift'].apply(
            lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x)
        )

        # Step 3: Linear model analysis
        step_start = time.time()
        print("Step 3: Running linear model analysis...")
        # Initialize the linear model with parameters
        cur_model = model.LinearModel(
            data=test_data,
            protein_col="Protein",
            peptide_col="Peptide",
            cond_col="Condition",
            intensity_col="adjIntensity",
            weight_col="Weight",
        )

        test_data = cur_model.run_analysis(
            model_type=model_to_use,
            correction_strategy= correction['strategy'],
            correction_methods=correction['methods'],
            n_jobs=n_jobs
        )

        print(f"  Linear model type: {model_to_use}")
        print(f"  Multiple testing correction: {correction['strategy']} with {correction['methods']}")
        print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
        print()
    
        # Step 4: Correlation analysis
        step_start = time.time()

        print("Step 4: Performing peptide clustering...")

        clusters = disluster.distance_and_cluster(
            data=test_data,
            protein_col='Protein',
            peptide_col='PeptideID',
            cond_col='Condition',
            quant_col='adjIntensity',
            clustering_params={
                'min_clusters': 1,
                'distance_transform': 'corr',
                'clustering_method': 'hybrid_outlier_cut',
                'linkage_method': 'ward',
                'distance_metric': 'euclidean'
            },
            n_jobs=n_jobs,
            verbose=False
        )

        test_data = test_data.merge(
            clusters[['Protein', 'PeptideID', 'cluster_label']],
            on=['Protein', 'PeptideID'],
            how='left'
        ).rename(columns={'cluster_label': 'ClusterID'})

        # Count proteins and peptides processed
        proteins_processed = test_data['Protein'].nunique()
        peptides_processed = test_data['Peptide'].nunique()
        
        print(f"  Performed hierarchical clustering with average linkage")
        print(f"  Auto-determined optimal number of clusters (1-12 range)")
        print(f"  Used 1 - correlation distance metric with {n_jobs} parallel jobs")
        print(f"  Processed {proteins_processed} proteins with {peptides_processed} peptides")
        print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
        print()


        # Step 6: Proteoform classification and Summarization
        step_start = time.time()
        print("Step 5: Summarizing results...")
        # Update perturbed based on imputation status
        test_data['pertPeptide'] = np.where(
            test_data['isCompMiss'] == 1, 1, test_data['pertPeptide']
        ).astype(bool)
        test_data['Reason'] = np.where(
            test_data['isCompMiss'] == 1, "Biological Absence", test_data['Reason']
        )
        test_data.loc[
            (test_data['isCompMiss']==1) & (test_data['pertPeptide']), 
            'pertPFG'
        ] = 1

        # Summarize data for proteoform classification
        summary_data = test_data[[
            'Protein', 'pertProtein', 
            'PeptideID', 'pertPeptide', 'pertPFG', 
            'pval', 'adj_pval', 'ClusterID'
        ]].drop_duplicates().copy()

        # Step 7: Saving results
        step_start = time.time()
        print("Step 7: Saving results...")

        summary_data.to_feather(output_data_path)

        print(f"  Results saved to: {output_file}")
        print(f"  Output dimensions: {summary_data.shape[0]} rows x {summary_data.shape[1]} columns")
        print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
        print()
        
        # Calculate total time for this dataset
        end_time = time.time()
        total_time_diff = end_time - file_start_time
        total_time_taken = format_time_diff(total_time_diff)

        print(f"Completed processing for {data_id} in {total_time_taken}")
        print("-" * 80)

    # Simulation summary
    sim_end_time = time.time()
    sim_time_diff = sim_end_time - sim_start_time
    sim_time_taken = format_time_diff(sim_time_diff)
    print(f"Summary for dataset {data_id}:")
    print(f" Total proteins processed: {proteins_processed}")
    print(f" Total peptides processed: {peptides_processed}")
    print(f" Total time taken: {sim_time_taken}")
    print("=" * 80)
    print()

# Overall completion time
overall_end_time = time.time()
overall_time_diff = overall_end_time - overall_start_time
overall_time_taken = format_time_diff(overall_time_diff)

print("=" * 80)
print("ProteoForge Analysis Completed")
print(f"Overall time taken for all datasets: {overall_time_taken}")
print("=" * 80)
