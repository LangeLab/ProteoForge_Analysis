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
# if cur_wd has /Benchmark at the end, remove it
if cur_wd.endswith("/Benchmark"):
    os.chdir(os.path.dirname(cur_wd))

# Add project root directory to path for custom modules
project_root = os.getcwd()
sys.path.insert(0, project_root)

# Main Computational Components of ProteoForge
from ProteoForge import normalize
from ProteoForge import disluster
from ProteoForge import weight, model


# Set back to Benchmark directory for data paths
os.chdir(os.path.join(project_root, "Benchmark"))

# Globals and Constants
cur_method = 'ProteoForge'
data_ids = ["1pep", "2pep", "050pep", "random"]  # Data IDs to process

## Configuration for the analysis
# Model type to use:
# - 'quantile' for median quantile regression (auto-weights)
# - 'rlm' for robust linear model (auto-weights - robust)
# - 'ols' for ordinary least squares (no weights)
# - 'wls' for weighted least squares (user-configured weights)
# - 'glm' for generalized linear model (user-configured weights)
# Notes: rlm and quantile best for non-imputed data,
#        wls and glm best for imputed data with user-defined weights
model_to_use = 'quantile' #
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

# Create results directory if it doesn't exist
data_path = "./data/prepared/"
output_path = "./data/results/"
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
    print(f"Created results directory: {output_path}")

# Initialize results tracking
results_summary = {}

# Step 0: Create condition-run dictionary (done once for all datasets)
print("Step 0: Creating condition-run dictionary...")
step_start = time.time()

# Open single data for cond_run_dict
try:
    initial_data = pd.read_feather(f"{data_path}bench_1pep_input.feather")
    # Create day:[filenames] dictionary
    cond_run_dict = initial_data.groupby("day")["filename"].unique().to_dict()
    print(f"  Created condition-run dictionary with {len(cond_run_dict)} conditions")
    print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
    print()
except Exception as e:
    print(f"  ERROR: Failed to create condition-run dictionary: {e}")
    sys.exit(1)

# Process each data ID
for i, data_id in enumerate(data_ids, 1):
    print(f"\n[{i}/{len(data_ids)}] Processing dataset: {data_id}")
    print("-" * 80)
    
    # Initialize time for this dataset
    start_time = time.time()
    
    try:
        # Step 1: Data loading
        step_start = time.time()
        print("Step 1: Loading data...")
        input_data_path = f"{data_path}bench_{data_id}_input.feather"
        
        # Check if file exists
        if not os.path.exists(input_data_path):
            raise FileNotFoundError(f"File not found: {input_data_path}")
        
        # Read the input data (feather format)
        data = pd.read_feather(input_data_path).sort_values([
            'protein_id', 'peptide_id', 'day', 'filename'
        ]).reset_index(drop=True)
        
        print(f"  Data read from: {input_data_path}")
        print(f"  Dimensions: {data.shape[0]} rows x {data.shape[1]} columns")
        print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
        print()
        
        # Step 2: Data normalization
        step_start = time.time()
        print("Step 2: Normalizing data against condition...")
        
        test_data = normalize.against_condition(
            data, 
            cond_run_dict,
            run_col="filename",
            index_cols=["protein_id", "peptide_id"],
            norm_against="day1",
            intensity_col="intensity",
            is_log2=False,
            norm_intensity_col="ms1adj" 
        )
        test_data['log10Intensity'] = np.log10(test_data['intensity'])  
        
        print(f"  Normalized data against 'day1' condition")
        print(f"  Normalized data dimensions: {test_data.shape[0]} rows x {test_data.shape[1]} columns")
        print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
        print()
        
        # Step 3: Weight calculation
        step_start = time.time()
        print("Step 3: Calculating peptide weights...")
        weights_data = weight.generate_weights_data(
            test_data,
            sample_cols=['filename'],
            log_intensity_col='log10Intensity',
            adj_intensity_col='ms1adj',
            control_condition='day1',
            condition_col='day',
            protein_col='protein_id',
            peptide_col='peptide_id',
            is_real_col=None,
            is_comp_miss_col=None,
            verbose=False,
        )

        test_data['Weight'] = (
            (weights_data['W_Impute'] * 0.90) + 
            (weights_data['W_RevTechVar'] * 0.10)
        )
        
        # Create numerical peptide index
        test_data['peptide_Idx'] = test_data.groupby(['protein_id', 'peptide_id']).cumcount()+1
        
        print(f"  Calculated optimal weights for the dataset")
        print(f"  Added peptide index column")
        print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
        print()
        
        # Step 4: Linear model analysis
        step_start = time.time()
        print("Step 4: Running linear model analysis...")
        
        # Initialize the linear model with parameters
        cur_model = model.LinearModel(
            data=test_data,
            protein_col="protein_id",
            peptide_col="peptide_id",
            cond_col="day",
            intensity_col="ms1adj",
            weight_col="Weight",
        )
        
        test_data = cur_model.run_analysis(
            model_type=model_to_use,
            correction_strategy= correction['strategy'],
            correction_methods=correction['methods'],
            n_jobs=n_jobs
        )
        
        if correction['strategy'] == 'two-step' and isinstance(correction['methods'], tuple):
            methods_str = " and ".join(correction['methods'])
            print(f"  Ran weighted least squares (WLS) model with two-step correction")
            print(f"  Used {n_jobs} parallel jobs")
            print(f"  Applied {methods_str} corrections")
        else:
            method_str = correction['methods'] if isinstance(correction['methods'], str) else ", ".join(correction['methods'])
            print(f"  Ran weighted least squares (WLS) model with {correction['strategy']} correction")
            print(f"  Used {n_jobs} parallel jobs")
            print(f"  Applied {method_str} correction")
        print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
        print()
        
        # Step 5: Correlation analysis
        step_start = time.time()        
        print("Step 5: Performing peptide clustering...")
    
        clusters = disluster.distance_and_cluster(
            data=test_data,
            protein_col='protein_id',
            peptide_col='peptide_id',
            cond_col='day',
            quant_col='ms1adj',
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
            clusters[['protein_id', 'peptide_id', 'cluster_label']],
            on=['protein_id', 'peptide_id'],
            how='left'
        ).rename(columns={'cluster_label': 'ClusterID'})

        # Count proteins and peptides processed
        proteins_processed = test_data['protein_id'].nunique()
        peptides_processed = test_data['peptide_id'].nunique()
        
        print(f"  Performed hierarchical clustering with average linkage")
        print(f"  Auto-determined optimal number of clusters (1-12 range)")
        print(f"  Used correlation distance metric with {n_jobs} parallel jobs")
        print(f"  Processed {proteins_processed} proteins with {peptides_processed} peptides")
        print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
        print()
        
        # Step 6: Saving results
        step_start = time.time()
        print("Step 6: Saving results...")
        
        # Save data to a feather file
        output_file_path = f"{output_path}{cur_method}_{data_id}_result.feather"
        test_data.to_feather(output_file_path)
        
        print(f"  Results saved to: {output_file_path}")
        print(f"  Output dimensions: {test_data.shape[0]} rows x {test_data.shape[1]} columns")
        print(f"  Time taken: {format_time_diff(time.time() - step_start)}")
        print()
        
        # Calculate total time for this dataset
        end_time = time.time()
        total_time_diff = end_time - start_time
        total_time_taken = format_time_diff(total_time_diff)
        
        # Store results summary
        results_summary[data_id] = {
            'status': 'SUCCESS',
            'total_time': total_time_taken,
            'total_time_seconds': total_time_diff,
            'proteins_processed': proteins_processed,
            'peptides_processed': peptides_processed,
            'input_rows': data.shape[0],
            'output_rows': test_data.shape[0],
            'output_file': output_file_path
        }
        
        print(f"Dataset {data_id} completed successfully in {total_time_taken}")
        print("=" * 80)
        
    except Exception as e:
        # Handle errors gracefully
        end_time = time.time()
        total_time_diff = end_time - start_time
        
        results_summary[data_id] = {
            'status': 'FAILED',
            'error': str(e),
            'total_time_seconds': total_time_diff
        }
        
        print(f"  ERROR: Dataset {data_id} failed: {e}")
        print("=" * 80)
        continue

# Print overall summary
overall_end_time = time.time()
overall_time_diff = overall_end_time - overall_start_time

print("\n\n")
print("=" * 80)
print("                             FINAL SUMMARY                                      ")
print("=" * 80)
print(f"Overall execution time: {format_time_diff(overall_time_diff)}")
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end_time))}")
print()

print("Results by dataset:")
print("------------------")
for data_id in results_summary:
    result = results_summary[data_id]
    if result['status'] == 'SUCCESS':
        print(f"✓ {data_id:<8}: {result['total_time']} ({result['proteins_processed']} proteins, {result['peptides_processed']} peptides)")
    else:
        print(f"✗ {data_id:<8}: FAILED ({result['error']})")

print("\nDetailed results:")
print("-----------------")
for data_id in results_summary:
    result = results_summary[data_id]
    if result['status'] == 'SUCCESS':
        print(f"{data_id}:")
        print(f"  - Processing time: {result['total_time']}")
        print(f"  - Input rows: {result['input_rows']}")
        print(f"  - Output rows: {result['output_rows']}")
        print(f"  - Proteins processed: {result['proteins_processed']}")
        print(f"  - Peptides processed: {result['peptides_processed']}")
        print(f"  - Output file: {result['output_file']}")
        print()

# Calculate and display performance metrics
successful_runs = sum(1 for r in results_summary.values() if r['status'] == 'SUCCESS')
total_runs = len(results_summary)

print(f"Success rate: {successful_runs}/{total_runs} ({(successful_runs/total_runs)*100:.1f}%)")

if successful_runs > 0:
    successful_results = [r for r in results_summary.values() if r['status'] == 'SUCCESS']
    avg_time = sum(r['total_time_seconds'] for r in successful_results) / len(successful_results)
    total_proteins = sum(r['proteins_processed'] for r in successful_results)
    total_peptides = sum(r['peptides_processed'] for r in successful_results)
    
    print(f"Average processing time per dataset: {format_time_diff(avg_time)}")
    print(f"Total proteins processed: {total_proteins}")
    print(f"Total peptides processed: {total_peptides}")

print("=" * 80)
print("ProteoForge benchmark analysis completed.")
print("=" * 80)