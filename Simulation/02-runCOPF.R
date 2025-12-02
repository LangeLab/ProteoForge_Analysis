# Initialize overall start time
overall_start_time <- Sys.time()
# Set to not show warnings
options(warn=-1)

suppressMessages(library(arrow, warn.conflicts = FALSE))
suppressMessages(library(data.table, warn.conflicts = FALSE))
 # Needs the proteoformmapping branch
suppressMessages(library(CCprofiler, warn.conflicts = FALSE))

cur_wd <- getwd()
# if cur_wd has /Simulation at the end, remove it
if (grepl("/Simulation$", cur_wd)) {
  setwd(dirname(cur_wd))
}
# Set back to Simulation
setwd(paste0(getwd(), "/Simulation"))

# Function to format time difference
format_time_diff <- function(time_diff) {
  if (time_diff < 60) {
    return(paste(round(time_diff, 2), "seconds"))
  } else if (time_diff < 3600) {
    return(paste(round(time_diff / 60, 2), "minutes"))
  } else {
    return(paste(round(time_diff / 3600, 2), "hours"))
  }
}

sim_ids = c("Sim1", "Sim2", "Sim3", "Sim4")  # List of simulation IDs to process
# sim_ids = c("Sim4")  # For testing, only process Sim1
score_cutoff <- 0.1 
adj_pval_cutoff <- 0.01

# Print header
cat("================================================================================\n")
cat("                        SIMULATIONS RUN ON COPF ALGORITHM                       \n")
cat("================================================================================\n")
cat("Starting COPF analysis on multiple simulation setups...\n")
cat("Simulation IDs to process:", paste(sim_ids, collapse = ", "), "\n")
cat("Start time:", format(overall_start_time), "\n")
cat("================================================================================\n\n")

# Data path -> ./data/[simID]
# Files start with 2_ and ends with _InputData.feather

# Each simulation has a different naming setup since they look at different aspects
# Sim1 File Name Rules:
#  2_[nPertPep]_[complete|imputed]_InputData.feather
# Sim2 File Name Rules:
# 2_Pro[MissPct]_Pep[MissPct]_imputed_InputData.feather
# Sim3 File Name Rules:
# 2_[nCondition]Cond_[NonOverlap|Overlap]_InputData.feather
# Sim4 File Name Rules:
# 2_[pertMagRange[1]]_[pertMagRange[2]]_InputData.feather


# Initialize results tracking
results_summary <- list()

# Process each data ID
for (i in seq_along(sim_ids)) {
    data_id <- sim_ids[i]
    
    cat(sprintf("\n[%d/%d] Processing dataset: %s\n", i, length(sim_ids), data_id))
    cat("--------------------------------------------------------------------------------\n")
    
    # Initialize time for the simulation
    sim_start_time <- Sys.time()

    # Check available data within the simulation folder
    data_files <- list.files(
        path = paste0("./data/", data_id), 
        pattern = "^2_.*_InputData.feather$", full.names = TRUE
    )
    if (length(data_files) == 0) {
        cat(" No input data files found for this simulation ID.\n")
        next
    }
    cat(sprintf(" Number of available input data files: %d\n", length(data_files)))

    # Process each version of the data
    for (cur_file in data_files) {
        # File analysis tracking
        file_start_time <- Sys.time()
        # Step 1: Data loading
        step_start <- Sys.time()
        cat(" Step 1: Loading data...\n")
        # Check if file exists
        if (!file.exists(cur_file)) {
            cat("  ERROR: File not found:", input_data_path, "\n")
            results_summary[[paste(data_id, cur_file, sep = "_")]] <- list(status = "FAILED", error = "File not found")
            next
        }
        # Read the input data (feather format)
        input_data <- arrow::read_feather(cur_file)
        cat("  Data read from:", cur_file, "\n")
        cat("  Dimensions:", nrow(input_data), "rows x", ncol(input_data), "columns\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")

        # Convert to data.table
        input_data <- as.data.table(input_data)
        input_data[,"peptide_id":=paste(Protein, Peptide, sep="-")]
        # Rename columns
        data.table::setnames(
            input_data,
            c("Peptide","Protein", "Intensity","Condition", "Sample"), 
            c("Peptide","protein_id","intensity","Condition", "filename")
        )
        
        
        # Step 2: Preparing annotation and traces
        step_start <- Sys.time()
        cat(" Step 2: Preparing annotation and trace structure...\n") 

        # Create an annotation table for the input data
        input_annot <- unique(subset(input_data, select=c("filename","Condition")))
        # Set the order of the annotation table by day and filename
        setorderv(input_annot, c("Condition","filename"))
        # Add a column with the fraction number (from 1 to nrow(input_annot))
        input_annot[,fraction_number:=.I]
        # Keep only the filename and fraction number columns
        input_annot <- subset(input_annot, select=c("filename","fraction_number"))

        cat("  Created fraction annotation with", nrow(input_annot), "fractions\n")

        # Essentially the fancy dictionary with the following subsets:
        # - traces: quant_data with samples as columns + id column (tracetype:peptide)
        # - trace_type: peptide
        # - trace_annotation: id column +  other info_columns here (similar to my info_data)
        # - fraction_annotation: metadata for the sample info to be used.
        traces <- CCprofiler::importPCPdata(
            input_data = input_data,
            fraction_annotation = input_annot
        )
        
        cat("  Imported PCP data structure\n")
        
        # Get the unique subset for the data with useful columns
        trace_annotation <- unique(
            subset(
                input_data, 
                select=c(
                    "peptide_id",
                    "pertProtein",
                    "pertPeptide",
                    "pertPFG"
                )
            )
        )
        # There is a weird bug in the code that requires the peptide_id to be duplicated
        trace_annotation <- trace_annotation[!duplicated(peptide_id), ]

        ## Annotation of traces steps, which expands the traces list with additionall annotation for proteins etc.
        # (seems for this data doesn't do anything maybe since it is at peptide level)
        traces <- CCprofiler::annotateTraces(
            traces,
            trace_annotation,
            traces_id_column = "id",
            trace_annotation_id_column = "peptide_id", 
            uniprot_mass_format = FALSE
        )
        
        cat("  Annotated traces with additional metadata\n")
        cat("  Final trace structure:", nrow(traces$trace_annotation), "rows x", ncol(traces$trace_annotation), "columns\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")

        # Step 3: Data preprocessing
        step_start <- Sys.time()
        cat(" Step 3: Data preprocessing (removing zero variance and single peptide genes)...\n")
        
        ## Remove traces with 0 standard deviation (can't be clustered)
        # Binary vector of traces puts TRUE for non-zero variance traces
        zerovar <- apply(CCprofiler::getIntensityMatrix(traces), 1, var) > 0
        cat("  Traces with zero variance:", sum(!zerovar), "out of", length(zerovar), "\n")
        
        # Subset for the traces with non-zero variance
        traces_zerovar <- subset(
            traces,
            trace_subset_ids = names(zerovar[zerovar])
        )
        
        cat("  Remaining traces after zero variance removal:", nrow(traces_zerovar$trace_annotation), "\n")
        
        #' ## Remove single peptide genes
        traces_multiPep <- CCprofiler::filterSinglePeptideHits(traces_zerovar)
        
        cat("  Traces after removing single peptide genes:", nrow(traces_multiPep$trace_annotation), "\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")

        
        # Step 4: Correlation matrix calculation and clustering
        step_start <- Sys.time()
        cat(" Step 4: Calculating correlation matrices and clustering peptides...\n")
        
        #' ## Estimate proteoform scores by correlation clustering
        traces_corr <- CCprofiler::calculateGeneCorrMatrices(traces_multiPep)
        # This stores correlation matrix calculated for peptides stored at $geneCorrMatrices
        
        cat("  Calculated correlation matrices for", length(traces_corr$geneCorrMatrices), "genes\n")
        
        # Add clustering to each protein d as 1-genecorr with average cluster method
        traces_clustered <- CCprofiler::clusterPeptides(
            traces_corr,
            method = "average", 
            plot = F, 
            PDF = F,
            name=paste0("ProteoformClusters_interlab_", data_id)
        )
        
        cat("  Performed hierarchical clustering using average linkage\n")
        
        traces_clusteredInN <- CCprofiler::cutClustersInNreal(
            traces_clustered, 
            clusterN = 2,
            min_peptides_per_cluster = 2
        )
        
        cat("  Cut clusters into 2 groups with minimum 2 peptides per cluster\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")

        
        # Step 5: Proteoform scoring
        step_start <- Sys.time()
        cat(" Step 5: Calculating proteoform scores...\n")
        
        traces_scored <- CCprofiler::calculateProteoformScore(traces_clusteredInN)
        
        cat("  Calculated proteoform scores for clustered peptides\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")
        
        # Step 6: Proteoform annotation
        step_start <- Sys.time()
        cat(" Step 6: Annotating traces with proteoforms...\n")

        
        cat("  Using score cutoff:", score_cutoff, "and adjusted p-value cutoff:", adj_pval_cutoff, "\n")
        
        traces_proteoforms <- CCprofiler::annotateTracesWithProteoforms(
            traces_scored, 
            score_cutoff = score_cutoff, 
            adj_pval_cutoff =  adj_pval_cutoff
        )

        # Count proteins and peptides processed
        proteins_processed <- length(unique(traces_proteoforms$trace_annotation$protein_id))
        peptides_processed <- nrow(traces_proteoforms$trace_annotation)
        cat("  Processed", proteins_processed, "proteins with", peptides_processed, "peptides\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")

        # Step 7: Saving results
        step_start <- Sys.time()
        cat(" Step 7: Saving results...\n")

        # Output Path: replace '2_' with '3_' and '_InputData.feather' with '_COPF_ResultData.feather' in cur_file name
        output_path <- sub("^2_", "3_", cur_file)
        output_path <- sub("_InputData.feather$", "_COPF_ResultData.feather", output_path)
        arrow::write_feather(
            as.data.frame(traces_proteoforms$trace_annotation), 
            output_path
        )

        cat("  Results saved to:", output_path, "\n")
        cat("  Output dimensions:", nrow(traces_proteoforms$trace_annotation), "rows x", ncol(traces_proteoforms$trace_annotation), "columns\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")
        
        # Calculate total time for this dataset
        file_end_time <- Sys.time()
        total_time_diff <- as.numeric(difftime(file_end_time, file_start_time, units = "secs"))
        total_time_taken <- format_time_diff(total_time_diff)

        # Store results summary
        results_summary[[paste(data_id, cur_file, sep = "_")]] <- list(
            status = "SUCCESS",
            total_time = total_time_taken,
            total_time_seconds = total_time_diff,
            proteins_processed = proteins_processed,
            peptides_processed = peptides_processed,
            input_rows = nrow(input_data),
            output_rows = nrow(traces_proteoforms$trace_annotation),
            output_file = output_path
        )
        
        cat("File", basename(cur_file), "for dataset", data_id, "completed successfully in", total_time_taken, "\n")
        cat("================================================================================\n\n")
    }

    # Summarize and print time taken for current data_id
    sim_end_time <- Sys.time()
    sim_time_diff <- as.numeric(difftime(sim_end_time, sim_start_time, units = "secs"))
    cat(sprintf("\nSummary for %s:\n", data_id))
    cat(sprintf("  Total time taken: %s\n", format_time_diff(sim_time_diff)))
    cat(sprintf("  End time: %s\n", format(sim_end_time)))
    cat("--------------------------------------------------------------------------------\n\n")

}

# Print overall summary
overall_end_time <- Sys.time()
overall_time_diff <- as.numeric(difftime(overall_end_time, overall_start_time, units = "secs"))


cat("\n\n")
cat("================================================================================\n")
cat("                             FINAL SUMMARY                                      \n")
cat("================================================================================\n")
cat("Overall execution time:", format_time_diff(overall_time_diff), "\n")
cat("End time:", format(overall_end_time), "\n\n")

cat("Results by dataset:\n")
cat("------------------\n")
for (data_id in names(results_summary)) {
    result <- results_summary[[data_id]]
    if (result$status == "SUCCESS") {
        cat(
            sprintf(
                "✓ %-8s: %s (%d proteins, %d peptides)\n", 
                data_id, result$total_time, 
                result$proteins_processed, result$peptides_processed
            )
        )
    } else {
        cat(sprintf("✗ %-8s: FAILED (%s)\n", data_id, result$error))
    }
}


# cat("\nDetailed results:\n")
# cat("-----------------\n")
# for (data_id in names(results_summary)) {
#     result <- results_summary[[data_id]]
#     if (result$status == "SUCCESS") {
#         cat(sprintf("%s:\n", data_id))
#         cat(sprintf("  - Processing time: %s\n", result$total_time))
#         cat(sprintf("  - Input rows: %d\n", result$input_rows))
#         cat(sprintf("  - Output rows: %d\n", result$output_rows))
#         cat(sprintf("  - Proteins processed: %d\n", result$proteins_processed))
#         cat(sprintf("  - Peptides processed: %d\n", result$peptides_processed))
#         cat(sprintf("  - Output file: %s\n", result$output_file))
#         cat("\n")
#     }
# }

# Calculate and display performance metrics
successful_runs <- sum(sapply(results_summary, function(x) x$status == "SUCCESS"))
total_runs <- length(results_summary)

cat(
    sprintf("Success rate: %d/%d (%.1f%%)\n", 
    successful_runs, total_runs, 
    (successful_runs/total_runs)*100)
)

if (successful_runs > 0) {
    successful_results <- results_summary[sapply(results_summary, function(x) x$status == "SUCCESS")]
    avg_time <- mean(sapply(successful_results, function(x) x$total_time_seconds))
    total_proteins <- sum(sapply(successful_results, function(x) x$proteins_processed))
    total_peptides <- sum(sapply(successful_results, function(x) x$peptides_processed))
    
    cat(sprintf("Average processing time per dataset: %s\n", format_time_diff(avg_time)))
    cat(sprintf("Total proteins processed: %d\n", total_proteins))
    cat(sprintf("Total peptides processed: %d\n", total_peptides))
}

cat("================================================================================\n")
cat("                 COPF benchmark for simulation setups completed.                \n")
cat("================================================================================\n")