# Initialize overall start time
overall_start_time <- Sys.time()
# Set to not show warnings
options(warn=-1)

suppressMessages(library(arrow, warn.conflicts = FALSE))
suppressMessages(library(data.table, warn.conflicts = FALSE))
suppressMessages(library(PeCorA, warn.conflicts = FALSE))

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

# Print header
cat("================================================================================\n")
cat("                             SIMULATIONS RUN ON PeCorA                          \n")
cat("================================================================================\n")
cat("Starting PeCorA analysis on multiple datasets...\n")
cat("Data IDs to process:", paste(sim_ids, collapse = ", "), "\n")
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
            cat("  ERROR: File not found:", cur_file, "\n")
            results_summary[[paste(data_id, cur_file, sep = "_")]] <- list(status = "FAILED", error = "File not found")
            next
        }
        # Read the input data (feather format)
        orig_data <- arrow::read_feather(cur_file)
        
        input_data <- copy(orig_data)
        cat("  Data read from:", cur_file, "\n")
        cat("  Dimensions:", nrow(input_data), "rows x", ncol(input_data), "columns\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")

        # Convert to data.table
        input_data <- as.data.table(input_data)

        # Rename columns
        data.table::setnames(
            input_data,
            c("Intensity", "Sample"), 
            c("Normalized.Area", "filename")
        )
        # Add a new column () with the same values as Peptide
        # Make the Peptide.Modified.Sequence column with Protein_Peptide
        input_data[,"Peptide.Modified.Sequence":=paste(Protein, Peptide, sep="-")]
        cat("  Renamed columns for PeCorA compatibility\n")

        # Create an annotation table for the input data
        ann <- unique(subset(input_data, select=c("filename","Condition"))) 
        # Order by Condition and filename
        setorderv(ann,c("Condition","filename")) 
        # Extract BioReplicate number from filename (last number 1-10)
        ann[, BioReplicate := as.integer(stringr::str_extract(filename, "(?<=-)([1-9]|10)$"))]

        cat("  Created annotation table with", nrow(ann), "samples\n")
        cat("  Conditions:", paste(unique(ann$Condition), collapse = ", "), "\n")

        # Merge the annotation table with the input data
        input_data <- merge(input_data,ann,by=c("filename","Condition"))
        
        # Subset the data to keep only the useful columns
        input_data <- subset(
            input_data, 
            select=c("Protein","Peptide","Peptide.Modified.Sequence","Condition","BioReplicate","Normalized.Area")
        )
        
        # Move the data.table to a data.frame
        pecora_df <- setDF(input_data)
        
        cat("  Merged annotation and subset data\n")
        cat("  Final data dimensions for PeCorA:", nrow(pecora_df), "rows x", ncol(pecora_df), "columns\n")
        cat("  Unique proteins:", length(unique(pecora_df$Protein)), "\n")
        cat("  Unique peptides:", length(unique(pecora_df$Peptide.Modified.Sequence)), "\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")

        # Step 3: PeCorA preprocessing (can be slow)
        step_start <- Sys.time()
        cat("Step 3: Running PeCorA preprocessing...\n")
        cat("  This step can take several minutes depending on data size\n")

        # Preprocess the data for PeCorA (Super Slow - 1.97m)
        scaled_peptides <- PeCorA::PeCorA_preprocessing(
            pecora_df,
            area_column_name=6,
            threshold_to_filter=min(pecora_df$Normalized.Area),
            control_name="control"
        )
        
        cat("  Preprocessing completed\n")
        cat("  Processed data dimensions:", nrow(scaled_peptides), "rows x", ncol(scaled_peptides), "columns\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")

        # Step 4: Running main PeCorA algorithm
        step_start <- Sys.time()
        cat("Step 4: Running main PeCorA algorithm...\n")
        
        # Runs the main PeCorA logic (data has to be certain format with specific column names)
        disagree_peptides <- PeCorA::PeCorA(scaled_peptides)
        
        cat("  PeCorA algorithm completed\n")
        cat("  Raw results dimensions:", nrow(disagree_peptides), "rows x", ncol(disagree_peptides), "columns\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")

        
        # Step 5: Post-processing and annotation
        step_start <- Sys.time()
        cat("Step 5: Post-processing results and adding annotations...\n")
        
        # Puts the results in a data.table
        pecora_res <- data.table(disagree_peptides)
        # Create peptide_id column by removing the _all from the peptide column
        pecora_res[,"peptide_id":=gsub("_all","",peptide)]
        # Create Protein and Peptide columns by splitting the Peptide.Modified.Sequence column
        # Warning: protein has pro_N and peptide has pep_N and the combined format is pro_N_pep_N
        pecora_res[, c("Protein", "Peptide") := tstrsplit(peptide_id, "-", fixed=TRUE)]
        
        cat("  Converted results to data.table format\n")
    
        # Add pvalue and adj_pvalue columns to the original data by Protein and Peptide
        orig_data <- merge(
            orig_data, 
            pecora_res[, .(Protein, Peptide, pvalue, adj_pval)],
            by=c("Protein", "Peptide")
        )
        # Subset for results only
        orig_data <- unique(
            subset(
                orig_data, 
                select=c("Protein", "Peptide", "pertProtein", "pertPeptide", "pvalue", "adj_pval")
            )
        )
        
        # Count proteins and peptides processed
        proteins_processed <- length(unique(orig_data$Protein))
        peptides_processed <- nrow(pecora_res)
        
        cat("  Processed", proteins_processed, "proteins with", peptides_processed, "peptides\n")
        
        cat("  Final results dimensions:", nrow(pecora_res), "rows x", ncol(pecora_res), "columns\n")
        cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")

        
        # Step 6: Saving results
        step_start <- Sys.time()
        cat("Step 6: Saving results...\n")

        # Output Path: replace '2_' with '3_' and '_InputData.feather' with '_PeCorA_ResultData.feather' in cur_file name
        output_path <- sub("^2_", "3_", cur_file)
        output_path <- sub("_InputData.feather$", "_PeCorA_ResultData.feather", output_path)
        arrow::write_feather(
            orig_data, 
            output_path
        )

        cat("  Results saved to:", output_path, "\n")
        cat("  Output dimensions:", nrow(pecora_res), "rows x", ncol(pecora_res), "columns\n")
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
            output_rows = nrow(pecora_res),
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
cat("               PeCorA benchmark for simulation setups completed.                \n")
cat("================================================================================\n")