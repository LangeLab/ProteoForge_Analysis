# Initialize overall start time
overall_start_time <- Sys.time()
# Set to not show warnings
options(warn=-1)

suppressMessages(library(arrow, warn.conflicts = FALSE))
suppressMessages(library(data.table, warn.conflicts = FALSE))
suppressMessages(library(PeCorA, warn.conflicts = FALSE))

cur_wd <- getwd()
# if cur_wd has /Benchmark at the end, remove it
if (grepl("/Benchmark$", cur_wd)) {
  setwd(dirname(cur_wd))
}
# Set back to Benchmark
setwd(paste0(getwd(), "/Benchmark"))

# Globals and Constantss
data_ids <- c("1pep", "2pep", "050pep", "random") # Data IDs to process

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

# Print header
cat("================================================================================\n")
cat("                            PeCorA ALGORITHM BENCHMARK                          \n")
cat("================================================================================\n")
cat("Starting PeCorA analysis on multiple datasets...\n")
cat("Data IDs to process:", paste(data_ids, collapse = ", "), "\n")
cat("Start time:", format(overall_start_time), "\n")
cat("================================================================================\n\n")

# Create results directory if it doesn't exist
if (!dir.exists("./data/results")) {
  dir.create("./data/results", recursive = TRUE)
  cat("Created results directory: ./data/results\n")
}

# Initialize results tracking
results_summary <- list()

# Process each data ID
for (i in seq_along(data_ids)) {
  data_id <- data_ids[i]
  
  cat(sprintf("\n[%d/%d] Processing dataset: %s\n", i, length(data_ids), data_id))
  cat("--------------------------------------------------------------------------------\n")
  
  # Initialize time for this dataset
  start_time <- Sys.time()
  
  # Step 1: Data loading
  step_start <- Sys.time()
  cat("Step 1: Loading data...\n")
  # input_data_path <- paste0('./data/processed/bench_', data_id, "_input.feather")
  input_data_path <- paste0('./data/prepared/bench_', data_id, "_input.feather")
  
  # Check if file exists
  if (!file.exists(input_data_path)) {
    cat("  ERROR: File not found:", input_data_path, "\n")
    results_summary[[data_id]] <- list(status = "FAILED", error = "File not found")
    next
  }
  
  # Read the input data (feather format)
  orig_data <- arrow::read_feather(input_data_path)
  
  input_data <- copy(orig_data)
  
  cat("  Data read from:", input_data_path, "\n")
  cat("  Dimensions:", nrow(input_data), "rows x", ncol(input_data), "columns\n")
  cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")
  
  # Convert to data.table
  input_data <- as.data.table(input_data)

  
  # Step 2: Data preparation and formatting for PeCorA
  step_start <- Sys.time()
  cat("Step 2: Preparing data structure for PeCorA...\n")
  
  # Rename columns
  data.table::setnames(
      input_data,
      c("peptide_id","protein_id","intensity","day"), 
      c("Peptide","Protein","Normalized.Area","Condition")
  )
  # Add a new column () with the same values as Peptide
  input_data[,"Peptide.Modified.Sequence":=Peptide] 
  
  cat("  Renamed columns for PeCorA compatibility\n")
  
  # Create an annotation table for the input data
  ann <- unique(subset(input_data, select=c("filename","Condition"))) 
  # Order by Condition and filename
  setorderv(ann,c("Condition","filename")) 
  # Add BioReplicate column 7 replicates
  ann[,BioReplicate:=c(1:7), by="Condition"]
  
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
  cat("  Unique peptides:", length(unique(pecora_df$Peptide)), "\n")
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
      control_name="day1"
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
  
  cat("  Converted results to data.table format\n")
  
  # Get the initial data and create annotation of perturbation 
  trace_annotation <- unique(
      subset(
          orig_data, 
          select=c("peptide_id","n_pep","n_perturbed_peptides", "perturbed_protein","perturbed_peptide","red_fac")
      )
  )
  
  cat("  Created trace annotation with", nrow(trace_annotation), "peptides\n")
  
  # Merge the annotation with the PeCorA results
  pecora_res <- merge(pecora_res,trace_annotation,by="peptide_id")
  # Add a column with the adjusted p-value (BH)
  pecora_res[, adj_adj_pval := p.adjust(adj_pval, method = "BH")]
  
  # Count proteins and peptides processed
  proteins_processed <- length(unique(pecora_res$perturbed_protein))
  peptides_processed <- nrow(pecora_res)
  
  cat("  Processed", proteins_processed, "proteins with", peptides_processed, "peptides\n")
  
  cat("  Final results dimensions:", nrow(pecora_res), "rows x", ncol(pecora_res), "columns\n")
  cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")

  
  # Step 6: Saving results
  step_start <- Sys.time()
  cat("Step 6: Saving results...\n")
  
  # Save the proteoforms
  output_path <- paste0("./data/results/PeCorA_", data_id,"_result.feather")
  arrow::write_feather( as.data.frame(pecora_res), output_path )
  
  cat("  Results saved to:", output_path, "\n")
  cat("  Output dimensions:", nrow(pecora_res), "rows x", ncol(pecora_res), "columns\n")
  cat("  Time taken:", format_time_diff(as.numeric(difftime(Sys.time(), step_start, units = "secs"))), "\n\n")
  
  # Calculate total time for this dataset
  end_time <- Sys.time()
  total_time_diff <- as.numeric(difftime(end_time, start_time, units = "secs"))
  total_time_taken <- format_time_diff(total_time_diff)
  
  # Store results summary
  results_summary[[data_id]] <- list(
    status = "SUCCESS",
    total_time = total_time_taken,
    total_time_seconds = total_time_diff,
    proteins_processed = proteins_processed,
    peptides_processed = peptides_processed,
    input_rows = nrow(orig_data),
    output_rows = nrow(pecora_res),
    output_file = output_path
  )
  
  cat("Dataset", data_id, "completed successfully in", total_time_taken, "\n")
  cat("================================================================================\n")
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
    cat(sprintf("✓ %-8s: %s (%d proteins, %d peptides)\n", 
                data_id, result$total_time, result$proteins_processed, result$peptides_processed))
  } else {
    cat(sprintf("✗ %-8s: FAILED (%s)\n", data_id, result$error))
  }
}

cat("\nDetailed results:\n")
cat("-----------------\n")
for (data_id in names(results_summary)) {
  result <- results_summary[[data_id]]
  if (result$status == "SUCCESS") {
    cat(sprintf("%s:\n", data_id))
    cat(sprintf("  - Processing time: %s\n", result$total_time))
    cat(sprintf("  - Input rows: %d\n", result$input_rows))
    cat(sprintf("  - Output rows: %d\n", result$output_rows))
    cat(sprintf("  - Proteins processed: %d\n", result$proteins_processed))
    cat(sprintf("  - Peptides processed: %d\n", result$peptides_processed))
    cat(sprintf("  - Output file: %s\n", result$output_file))
    cat("\n")
  }
}

# Calculate and display performance metrics
successful_runs <- sum(sapply(results_summary, function(x) x$status == "SUCCESS"))
total_runs <- length(results_summary)

cat(sprintf("Success rate: %d/%d (%.1f%%)\n", 
            successful_runs, total_runs, 
            (successful_runs/total_runs)*100))

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
cat("PeCorA benchmark analysis completed.\n")
cat("================================================================================\n")
