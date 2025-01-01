# Initialize time
start_time <- Sys.time()
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


# Get the path from command line
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Please provide the id indicator for the data (eg. oneN, halfN, etc.).")
}

data_id <- args[1]
# input_data_path <- paste0('./data/processed/bench_', data_id, "_input.feather")
input_data_path <- paste0('./data/prepared/bench_', data_id, "_input.feather")

# Read the input data (feather format)
orig_data <- arrow::read_feather(input_data_path)

input_data <- copy(orig_data)

# print(dim(input_data))
cat("Data read from", input_data_path, "with", nrow(input_data), "rows and", ncol(input_data), "columns.\n")

# Convert to data.table
input_data <- as.data.table(input_data)

# Rename columns
data.table::setnames(
    input_data,
    c("peptide_id","protein_id","intensity","day"), 
    c("Peptide","Protein","Normalized.Area","Condition")
)
# Add a new column () with the same values as Peptide
input_data[,"Peptide.Modified.Sequence":=Peptide] 

# Create an annotation table for the input data
ann <- unique(subset(input_data, select=c("filename","Condition"))) 
# Order by Condition and filename
setorderv(ann,c("Condition","filename")) 
# Add BioReplicate column 7 replicates
ann[,BioReplicate:=c(1:7), by="Condition"]

# Merge the annotation table with the input data
input_data <- merge(input_data,ann,by=c("filename","Condition"))

# Subset the data to keep only the useful columns
input_data <- subset(
    input_data, 
    select=c("Protein","Peptide","Peptide.Modified.Sequence","Condition","BioReplicate","Normalized.Area")
)

# Move the data.table to a data.frame
pecora_df <- setDF(input_data)

# Preprocess the data for PeCorA (Super Slow - 1.97m)
scaled_peptides <- PeCorA::PeCorA_preprocessing(
    pecora_df,
    area_column_name=6,
    threshold_to_filter=min(pecora_df$Normalized.Area),
    control_name="day1"
)

# Runs the main PeCorA logic (data has to be certain format with specific column names)
disagree_peptides <- PeCorA::PeCorA(scaled_peptides)

# Puts the results in a data.table
pecora_res <- data.table(disagree_peptides)
# Create peptide_id column by removing the _all from the peptide column
pecora_res[,"peptide_id":=gsub("_all","",peptide)]

# Get the initial data and create annotation of perturbation 
trace_annotation <- unique(
    subset(
        orig_data, 
        select=c("peptide_id","n_pep","n_perturbed_peptides", "perturbed_protein","perturbed_peptide","red_fac")
    )
)

# Merge the annotation with the PeCorA results
pecora_res <- merge(pecora_res,trace_annotation,by="peptide_id")
# Add a column with the adjusted p-value (BH)
pecora_res[, adj_adj_pval := p.adjust(adj_pval, method = "BH")]

cat("Saving the proteoform results into a feather file... \n")
# Save the proteoforms
output_path <- paste0("./data/results/PeCorA_", data_id,"_result.feather")
arrow::write_feather( as.data.frame(pecora_res), output_path )

# Print the time taken
end_time <- Sys.time()

#Pretty print the time taken
# Calculate the time difference
time_diff <- as.numeric(difftime(end_time, start_time, units = "secs"))
# Convert the time difference to a human-readable format
if (time_diff < 60) {
    time_taken <- paste(round(time_diff, 2), "seconds")
} else if (time_diff < 3600) {
    time_taken <- paste(round(time_diff / 60, 2), "minutes")
} else {
    time_taken <- paste(round(time_diff / 3600, 2), "hours")
}
# Pretty print the time taken
cat(paste("PeCorA logic run on data in", time_taken))
