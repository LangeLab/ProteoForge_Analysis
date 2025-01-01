# Initialize time
start_time <- Sys.time()
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

# Get the path from command line
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Please provide the experiment folder name to run the simulated data on PeCorA.")
}
# Get the simulation name (folder)
data_id <- args[1]

cat("Running PeCorA on the simulated data in", data_id, "folder\n")
conditions <- c(2,3,4,5)
status <- c('Overlap', 'NonOverlap')

for (cond in conditions){
    for (isOverlap in status){
        
        # Read the input data (feather format) with path
        input_data_path <- paste0('./data/exp/', data_id, "/", cond, "Cond_", isOverlap, "_InputData.feather")
        # Read the input data (feather format)
        orig_data <- arrow::read_feather(input_data_path)
        # Convert to data.table
        orig_data <- as.data.table(orig_data)
        # Make a copy of the original data
        input_data <- copy(orig_data)
        # Rename columns
        data.table::setnames(
            input_data,
            c("Peptide","Protein","Intensity","Condition", "Sample"), 
            c("Peptide","Protein","Normalized.Area","Condition", "filename")
        )
        # Add a new column () with the same values as Peptide
        # Make the Peptide.Modified.Sequence column with Protein_Peptide
        input_data[,"Peptide.Modified.Sequence":=paste(Protein, Peptide, sep="-")]
        
        # Create an annotation table for the input data
        ann <- unique(subset(input_data, select=c("filename","Condition"))) 
        # Order by Condition and filename
        setorderv(ann,c("Condition","filename")) 
        # Add BioReplicate column 11 replicates
        ann[,BioReplicate:=c(1:10), by="Condition"]
        
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
            control_name="control"
        )
        # Runs the main PeCorA logic (data has to be certain format with specific column names)
        disagree_peptides <- PeCorA::PeCorA(scaled_peptides)

        # Puts the results in a data.table
        pecora_res <- data.table(disagree_peptides)
        # Create peptide_id column by removing the _all from the peptide column
        pecora_res[,"peptide_id":=gsub("_all","",peptide)]
        # Create Protein and Peptide columns by splitting the Peptide.Modified.Sequence column
        # Warning: protein has pro_N and peptide has pep_N and the combined format is pro_N_pep_N
        pecora_res[, c("Protein", "Peptide") := tstrsplit(peptide_id, "-", fixed=TRUE)]
        
        # Add pvalue and adj_pvalue columns to the original data by Protein and Peptide
        orig_data <- merge(
            orig_data, 
            pecora_res[, .(Protein, Peptide, pvalue, adj_pval)],
            by=c("Protein", "Peptide")
        )

        cat("Saving the proteoform results into a feather file... \n")
        # Save the results in a feather file
        output_path <- paste0('./data/exp/', data_id, "/", cond, "Cond_", isOverlap, "_PeCorA_ModelResults.feather")
        arrow::write_feather( as.data.frame(orig_data), output_path )
    }
}

# Print the time taken
end_time <- Sys.time()

# Pretty print the time taken
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
cat(paste("PeCorA logic run on for all available data in", time_taken, "\n"))