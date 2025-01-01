# Initialize time
start_time <- Sys.time()
# Set to not show warnings
options(warn=-1)

library(arrow, warn.conflicts = FALSE)
library(data.table, warn.conflicts = FALSE)
 # Needs the proteoformmapping branch
library(CCprofiler, warn.conflicts = FALSE)

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
input_data <- arrow::read_feather(input_data_path)

# print(dim(input_data))
cat("Data read from", input_data_path, "with", nrow(input_data), "rows and", ncol(input_data), "columns.\n")

# Convert to data.table
input_data <- as.data.table(input_data)

# Create an annotation table for the input data
input_annot <- unique(subset(input_data, select=c("filename","day")))
# Set the order of the annotation table by day and filename
setorderv(input_annot, c("day","filename"))
# Add a column with the fraction number (from 1 to nrow(input_annot))
input_annot[,fraction_number:=.I]
# Keep only the filename and fraction number columns
input_annot <- subset(input_annot, select=c("filename","fraction_number"))

# Essentially the fancy dictionary with the following subsets:
# - traces: quant_data with samples as columns + id column (tracetype:peptide)
# - trace_type: peptide
# - trace_annotation: id column +  other info_columns here (similar to my info_data)
# - fraction_annotation: metadata for the sample info to be used.
traces <- CCprofiler::importPCPdata(
    input_data = input_data,
    fraction_annotation = input_annot
)

# Get the unique subset for the data with useful columns
trace_annotation <- unique(
    subset(
        input_data, 
        select=c(
            "peptide_id",
            "n_pep",
            "n_perturbed_peptides",
            "perturbed_protein",
            "perturbed_peptide",
            "red_fac"
        )
    )
)

## Annotation of traces steps, which expands the traces list with additionall annotation for proteins etc.
# (seems for this data doesn't do anything maybe since it is at peptide level)
traces <- CCprofiler::annotateTraces(
    traces,
    trace_annotation,
    traces_id_column = "id",
    trace_annotation_id_column = "peptide_id", 
    uniprot_mass_format = FALSE
)

cat(
    "Data placed into trace structure from ccprofiler", 
    nrow(traces$trace_annotation), "rows and", 
    ncol(traces$trace_annotation), "columns.\n"
)

## Remove traces with 0 standard deviation (can't be clustered)

# Binary vector of traces puts TRUE for non-zero variance traces
zerovar <- apply(CCprofiler::getIntensityMatrix(traces), 1, var) > 0
# Subset for the traces with non-zero variance
traces_zerovar <- subset(
    traces,
    trace_subset_ids = names(zerovar[zerovar])
)

#' ## Remove single peptide genes
traces_multiPep <- CCprofiler::filterSinglePeptideHits(traces_zerovar)

cat("Running the clustering... \n")

#' ## Estimate proteoform scores by correlation clustering
traces_corr <- CCprofiler::calculateGeneCorrMatrices(traces_multiPep)
# This stores correlation matrix calculated for peptides stored at $geneCorrMatrices

# Add clustering to each protein d as 1-genecorr with average cluster method
traces_clustered <- CCprofiler::clusterPeptides(
    traces_corr,
    method = "average", 
    plot = F, 
    PDF = F,
    name=paste0("ProteoformClusters_interlab_",name)
)

traces_clusteredInN <- CCprofiler::cutClustersInNreal(
    traces_clustered, 
    clusterN = 2,
    min_peptides_per_cluster = 2
)

traces_clusteredInN$trace_annotation

cat("Calculating proteoform scores... \n")

traces_scored <- CCprofiler::calculateProteoformScore(traces_clusteredInN)
  
score_cutoff <- 0.1
adj_pval_cutoff <- 0.01

traces_proteoforms <- CCprofiler::annotateTracesWithProteoforms(
    traces_scored, 
    score_cutoff = score_cutoff, 
    adj_pval_cutoff =  adj_pval_cutoff
)

cat("Saving the proteoform results into a feather file... \n")
# Save the proteoforms
output_path <- paste0("./data/results/COPF_", data_id,"_result.feather")
arrow::write_feather(
    as.data.frame(traces_proteoforms$trace_annotation), 
    output_path
)

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
cat(paste("COPF logic run on data in", time_taken))