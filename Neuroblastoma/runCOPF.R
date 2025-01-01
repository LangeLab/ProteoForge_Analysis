# Initialize time
start_time <- Sys.time()
# Set to not show warnings
options(warn=-1)

suppressMessages(library(arrow, warn.conflicts = FALSE))
suppressMessages(library(data.table, warn.conflicts = FALSE))
suppressMessages(library(CCprofiler, warn.conflicts = FALSE))

cur_wd <- getwd()
# if cur_wd has /Simulation at the end, remove it
if (grepl("/Neuroblastoma$", cur_wd)) {
  setwd(dirname(cur_wd))
}
# Set back to Simulation
setwd(paste0(getwd(), "/Neuroblastoma"))

# Run COPF on the complete and imputed data
input_data_path <- paste0("./data/processed/MatchedPatient_TumorNormal_PeptideLevel_TestData.feather")
# Read the input data (feather format)
orig_data <- arrow::read_feather(input_data_path)
# Convert to data.table
orig_data <- as.data.table(orig_data)
# Make a copy of the original data
input_data <- copy(orig_data)
input_data[,"peptide_id":=paste(Protein, Peptide, sep="-")]

# Rename columns
data.table::setnames(
    input_data,
    c("Peptide","Protein", "Intensity","Condition", "Sample"), 
    c("Peptide","protein_id","intensity","Condition", "filename")
)

# Create an annotation table for the input data
input_annot <- unique(subset(input_data, select=c("filename","Condition")))
# Set the order of the annotation table by Condition and filename
setorderv(input_annot, c("Condition","filename"))
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
            "peptide_id"
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

# Add clustering to each protein d as 1-genecorr with average cluster method
traces_clustered <- CCprofiler::clusterPeptides(
    traces_corr,
    method = "complete", 
    plot = F, 
    PDF = F,
    name="clustering"
)

traces_clusteredInN <- CCprofiler::cutClustersInNreal(
    traces_clustered, 
    clusterN = 2,
    min_peptides_per_cluster = 2
)

cat("Calculating proteoform scores... \n")
traces_scored <- CCprofiler::calculateProteoformScore(traces_clusteredInN)

score_cutoff <- 0.1
adj_pval_cutoff <- 0.01

traces_proteoforms <- CCprofiler::annotateTracesWithProteoforms(
    traces_scored, 
    score_cutoff = score_cutoff, 
    adj_pval_cutoff =  adj_pval_cutoff
)

traces_proteoforms$trace_annotation

results <- traces_proteoforms$trace_annotation
results[, c("Protein", "Peptide") := tstrsplit(id, "-", fixed=TRUE)]

# Save the results in a feather file
output_path <- paste0("./data/results/COPF_NBResults.feather")

arrow::write_feather( as.data.frame(results), output_path )

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
cat(paste("COPF logic run on two versions of the data in", time_taken, "\n"))
