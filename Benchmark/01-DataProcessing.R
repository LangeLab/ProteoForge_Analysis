# Initialize time
start_time <- Sys.time()
# Set to not show warnings
options(warn=-1)

# Load libraries
library(arrow, warn.conflicts = FALSE)
library(data.table, warn.conflicts = FALSE)

cur_wd <- getwd()
# if cur_wd has /Benchmark at the end, remove it
if (grepl("/Benchmark$", cur_wd)) {
  setwd(dirname(cur_wd))
}
# Set back to Benchmark
setwd(paste0(getwd(), "/Benchmark"))

#####################################
### Create dataset ##################
#####################################

dt <- fread("./data/input/site02_global_q_0.01_applied_to_local_global.txt")
cat("Data read from site02_global_q_0.01_applied_to_local_global.txt with", nrow(dt), "rows and", ncol(dt), "columns.\n")

# rename protein
dt[,ProteinName:=gsub("1/","",ProteinName)]
dt <- subset(dt, ProteinName != "iRT_protein") # remove iRT peptides
dt <- subset(dt, ProteinName != "sp|AQUA30|AQUA30") # remove AQUA peptides

# rename runs
dt[,run:=gsub("Site2_AQUA_HEK_","",run_id)]
dt[,run:=gsub("_180714.mzXML.gz","",run)]

# determine day
dt[,day:=gsub("S.*_SW_","",run)]
dt[,day:=gsub("_.*","",day)]

# subset to important columns
dt_sub <- subset(dt, select=c("run","day","ProteinName","FullPeptideName","peptide_group_label", "Intensity"))

# aggregate charge states
dt_sub[,pep_int:=sum(Intensity), by=c("ProteinName","FullPeptideName","run")]
dt_sub <- unique(subset(dt_sub, select=c("run","day","ProteinName","FullPeptideName","pep_int")))

# subset to sufficient data per condition
dt_sub[,n_day1 := sum(day=="day1"), by=c("FullPeptideName")]
dt_sub[,n_day3 := sum(day=="day3"), by=c("FullPeptideName")]
dt_sub[,n_day5 := sum(day=="day5"), by=c("FullPeptideName")]
dt_sub[,min_n_per_day := min(n_day1,n_day3,n_day5),by=c("FullPeptideName")]
dt_sub <- subset(dt_sub, min_n_per_day==7)


# subset to proteins with > 4 peptides:
dt_sub[,n_pep:=length(unique(FullPeptideName)),by=c("ProteinName")]
dt_sub <- dt_sub[n_pep>=4]

# median normalization
dt_sub[,log2_int:=log2(pep_int)]
dt_sub[,median_perRun:=median(log2_int), by="run"]
dt_sub[,median_median:=mean(median_perRun)]
dt_sub[,diff_median:=median_median-median_perRun]
dt_sub[,norm_log2_int := log2_int+diff_median]
dt_sub[,norm_int := 2^norm_log2_int]

# introduce variation
set.seed(1)
dt_sub[,diff_fac_3 := runif(1, min = 1, max = 6), by="ProteinName"]
set.seed(2)
dt_sub[,diff_fac_5 := runif(1, min = 1, max = 6), by="ProteinName"]
#dt_sub[,diff_norm_int:=ifelse(day %in% c("day3","day5"), diff_fac*norm_int, norm_int)]
dt_sub[,diff_norm_int:=ifelse(day == "day3", diff_fac_3*norm_int, norm_int)]
dt_sub[,diff_norm_int:=ifelse(day == "day5", diff_fac_5*norm_int, diff_norm_int)]

# randomly select 1000 proteins to perturb
set.seed(22)
proteins_to_perturb = sample(unique(dt_sub$ProteinName),1000)
dt_sub[,perturbed_protein := ifelse(ProteinName %in% proteins_to_perturb, TRUE, FALSE), by="ProteinName"]

# determine reduction factor for each protein
# sample from uniform distribution
set.seed(44)
dt_sub[,red_fac:=ifelse(perturbed_protein, runif(1, min = 0.01, max = 0.90), 1), by="ProteinName"]

cat("Generating perturbed profiles...\n")
# Function to generate perturbed profiles
generatePerturbedProfiles <- function(input_data, nf_peptides_to_perturb="random"){
    dt_input <- copy(input_data)
    
    set.seed(66)
    
    # if (nf_peptides_to_perturb == "random") {
    #     dt_input[,max_perturbed_peptides := floor(n_pep*0.5), by="ProteinName"] 
    #     dt_input[,n_perturbed_peptides := max(sample(seq(2,max(2,max_perturbed_peptides),1), 1),2), by="ProteinName"]
    # } else if (nf_peptides_to_perturb >= 1) {
    #     dt_input[,n_perturbed_peptides := nf_peptides_to_perturb, by="ProteinName"]
    # } else if (nf_peptides_to_perturb < 1) {
    #     dt_input[,n_perturbed_peptides := floor(n_pep*nf_peptides_to_perturb), by="ProteinName"]
    #     dt_input[,n_perturbed_peptides := ifelse(n_perturbed_peptides<2,2,n_perturbed_peptides), by="ProteinName"]
    # }
    # original one used ceiling (which perturbed more than half for odd numbers)
    # This is a wrong and misleading, however I want to keep it for reproducibility and comparison
    if (nf_peptides_to_perturb == "random") {
        dt_input[,max_perturbed_peptides := ceiling(n_pep*0.5), by="ProteinName"]
        dt_input[,n_perturbed_peptides := max(sample(seq(2,max(2,max_perturbed_peptides),1), 1),2), by="ProteinName"]
    } else if (nf_peptides_to_perturb >= 1) {
        dt_input[,n_perturbed_peptides := nf_peptides_to_perturb, by="ProteinName"]
    } else if (nf_peptides_to_perturb < 1) {
        dt_input[,n_perturbed_peptides := ceiling(n_pep*nf_peptides_to_perturb), by="ProteinName"]
        dt_input[,n_perturbed_peptides := ifelse(n_perturbed_peptides<2,2,n_perturbed_peptides), by="ProteinName"]
    }
    
    dt_input[,perturbed_peptides := paste(unique(FullPeptideName)[c(1:n_perturbed_peptides)],collapse=";"), by = c("ProteinName")]
    
    # reduce day 5 by reduction factor for peptides in perturbed peptides
    dt_input[,perturbed_peptide:=((FullPeptideName %in% unlist(strsplit(perturbed_peptides,";"))) & (perturbed_protein==TRUE)), by=c("FullPeptideName","ProteinName")]
    
    dt_input[,mod_pep_int:=ifelse(((day=="day5") & (perturbed_peptide) & (perturbed_protein)), diff_norm_int*red_fac, diff_norm_int)]
    
    setnames(dt_input, c("FullPeptideName","ProteinName","run","mod_pep_int"), c("peptide_id","protein_id","filename","intensity"))
    
    # Select columns relevant for the benchmark
    dt_input <- dt_input[,c("protein_id","peptide_id","day","filename","intensity","n_pep","n_perturbed_peptides","perturbed_protein","perturbed_peptide","red_fac")]
    return(dt_input)
}

# Check if ./data/prepared exists, if not create it
if (!dir.exists("./data/prepared")) {
    dir.create("./data/prepared")
}

# Random Perturbation
arrow::write_feather(
    generatePerturbedProfiles(input_data=dt_sub, nf_peptides_to_perturb="random"),
    "./data/prepared/bench_random_input.feather"
)
# 1 Peptide Perturbation
arrow::write_feather(
    generatePerturbedProfiles(input_data=dt_sub, nf_peptides_to_perturb=1), 
    "./data/prepared/bench_1pep_input.feather"
)
# 2 Peptide Perturbation

arrow::write_feather(
    generatePerturbedProfiles(input_data=dt_sub, nf_peptides_to_perturb=2), 
    "./data/prepared/bench_2pep_input.feather"
)
# 2 to 50% Peptide Perturbation
arrow::write_feather(
    generatePerturbedProfiles(input_data=dt_sub, nf_peptides_to_perturb=0.5), 
    "./data/prepared/bench_050pep_input.feather"
)

# Print the time taken
end_time <- Sys.time()

#Pretty print the time taken
cat(paste("Data prepared and saved in", round(as.numeric(end_time - start_time), 2), "seconds\n"))