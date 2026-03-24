# ==============================================================================
# run_r_methods.R
#
# FDR calibration for COPF and PeCorA under null simulation.
# Companion to the FDRCalibration notebook (ProteoForge calibration).
#
# Prerequisite: run generate_null_data.py first to create
#     Revisions/outputs/simulation/fdr_calibration/null_input/null_run_{k}_InputData.feather
#
# Output CSVs (loaded by the FDRCalibration notebook for comparison plots):
#     Revisions/outputs/simulation/fdr_calibration/rmethod_id_fpr_K<K>.csv
#     Revisions/outputs/simulation/fdr_calibration/rmethod_grp_fpr_K<K>.csv
#
# Usage (from project root):
#     PF_SIM_K=50 Rscript Revisions/logic/r/simulation/fdr_calibration/run_r_methods.R
#     Rscript Revisions/logic/r/simulation/fdr_calibration/run_r_methods.R
# ==============================================================================

overall_start <- Sys.time()
options(warn = -1)

# ── Environment ────────────────────────────────────────────────────────────────
# Resolve project root: this script lives at
#   Revisions/logic/r/simulation/fdr_calibration/run_r_methods.R
# so project root is 5 levels up.
script_dir <- tryCatch(
    dirname(sys.frame(1)$ofile),
    error = function(e) getwd()
)

# If run via Rscript, script_dir resolves correctly.
# If sourced interactively, assume cwd is project root.
if (grepl("fdr_calibration$", script_dir)) {
    project_root <- normalizePath(file.path(script_dir, "..", "..", "..", "..", ".."))
} else {
    project_root <- getwd()
}
cat("Project root:", project_root, "\n")

# Activate renv from project root
renv_activate <- file.path(project_root, "renv", "activate.R")
if (file.exists(renv_activate)) source(renv_activate)

setwd(project_root)

suppressMessages(library(arrow,      warn.conflicts = FALSE))
suppressMessages(library(data.table, warn.conflicts = FALSE))
suppressMessages(library(CCprofiler, warn.conflicts = FALSE))
suppressMessages(library(PeCorA,     warn.conflicts = FALSE))
suppressMessages(library(stringr,    warn.conflicts = FALSE))

# ── Parameters ─────────────────────────────────────────────────────────────────
env_k <- Sys.getenv("PF_SIM_K", unset = "50")
K <- suppressWarnings(as.integer(env_k))
if (is.na(K) || K < 1) {
    stop("PF_SIM_K must be a positive integer. Got: ", env_k)
}
score_cutoff   <- 0.0      # permissive: keep all COPF peptides
adj_pval_cutoff <- 1.1     # permissive: keep all COPF proteins regardless of pval

output_dir <- file.path(project_root, "Revisions", "outputs", "simulation", "fdr_calibration")
input_dir  <- file.path(output_dir, "null_input")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ── Alpha grid (mirrors FDRCalibration notebook exactly) ──────────────────────
alpha_log    <- exp(seq(log(1e-15), log(0.01), length.out = 25))
alpha_mid    <- seq(0.01, 0.10, by = 0.005)
alpha_coarse <- seq(0.10, 0.90, by = 0.10)
alpha_grid   <- sort(unique(round(c(alpha_log, alpha_mid, alpha_coarse), 15)))

cat("================================================================================\n")
cat("     FDR CALIBRATION — R Methods (COPF & PeCorA)   K =", K, "\n")
cat("================================================================================\n")
cat("PF_SIM_K:", K, "\n")
cat("Alpha grid:", length(alpha_grid), "points\n")
cat("Input dir: ", input_dir, "\n")
cat("Output dir:", output_dir, "\n\n")

# ── Helper: format elapsed time ────────────────────────────────────────────────
fmt_time <- function(secs) {
    if (secs < 60)   return(sprintf("%.1f sec", secs))
    if (secs < 3600) return(sprintf("%.1f min", secs / 60))
    return(sprintf("%.2f hr", secs / 3600))
}

# ── Result collectors ──────────────────────────────────────────────────────────
copf_id_rows  <- list()
copf_grp_rows <- list()
pec_id_rows   <- list()

# ==============================================================================
# Main loop over K null runs
# ==============================================================================
for (k in 0:(K - 1)) {
    run_idx <- k + 1
    cat(sprintf("--- Run %d/%d ---\n", run_idx, K))

    input_file  <- file.path(input_dir,  sprintf("null_run_%d_InputData.feather", k))
    copf_cache  <- file.path(output_dir, sprintf("copf_null_run%02d.feather",     k))
    pec_cache   <- file.path(output_dir, sprintf("pecora_null_run%02d.feather",   k))

    if (!file.exists(input_file)) {
        cat("  ERROR: input data not found:", input_file, "\n",
            "  Run generate_null_data.py first.\n\n")
        next
    }

    # ==========================================================================
    # COPF pipeline
    # ==========================================================================
    if (!file.exists(copf_cache)) {
        cat("  [COPF] Starting pipeline...\n")
        t0 <- Sys.time()

        input_data <- as.data.table(arrow::read_feather(input_file))
        input_data[, peptide_id := paste(Protein, Peptide, sep = "-")]
        setnames(input_data,
                 c("Protein", "Intensity", "Sample"),
                 c("protein_id", "intensity", "filename"))

        input_annot <- unique(input_data[, .(filename, Condition)])
        setorderv(input_annot, c("Condition", "filename"))
        input_annot[, fraction_number := .I]
        input_annot <- input_annot[, .(filename, fraction_number)]

        trace_annot <- unique(input_data[, .(peptide_id, pertProtein, pertPeptide, pertPFG)])
        trace_annot <- trace_annot[!duplicated(peptide_id)]

        cat("  [COPF] Importing PCP data...\n")
        traces <- CCprofiler::importPCPdata(
            input_data        = input_data,
            fraction_annotation = input_annot
        )
        traces <- CCprofiler::annotateTraces(
            traces,
            trace_annot,
            traces_id_column       = "id",
            trace_annotation_id_column = "peptide_id",
            uniprot_mass_format    = FALSE
        )

        zerovar <- apply(CCprofiler::getIntensityMatrix(traces), 1, var) > 0
        cat("  [COPF] Zero-variance traces:", sum(!zerovar), "/", length(zerovar), "\n")
        traces <- subset(traces, trace_subset_ids = names(zerovar[zerovar]))
        traces <- CCprofiler::filterSinglePeptideHits(traces)
        cat("  [COPF] After filtering:", nrow(traces$trace_annotation),
            "peptides,", length(traces$geneCorrMatrices), "→ computing corr...\n")

        traces <- CCprofiler::calculateGeneCorrMatrices(traces)

        traces <- CCprofiler::clusterPeptides(
            traces,
            method = "average",
            plot   = FALSE,
            PDF    = FALSE,
            name   = sprintf("COPF_null_run%02d", k)
        )
        traces <- CCprofiler::cutClustersInNreal(
            traces,
            clusterN               = 2,
            min_peptides_per_cluster = 2
        )

        cat("  [COPF] Scoring proteoforms...\n")
        traces <- CCprofiler::calculateProteoformScore(traces)

        traces_final <- CCprofiler::annotateTracesWithProteoforms(
            traces,
            score_cutoff    = score_cutoff,
            adj_pval_cutoff = adj_pval_cutoff
        )

        ta <- as.data.frame(traces_final$trace_annotation)
        ta$run <- run_idx

        arrow::write_feather(ta, copf_cache)
        elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        cat(sprintf(
            "  [COPF] Done in %s → %d peptides, %d proteins\n",
            fmt_time(elapsed),
            nrow(ta),
            length(unique(ta$protein_id))
        ))
    } else {
        cat("  [COPF] Using cache →", basename(copf_cache), "\n")
    }

    # ── Load COPF results & compute FPR curves ─────────────────────────────────
    copf_df <- as.data.table(arrow::read_feather(copf_cache))

    n_pep_copf <- nrow(copf_df)
    for (a in alpha_grid) {
        n_rej <- sum(copf_df$proteoform_score_pval < a, na.rm = TRUE)
        copf_id_rows[[length(copf_id_rows) + 1]] <- list(
            method   = "COPF",
            run      = run_idx,
            alpha    = a,
            fpr      = n_rej / n_pep_copf,
            n_total  = n_pep_copf,
            n_reject = n_rej
        )
    }

    n_prot_copf <- uniqueN(copf_df$protein_id)
    for (a in alpha_grid) {
        grp <- copf_df[, .(
            n_cluster = uniqueN(cluster),
            any_sig   = any(proteoform_score_pval < a, na.rm = TRUE)
        ), by = protein_id]
        n_fp <- sum(grp$any_sig & grp$n_cluster > 1L)
        copf_grp_rows[[length(copf_grp_rows) + 1]] <- list(
            method  = "COPF",
            run     = run_idx,
            alpha   = a,
            fpr     = n_fp / n_prot_copf,
            n_total = n_prot_copf,
            n_fp    = n_fp
        )
    }

    # ==========================================================================
    # PeCorA pipeline
    # ==========================================================================
    if (!file.exists(pec_cache)) {
        cat("  [PeCorA] Starting pipeline...\n")
        t0 <- Sys.time()

        orig_data  <- arrow::read_feather(input_file)
        input_data <- as.data.table(copy(orig_data))
        setnames(input_data,
                 c("Intensity", "Sample"),
                 c("Normalized.Area", "filename"))
        input_data[, Peptide.Modified.Sequence := paste(Protein, Peptide, sep = "-")]

        ann <- unique(input_data[, .(filename, Condition)])
        setorderv(ann, c("Condition", "filename"))
        ann[, BioReplicate := as.integer(
            stringr::str_extract(filename, "(?<=-)([1-9]|10)$")
        )]
        input_data <- merge(input_data, ann, by = c("filename", "Condition"))
        input_data <- input_data[, .(
            Protein, Peptide, Peptide.Modified.Sequence,
            Condition, BioReplicate, Normalized.Area
        )]
        pecora_df <- setDF(input_data)

        cat(sprintf(
            "  [PeCorA] %d peptides × %d samples, running preprocessing...\n",
            length(unique(pecora_df$Peptide.Modified.Sequence)),
            length(unique(paste(pecora_df$Condition, pecora_df$BioReplicate)))
        ))

        t_pre <- Sys.time()
        scaled_peptides <- PeCorA::PeCorA_preprocessing(
            pecora_df,
            area_column_name   = 6,
            threshold_to_filter = min(pecora_df$Normalized.Area),
            control_name       = "control"
        )
        cat("  [PeCorA] Preprocessing:", fmt_time(
            as.numeric(difftime(Sys.time(), t_pre, units = "secs"))), "\n")

        t_pec <- Sys.time()
        disagree <- PeCorA::PeCorA(scaled_peptides)
        cat("  [PeCorA] Main algorithm:", fmt_time(
            as.numeric(difftime(Sys.time(), t_pec, units = "secs"))), "\n")

        pecora_res <- as.data.table(disagree)
        pecora_res[, peptide_id := gsub("_all", "", peptide)]
        pecora_res[, c("Protein", "Peptide") := tstrsplit(peptide_id, "-", fixed = TRUE)]

        orig_dt <- as.data.table(orig_data)
        pert_info <- unique(orig_dt[, .(Protein, Peptide, pertProtein, pertPeptide, pertPFG)])

        final_pec <- merge(
            pecora_res[, .(Protein, Peptide, pvalue, adj_pval)],
            pert_info,
            by = c("Protein", "Peptide")
        )
        final_pec$run <- run_idx

        arrow::write_feather(as.data.frame(final_pec), pec_cache)
        elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        cat(sprintf("  [PeCorA] Done in %s → %d peptides\n",
                    fmt_time(elapsed), nrow(final_pec)))
    } else {
        cat("  [PeCorA] Using cache →", basename(pec_cache), "\n")
    }

    # ── Load PeCorA results & compute identification FPR ──────────────────────
    pec_df    <- as.data.table(arrow::read_feather(pec_cache))
    n_pep_pec <- nrow(pec_df)

    for (a in alpha_grid) {
        n_rej <- sum(pec_df$adj_pval < a, na.rm = TRUE)
        pec_id_rows[[length(pec_id_rows) + 1]] <- list(
            method   = "PeCorA",
            run      = run_idx,
            alpha    = a,
            fpr      = n_rej / n_pep_pec,
            n_total  = n_pep_pec,
            n_reject = n_rej
        )
    }

    cat("\n")
}

# ==============================================================================
# Compile and save results
# ==============================================================================
cat("================================================================================\n")
cat("Compiling results...\n")

id_results  <- rbind(rbindlist(copf_id_rows), rbindlist(pec_id_rows))
grp_results <- rbindlist(copf_grp_rows)

id_path  <- file.path(output_dir, sprintf("rmethod_id_fpr_K%d.csv",  K))
grp_path <- file.path(output_dir, sprintf("rmethod_grp_fpr_K%d.csv", K))

fwrite(id_results,  id_path)
fwrite(grp_results, grp_path)

cat("Identification FPR saved →", id_path,  "\n")
cat("Grouping FPR saved       →", grp_path, "\n")

# ── Summary table ─────────────────────────────────────────────────────────────
cat("\nMedian FPR at α = 0.05 (across runs):\n")
cat("--------------------------------------\n")

for (m in unique(id_results$method)) {
    sub <- id_results[method == m & abs(alpha - 0.05) < 1e-10]
    if (nrow(sub) > 0) {
        cat(sprintf("  %-8s ID FPR:  %.4f (median across %d runs)\n",
                    m, median(sub$fpr, na.rm = TRUE), nrow(sub)))
    }
}
if (nrow(grp_results) > 0) {
    sub <- grp_results[abs(alpha - 0.05) < 1e-10]
    cat(sprintf("  %-8s Grp FPR: %.4f (median across %d runs)\n",
                "COPF", median(sub$fpr, na.rm = TRUE), nrow(sub)))
}

cat("\nTotal time:", fmt_time(
    as.numeric(difftime(Sys.time(), overall_start, units = "secs"))
), "\n")
cat("================================================================================\n")
cat("Done. Load the CSVs in the FDRCalibration notebook to add R-method curves.\n")
cat("================================================================================\n")
