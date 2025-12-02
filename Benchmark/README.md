# Benchmarking Framework for Proteoform Detection Methods

## Overview

This directory contains a benchmarking framework that evaluates proteoform detection methods at two biological levels: **peptide identification** and **protein-level proteoform grouping**. We use the same raw SWATH-MS interlab data and apply the same perturbation generation approach as the original COPF manuscript, then run COPF, PeCorA, and ProteoForge on identical prepared datasets.

The original COPF benchmark compared protein-level proteoform grouping between COPF and PeCorA. Since PeCorA was designed for peptide-level detection rather than grouping, we extended the framework to evaluate each method at its intended level:

1. **Discordant peptide identification benchmark** — where PeCorA and ProteoForge excel
2. **Peptide grouping benchmark** — where COPF and ProteoForge are directly comparable

**Key Contributions:**

- Standard machine learning metrics (ROC-AUC, PR-AUC, F1, MCC) for proteoform detection
- Multi-level evaluation framework matching each method's intended capabilities
- Fair comparison using identically prepared datasets
- Reproducible pipeline with documented justifications

## Background and Motivation

Proteoform detection methods are critical for understanding protein diversity, but standardized benchmarking practices are still developing. After studying the original COPF benchmarking approach, we identified opportunities to extend the evaluation:

1. **Multi-level evaluation:** The original benchmark focused on protein-level assessment; we added peptide-level evaluation
2. **Standard metrics:** We complemented the custom "perfectness" metrics with widely-used ML metrics (ROC-AUC, F1, MCC)
3. **Method-appropriate comparisons:** We evaluated each method at the level it was designed for (e.g., PeCorA for peptide identification, COPF for grouping)

## Dataset and Experimental Design

**Ensuring Fair Comparison Through Identical Data:**

We use the **same raw SWATH-MS interlab data** from the original COPF publication and apply the **same perturbation generation procedure** described in their manuscript. Rather than using the pre-processed `.rds` file (`interlab_benchmark_data.rds`), we regenerate the benchmark datasets from the raw input (`site02_global_q_0.01_applied_to_local_global.txt`) using `01-DataProcessing.R`. This ensures transparency and allows others to verify or modify the perturbation parameters.

**Dataset Characteristics:**

- **Raw Input:** `site02_global_q_0.01_applied_to_local_global.txt` (SWATH-MS interlab study, Site 2)
- **Origin:** Inter-laboratory HEK293 cell proteomics study
- **Perturbation Design:** 1,000 randomly selected proteins with engineered abundance changes
- **Temporal Conditions:** Three time points (day1, day3, day5) with 7 biological replicates each
- **Perturbation Application:** Intensity reduction applied to selected peptides in day5 samples
- **Ground Truth Labels:** `perturbed_peptide` (peptide-level) and `perturbed_protein` (protein-level)

## Our Comprehensive Benchmarking Framework

### Multi-Level Evaluation Strategy

We developed a two-tiered evaluation approach that addresses the limitations of existing frameworks:

1. **Peptide Identification Benchmark** (`05-IdentificationBenchmark.ipynb`) — Evaluates all four perturbation scenarios (`1pep`, `2pep`, `050pep`, `random`)
2. **Proteoform Grouping Benchmark** (`06-GroupingBenchmark.ipynb`) — Evaluates three scenarios (`2pep`, `050pep`, `random`); the `1pep` scenario is excluded because a single perturbed peptide cannot meaningfully group with unperturbed peptides, making this scenario trivial and unfair for clustering-based methods like COPF

**Why This Dual Approach Matters:**

**COPF's Design:** COPF was designed for protein-level proteoform detection through clustering. Its original benchmark appropriately focused on this level, though it didn't assess peptide-level accuracy separately.

**PeCorA's Design:** PeCorA identifies discordant peptides but doesn't group them into proteoforms, so protein-level grouping evaluation isn't appropriate for it.

**Our Extension:** By implementing both levels, we can fairly evaluate each method according to its intended capabilities while providing a more complete picture of performance.

### Peptide Identification Benchmark

**Objective:** Evaluate accuracy of identifying individual perturbed peptides

**Methods Evaluated:**

- **ProteoForge:** Direct evaluation of peptide discordance detection
- **PeCorA:** Direct evaluation as this is PeCorA's primary function  
- **COPF:** Included for completeness, though COPF is not designed for peptide-level predictions

**Scenarios:** All four (`1pep`, `2pep`, `050pep`, `random`)

### Proteoform Grouping Benchmark

**Objective:** Evaluate accuracy of identifying proteins containing proteoforms (multiple peptide groups)

**Methods Evaluated:**

- **ProteoForge:** Evaluation of proteoform grouping capability
- **COPF:** Direct evaluation as this is COPF's primary function
- **PeCorA:** Not included — PeCorA does not perform peptide grouping

**Scenarios:** Three only (`2pep`, `050pep`, `random`) — the `1pep` scenario is excluded because COPF's clustering requires at least two peptides to form a meaningful proteoform group. Including `1pep` would unfairly penalize COPF for a scenario outside its design scope.

## Understanding COPF's Original Benchmarking Approach

To design our evaluation framework, we studied the original COPF benchmarking strategy from the CCprofiler R scripts. This helped us understand their design choices and identify opportunities for a more comprehensive evaluation.

### COPF's "Perfectness" Metric Framework

The COPF evaluation used a "perfectness" requirement to assess grouping accuracy:

```r
# From COPF benchmarking script
traces_proteoforms$trace_annotation[, correct_peptide := 
    ((n_proteoforms_per_perturbed_group==1) & 
     (n_perturbed_groups_per_proteoform==1))]

traces_proteoforms$trace_annotation[, protein_without_mistakes := 
    all(correct_peptide), by="protein_id"]
```

**Observations from this approach:**

1. **Binary Perfectness Requirement:** This required all peptides within a protein to be perfectly grouped—any misclassification invalidated the entire protein. While rigorous, this may be stricter than necessary for practical applications.

2. **Protein-Only Evaluation:** The evaluation focused on the protein level ("Does this protein have proteoforms?") without separately assessing peptide-level accuracy.

3. **Custom Metrics:** The "perfectness" framework used metrics specific to this study, which made comparison with other computational biology benchmarks difficult.

### COPF's Performance Calculation

The COPF framework calculated performance at the protein level:

```r
# From COPF benchmarking script
res$TP[idx] <- length(unique(traces_proteoforms$trace_annotation[
    (n_proteoforms>1) & (perturbed_protein)]$protein_id))
res$FP[idx] <- length(unique(traces_proteoforms$trace_annotation[
    (n_proteoforms>1) & (perturbed_protein==FALSE)]$protein_id))
```

**What we noted:**

- **Protein-level aggregation:** This approach loses peptide-level granularity, which is important for understanding method sensitivity
- **Scope alignment:** This evaluation naturally favors methods designed for protein-level detection, which was appropriate for COPF's goals but less suitable for comparing methods with different objectives
- **Opportunity for extension:** We saw an opportunity to complement this with peptide-level evaluation for a more complete picture

### Our Standard Metrics Approach

To complement the original evaluation approach, we adopted established machine learning metrics that are:

- Widely understood across computational biology
- Comparable with other studies
- Threshold-independent when using ROC/PR curves
- Statistically well-founded

**Our Metric Suite:**

| Metric | Description |
|--------|-------------|
| **TPR (Recall)** | Fraction of true positives correctly identified |
| **FPR** | Fraction of negatives incorrectly identified |
| **Precision** | Fraction of positive predictions that are correct |
| **F1 Score** | Harmonic mean of precision and recall |
| **MCC** | Balanced metric accounting for all confusion matrix cells |
| **AUC** | Threshold-independent area under ROC curve |

**Comparison of Approaches:**

| Aspect | Our Approach | Original COPF Approach |
|--------|-------------|---------------|
| **Metrics** | Standard ML metrics (TPR, FPR, AUC, MCC) | Custom "perfectness" metrics |
| **Evaluation** | Threshold-independent ROC/PR curves | Fixed threshold evaluation |
| **Comparability** | Directly comparable with other studies | Study-specific |
| **Granularity** | Both peptide and protein levels | Protein level |

## Reproducibility and Implementation

**Execution Order:**

1. `01-DataProcessing.R` — Prepare datasets from raw SWATH-MS data
2. `02-runCOPF.R` — Run COPF on all scenarios
3. `03-runPeCorA.R` — Run PeCorA on all scenarios  
4. `04-runProteoForge.py` — Run ProteoForge on all scenarios
5. `05-IdentificationBenchmark.ipynb` — Analyze peptide identification performance
6. `06-GroupingBenchmark.ipynb` — Analyze peptide grouping performance

**Transparency:**

- All code is available for scrutiny and replication
- Same raw data and perturbation procedure as the original COPF publication
- Standard evaluation protocols from computational biology

## Files and Implementation

### Data Preparation

- `01-DataProcessing.R`: Prepares the SWATH-MS interlab data, creates perturbation scenarios, and outputs prepared datasets to `data/prepared/` as `.feather` files

### Method Execution Scripts

- `02-runCOPF.R`: Runs COPF on all four scenarios; outputs to `data/results/COPF_*_result.feather`
- `03-runPeCorA.R`: Runs PeCorA on all four scenarios; outputs to `data/results/PeCorA_*_result.feather`
- `04-runProteoForge.py`: Runs ProteoForge on all four scenarios; outputs to `data/results/ProteoForge_*_result.feather`

### Benchmark Analysis Notebooks

- `05-IdentificationBenchmark.ipynb`: Peptide-level discordant peptide identification evaluation (all 4 scenarios; compares ProteoForge, PeCorA, and COPF)
- `06-GroupingBenchmark.ipynb`: Protein-level proteoform grouping evaluation (3 scenarios excluding `1pep`; compares ProteoForge and COPF only)

### Utility Modules (in `../src/`)

- `utils.create_metric_data`: Standardized metric calculation framework
- `utils.calculate_metrics`: Standard classification metrics implementation
