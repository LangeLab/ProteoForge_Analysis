# NSCLC/Hypoxia Application (Tomin et al., 2025)

## Overview

This directory contains an application of ProteoForge to datasets associated with Tomin et al., 2025: “Increased antioxidative defense and reduced advanced glycation end-product formation by metabolic adaptation in non-small-cell-lung-cancer patients” ([manuscript](https://www.nature.com/articles/s41467-025-60326-y)). Raw data is available via PRIDE: [PXD052340](https://www.ebi.ac.uk/pride/archive/projects/PXD052340), [PXD062503](https://www.ebi.ac.uk/pride/archive/projects/PXD062503), and [PXD061610](https://www.ebi.ac.uk/pride/archive/projects/PXD061610).

The notebooks implement a reproducible flow: data preparation → applying ProteoForge at 48 hr, 72 hr, and combined → summarization → downstream comparisons and enrichment.

## Contents

- `01-DataPreparation.ipynb` — prepares hypoxia inputs from PRIDE; writes standardized artifacts under `data/cleaned/hypoxia/`.
- `02-ApplyingProteoForge_48hr.ipynb` — applies ProteoForge for the 48 hr condition; outputs tables to `data/results/hypoxia/` and figures under `figures/hypoxia/02-ProteoForge/`.
- `02-ApplyingProteoForge_72hr.ipynb` — applies ProteoForge for the 72 hr condition; outputs and figures analogous to 48 hr.
- `02-ApplyingProteoForge_Comb.ipynb` — combined 48/72 hr analysis; writes combined summaries and figures.
- `03-SummarizeProteoForge_Comb.ipynb` — aggregates proteoform signals across conditions; produces combined summary tables and views.
- `04-DownstreamComparison.ipynb` — downstream evaluation (e.g., enrichment via g:Profiler, overlaps, resource mapping).
- `notes.md` — biological context and study design references used to interpret findings.

## Data Inputs

Place raw inputs under `data/input/`:

- `hypoxia/` — quantification and metadata tables for H358 hypoxia study (PRIDE [PXD062503]).
    - `report.pg_matrix.tsv`, `report.pr_matrix.tsv`, `report.gg_matrix.tsv`, `report.meta.tsv`, `report.protein_description.tsv`, `report.stats.tsv`, `report.unique_genes_matrix.tsv`
- UniProt reference files — protein sequence/annotation:
    - `uniprot_HomoSapiens_20398_20220815.fasta`, `uniprotkb_Human_AND_model_organism_9606_2025_07_22.txt`
- FASTA reference:
    - `20230316_HumanCr_20806.fasta`
- External resources (downstream comparison):
    - `iPTMnet/` (e.g., `protein.txt`, `ptm.txt`, `score.txt`)
    - `merops/` (e.g., `mer.tab`, `merops.map.tab`)

Inputs are not tracked by git. Use `01-DataPreparation.ipynb` to validate paths and generate cleaned artifacts in `data/cleaned/hypoxia/`.

## Produced Artifacts

- Cleaned artifacts (`data/cleaned/hypoxia/`):
    - `cleaned_data.feather`, `imputed_data.feather`, `centered_data.feather`, `protein_data.feather`, `uniprot_data.feather`, `info_data.feather`, `metadata.csv`
- ProteoForge outputs (`data/results/hypoxia/`):
    - Condition-specific results: `OriginalProtein_48hr_QuEStVar.feather`, `Top3Protein_48hr_QuEStVar.feather`, `AllProtein_48hr_QuEStVar.feather`, `dPFProtein_48hr_QuEStVar.feather`; analogous files for 72 hr
    - Combined and summary tables: `summary_data_48hr.feather`, `summary_data_72hr.feather`, `summary_data_Comb.feather`, `uniprot_data_48hr.feather`, `uniprot_data_72hr.feather`, `uniprot_data_Comb.feather`
    - Enrichment: `gProfiler_Enrichment_Analysis_AllSets.csv`
    - Auxiliary: `test_data_48hr.feather`, `test_data_72hr.feather`, `test_data_Comb.feather`
- Figures (`figures/hypoxia/`):
    - Organized by stage: `01-DataPreparation/`, `02-ProteoForge/`, `03-ResultsAnalysis/`, `04-Downstream/` with `png/` and `pdf/` subfolders

## Experimental Context (from notes)

### Human Tissue (PXD052340)

- Tumor vs matched healthy tissue (70 pairs). Assessed redox proteomics (cysteine oxidation), protein abundance, thiol metabolites (GSH/GSSG and precursors), and MG-derived modifications.
- Key biology: elevated oxidative stress in tumors; increased glutathione synthesis; altered glyoxalase system state.

### Hypoxia Study (PXD062503)

- H358 lung cancer cells cultured at 1% vs 21% oxygen for 48 or 72 hours; quadruplicates; harvested with thiol-preserving protocol.
- Design labels include true hypoxia (1% O2), normoxia controls, and additional controls mimicking hypoxia or oxidative stress:
    - Chemical hypoxia via CoCl2 (stabilizes HIF-1α pathways)
    - Oxidative stress via H2O2 (induces ROS)

### GAPDH Perturbation Studies (PXD061610)

- Knockdown or inhibition (e.g., Koningic Acid) in A549/H358 lines to probe glyoxalase pathway responses.
- Biology: GAPDH activity regulates glycolytic intermediates; links to MG handling and glyoxalase system expression/activity.

## Findings Overview

- Condition-specific proteoform signals across 48 hr and 72 hr, and their combined patterns.
- Contrasts across Original/Top3/All protein sets and ProteoForge-derived proteoform features.
- Enrichment analysis (e.g., g:Profiler) contextualized by proteoform-level evidence.

See `03-SummarizeProteoForge_Comb.ipynb` for combined summaries and `04-DownstreamComparison.ipynb` for enrichment and downstream comparisons.

## Hypoxia Inputs: Notes on Controls

- CoCl2 (cobalt chloride) mimics hypoxia by stabilizing HIF-1α, triggering hypoxia-response pathways under normoxia.
- H2O2 (hydrogen peroxide) induces oxidative stress; useful to disentangle redox-specific changes from hypoxia-driven responses.

## Provenance and Citations

- Manuscript: Tomin et al., 2025 — “Increased antioxidative defense and reduced advanced glycation end-product formation by metabolic adaptation in non-small-cell-lung-cancer patients.”
- PRIDE datasets: [PXD052340](https://www.ebi.ac.uk/pride/archive/projects/PXD052340), [PXD062503](https://www.ebi.ac.uk/pride/archive/projects/PXD062503), [PXD061610](https://www.ebi.ac.uk/pride/archive/projects/PXD061610)
- References for external resources (UniProt, iPTMnet, MEROPS) are documented at the repository level; this folder records file usage and biological interpretation.
