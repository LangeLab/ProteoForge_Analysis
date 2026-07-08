# ProteoForge Manuscript Analysis

![Status](https://img.shields.io/badge/Status-Published-green)
![JPR](https://img.shields.io/badge/Journal-J._Proteome_Res.-blue)
![Language](https://img.shields.io/badge/Language-Python-yellow)
![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)
[![DOI](https://img.shields.io/badge/DOI-10.1021/acs.jproteome.5c01235-blue)](https://doi.org/10.1021/acs.jproteome.5c01235)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281/zenodo.17795845-blue)](https://doi.org/10.5281/zenodo.17795845)

This repository contains the code, analyses, and rendered figures supporting the ProteoForge manuscript published in the *Journal of Proteome Research*. It includes real-data benchmarks, simulation studies, and an application to a hypoxia study. This repo is the analysis snapshot.

> The scripts under `ProteoForge/` are not packaged; they are a collection of functions developed alongside the manuscript. An optimized, fully packaged version of ProteoForge is under active development at [github.com/eneskemalergin/ProteoForge](https://github.com/eneskemalergin/ProteoForge), featuring a proper Python package structure, Polars-based lazy evaluation, Numba-accelerated numerical kernels, a YAML-driven configuration system, a CLI entry point, comprehensive testing, and type-checked code. This packaged version is being prepared for a stable release on PyPI. See the repository for the latest improvements.

## Repository Layout

Top-level folders and their purpose:

- `ProteoForge/` - core Python scripts used in analyses (parsers, processing, modelling, clustering, classifiers).
- `Benchmark/` - scripts and notebooks for benchmark analyses (R and Python).
- `NSCLC/` - notebooks, data and figures for the hypoxia/NSCLC application.
- `Simulation/` - simulation scripts, notebooks and utilities used to evaluate methods.
- `src/` - auxiliary Python library used by some scripts (utilities, plotting helpers, tests).
- `requirements.txt`, `setup_project.sh`, `setup_project.ps1`, `setup_env.R` - environment and setup helpers.

> The setup utilities ensure you have venv and renv folders created with the required dependencies. They setup the environment for both R and Python analyses to facilitate reproducibility across OSes.

Notes on data and outputs:

- Raw and derived data/figures are not committed. Place raw inputs under the appropriate `*/data/input/` folders; scripts/notebooks will write to `*/data/` and `*/figures/` (see folder READMEs).
- A snapshot of the repository with input data, and the html renders of all notebooks, is available at Zenodo: [10.5281/zenodo.17795845](https://doi.org/10.5281/zenodo.17795845).

## Environment Setup (Cross-Platform)

Use the provided setup scripts to configure both Python (venv) and R (renv + pak). R 4.5.0 or newer is required for the R environment.

**Linux / macOS (bash):**

```bash
git clone https://github.com/LangeLab/ProteoForge_Analysis.git
cd ProteoForge_Analysis
bash setup_project.sh
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/LangeLab/ProteoForge_Analysis.git
cd ProteoForge_Analysis
./setup_project.ps1
```

If R is not on PATH, install it from CRAN and rerun the setup command, or run `Rscript setup_env.R` after installation. To activate the Python environment later, use `source .venv/bin/activate` (Linux/macOS) or `./.venv/Scripts/Activate.ps1` (PowerShell).

## Run Steps

Entry points for reproducing analyses and figures:

- Notebooks: `Benchmark/*.ipynb`, `Simulation/*.ipynb`, `NSCLC/*.ipynb`.
- Scripts (Python): `Benchmark/04-runProteoForge.py`, `Simulation/04-runProteoForge.py`.
- Scripts (R): `Benchmark/01-DataProcessing.R`, `Benchmark/02-runCOPF.R`, `Benchmark/03-runPeCorA.R`, plus analogous scripts in `Simulation/`.

Each notebook/script documents its required inputs and outputs. Place raw inputs under the corresponding `*/data/input/` directory before running. Outputs will be written under `*/data/` and `*/figures/`.

## Reproducibility Notes

- R environment: managed with `renv`; run via `setup_project.sh`/`setup_project.ps1` or `Rscript setup_env.R`. Required R version: `>= 4.5.0`.
- Python environment: `requirements.txt` lists dependencies; the setup scripts create `.venv` and install the requirements.
- Data locations: inputs are expected under `*/data/input/`; outputs are written to `*/data/` and `*/figures/`. Large files are not tracked in git.
- Software vs analysis: this repository is the analysis snapshot. A standalone Python package is under development at [github.com/eneskemalergin/ProteoForge](https://github.com/eneskemalergin/ProteoForge) and will be made publicly available on PyPI upon stable release.

## Citations

Please cite the manuscript and the analysis snapshot when using this work.

- Manuscript (published):
    - [ProteoForge: An Imputation-Aware Framework for Differential Proteoform Discovery in Bottom-Up Proteomics](https://pubs.acs.org/doi/10.1021/acs.jproteome.5c01235). _Journal of Proteome Research_. 2026. [10.1021/acs.jproteome.5c01235](https://doi.org/10.1021/acs.jproteome.5c01235).
- Preprint:
    - [ProteoForge: An Imputation-Aware Framework for Differential Proteoform Discovery in Bottom-Up Proteomics](https://www.biorxiv.org/content/10.64898/2025.12.12.694008v1). _bioRxiv_. Posted December 16, 2025.
- Analysis snapshot (this repository): use the Zenodo record and select the version matching the git tag you used.
    - "Snapshot of Benchmarking and Showcasing ProteoForge for Proteoform Deconvolution from Peptide Level Data. Version 1. Zenodo. [10.5281/zenodo.17795845](https://doi.org/10.5281/zenodo.17795845)."

## License

This repository is licensed under CC BY-NC 4.0: see [license][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
