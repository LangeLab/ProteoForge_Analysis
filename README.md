# ProteoForge Manuscript Analysis

| **`Status`** | **`License`** | **`Language`** | **`Release`** | **`Zenodo`** | **`Citation`** |
|---|---:|:---:|:---:|:---:|:---:|
| ![Status](https://img.shields.io/badge/Status-Under_Development-red) | [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc] | ![Language](https://img.shields.io/badge/Language-Python-yellow) | ![Release](https://img.shields.io/badge/Release-v1.0.0-green) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10694635.svg)](https://doi.org/10.5281/zenodo.10694635) | [![Citation](https://img.shields.io/badge/Citation-ProteoForge_Analysis-lightgrey)](#citations) |

This repository contains the code, analyses, and rendered figures supporting the ProteoForge manuscript. It includes real-data benchmarks, simulation studies, and an application to a hypoxia study. This repo is the analysis snapshot.

> The scripts used here (`ProteoForge`) are not packagized, they are simply collection of functions, however more rounded and complete package version in Python can be found at: [LangeLab/ProteoForge](https://github.com/LangeLab/ProteoForge). This was due to the fact that the analysis and manuscript were developed in parallel with the package, and some features especially plotting and printing functions were added ad-hoc for the manuscript. Please refer to the package repository for package-specific documentation, installation instructions, and citation information.

## Repository Layout

Top-level folders and their purpose:

- `ProteoForge/` — core Python scripts used in analyses (parsers, processing, modelling, clustering, classifiers).
- `Benchmark/` — scripts and notebooks for benchmark analyses (R and Python).
- `NSCLC/` — notebooks, data and figures for the hypoxia/NSCLC application.
- `Simulation/` — simulation scripts, notebooks and utilities used to evaluate methods.
- `src/` — auxiliary Python library used by some scripts (utilities, plotting helpers, tests).
- `requirements.txt`, `setup_project.sh`, `setup_project.ps1`, `setup_env.R` — environment and setup helpers.

> The setup utilities ensure you have venv and renv folders created with the required dependencies. They setup the environment for both R and Python analyses to facilitate reproducibility across OSes.

Notes on data and outputs:

- Raw and derived data/figures are not committed. Place raw inputs under the appropriate `*/data/input/` folders; scripts/notebooks will write to `*/data/` and `*/figures/` (see folder READMEs).
- A snapshot of the repository with input data, and the html renders of all notebooks, is available at Zenodo: [10.5281/zenodo.10694635](https://doi.org/10.5281/zenodo.10694635).

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
- Software vs analysis: this repository is the analysis snapshot. The ProteoForge package is developed separately at [LangeLab/ProteoForge](https://github.com/LangeLab/ProteoForge) (see its README and CITATION for package-specific details).

## Citations

Please cite both the analysis snapshot (this repository) and the ProteoForge software package when applicable.

- Analysis snapshot (this repository): use the Zenodo record and select the version matching the git tag you used.
    - DOI: [10.5281/zenodo.10694635](https://doi.org/10.5281/zenodo.10694635)
    - Suggested format (example): “ProteoForge Manuscript Analysis. Version [INSERT VERSION/TAG]. Zenodo. [10.5281/zenodo.10694635](https://doi.org/10.5281/zenodo.10694635).”
- Software package (ProteoForge): cite the package separately.
    - Repository: [LangeLab/ProteoForge](https://github.com/LangeLab/ProteoForge)
    - See the package’s README/CITATION for an up-to-date citation entry and version-specific references.
- Manuscript: cite the manuscript when referencing results or figures derived from this analysis.
    - [INSERT FULL MANUSCRIPT REFERENCE/DOI WHEN AVAILABLE]

## License

This repository is licensed under CC BY-NC 4.0: see [license][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
