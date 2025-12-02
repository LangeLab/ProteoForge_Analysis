# ProteoForge Manuscript Analysis

| **`Status`** | **`License`** | **`Language`** | **`Release`** | **`Zenodo`** | **`Citation`** |
|---|---:|:---:|:---:|:---:|:---:|
| ![Status](https://img.shields.io/badge/Status-Under_Development-red) | [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc] | ![Language](https://img.shields.io/badge/Language-Python-yellow) | ![Release](https://img.shields.io/badge/Release-v1.0.0-green) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10694635.svg)](https://doi.org/10.5281/zenodo.10694635) | [![Citation](https://img.shields.io/badge/Citation-ProteoForge_Analysis-lightgrey)](https://github.com/LangeLab/ProteoForge_Analysis/tree/Manuscript) |

This repository contains the code, analyses, and rendered figures supporting the ProteoForge manuscript. It includes real-data benchmarks, simulation studies, and an application to a hypoxia study. The analyses use the ProteoForge Python package (included in the `ProteoForge/` folder) together with a collection of R and Python scripts and notebooks to run benchmarks and reproduce figures.

## Repository layout

Top-level folders and their purpose:

- `ProteoForge/` — core Python package used for the analyses (parsers, clustering, normalization, plotting, classifiers).
- `Benchmark/` — scripts and notebooks for benchmark analyses and example workflows (R and Python).
- `NSCLC/` — notebooks, data and figures for the hypoxia / NSCLC application.
- `Simulation/` — simulation scripts, notebooks and utilities used to evaluate methods.
- `renders/` — rendered HTML versions of notebooks and assembled figures used for manuscript production.
- `src/` — auxiliary Python library used by some scripts (utilities, plotting helpers, tests).
- `renv/` — R environment metadata used to reproduce R analyses.
- `requirements.txt`, `setup_project.sh`, `setup_env.R` — environment and setup helpers.

Data and figures directories (large files) are excluded from git. See README sections below for how to obtain or regenerate inputs.

## Quick start

1. Clone the repository:

```bash
git clone https://github.com/USERNAME/Analysis_with_ProteoForge.git
cd Analysis_with_ProteoForge
```

1. Setup environments

- Python (recommended): create a venv and install Python requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- R (for R notebooks / scripts): follow the instructions in `renv/` or run the helper to recreate the R environment:

```r
# in an R session
source("setup_env.R")
# or use renv::restore()
```

## Running the benchmarks and notebooks

- Use the notebooks in `Benchmark/`, `Simulation/` or `NSCLC/` for step-by-step analyses. Rendered HTML files are available under `renders/` for convenience.
- For scripted runs, `Benchmark/` and `Simulation/` contain R and Python scripts with documented arguments. Many scripts write outputs to `Benchmark/data/results/` or `Simulation/data/`.

## Reproducibility notes

- R environments: the `renv/` snapshot lists R package versions used for the R analyses. Use `renv::restore()` inside an R session to recreate the environment.
- Python environment: `requirements.txt` contains the core Python dependencies. For exact reproducibility, pin versions or use the provided virtual environment creation steps.
- Data: raw datasets are not committed. Place raw inputs under the appropriate `*/data/input/` folders. Some scripts include download links or instructions to fetch public datasets.

## Citation and license

This repository is licensed under CC BY-NC 4.0: see [license][cc-by-nc]. If you use these analyses or the ProteoForge code in a publication, please cite the manuscript and repository as indicated in the manuscript.

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
