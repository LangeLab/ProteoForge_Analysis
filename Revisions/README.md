# Revisions Re-Execution Guide

This README is for rerunning the revision analyses in the correct order.

The important point is that the notebooks in `Revisions/reports/notebooks/` are report notebooks. They are not the right place to start a clean rerun. Most of the heavy computation happens earlier in `Benchmark/`, `Simulation/`, and `Revisions/logic/`. If those upstream artifacts are missing, the revision notebooks will be incomplete or fail.

## What You Need Before Starting

- Run everything from the repository root.
- Make sure the Python environment exists and includes the packages in `requirements.txt`.
- Make sure R is installed and the project `renv` has been restored.
- Make sure the required input data are already present in the expected `data/input/` locations.
- Use the project root paths exactly as shown below so relative paths resolve correctly.

Recommended setup:

```bash
./setup_project.sh
```

Manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
R --slave -e "source('setup_env.R')"
```

## Execution Order

Run the workflow in this order:

1. Rebuild the base Benchmark pipeline.
2. Rebuild the base Simulation pipeline.
3. Run the revision Python and R drivers.
4. Execute the revision notebooks.

If you skip steps 1 or 2, the revision scripts may not find the prepared benchmark or simulation artifacts they expect.

## Step 1: Run Benchmark First

The revision benchmark summaries depend on the prepared benchmark data and benchmark method outputs generated in `Benchmark/`.

Run:

```bash
Rscript Benchmark/01-DataProcessing.R
Rscript Benchmark/02-runCOPF.R
Rscript Benchmark/03-runPeCorA.R
.venv/bin/python Benchmark/04-runProteoForge.py
```

If you also want to regenerate the original benchmark summary notebooks, run:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Benchmark/05-IdentificationBenchmark.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Benchmark/06-GroupingBenchmark.ipynb
```

Those two notebooks are not the main revision deliverables, but they are useful if you want to confirm that the underlying benchmark results were rebuilt correctly.

## Step 2: Run Simulation Next

The revision simulation summaries, ARI analyses, and calibration workflows depend on the simulation-side prepared data and method outputs.

If the simulation datasets are not already present, generate them first:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Simulation/01-SimulatedDatasets.ipynb
```

Then run the main simulation methods:

```bash
Rscript Simulation/02-runCOPF.R
Rscript Simulation/03-runPeCorA.R
.venv/bin/python Simulation/04-runProteoForge.py
```

If you want to rebuild the original simulation summary notebooks as well, run:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Simulation/05-IdentificationBenchmark.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Simulation/06-GroupingBenchmark.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Simulation/07-ModelBenchmark.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Simulation/08-FigureAssembly.ipynb
```

These are upstream analyses. The revision layer uses their generated data, not the other way around.

## Step 3: Run Revision Scripts

After `Benchmark/` and `Simulation/` have been rebuilt, run the revision scripts. This is the step that creates the cached revision-level tables used by the report notebooks.

### 3.1 Classification and ARI Summaries

Run these first because they feed the benchmark, simulation, and ARI revision notebooks:

```bash
.venv/bin/python Revisions/logic/python/benchmark/classification/benchmark_classification_stats.py
.venv/bin/python Revisions/logic/python/simulation/classification/simulation_classification_stats.py
.venv/bin/python Revisions/logic/python/benchmark/ari/benchmark_ari.py
.venv/bin/python Revisions/logic/python/simulation/ari/benchmark_ari.py
```

### 3.2 Null FDR Calibration

Run the baseline null workflow before the MNAR workflow, because the MNAR notebook compares back to the standard null calibration outputs.

```bash
.venv/bin/python Revisions/logic/python/simulation/fdr_calibration/generate_null_data.py
Rscript Revisions/logic/r/simulation/fdr_calibration/run_r_methods.R
```

### 3.3 MNAR Null Calibration

```bash
.venv/bin/python Revisions/logic/python/simulation/fdr_calibration/generate_mnar_null_data.py
Rscript Revisions/logic/r/simulation/fdr_calibration/run_r_methods_mnar.R
```

### 3.4 Imputation Sensitivity Workflows

Run the imputation benchmark before the imputation power benchmark so the sensitivity analysis outputs are in place first.

```bash
.venv/bin/python Revisions/logic/python/simulation/fdr_calibration/imputation_benchmark.py
Rscript Revisions/logic/r/simulation/fdr_calibration/run_r_methods_imputation_benchmark.R
.venv/bin/python Revisions/logic/python/simulation/fdr_calibration/imputation_power_benchmark.py
Rscript Revisions/logic/r/simulation/fdr_calibration/run_r_methods_imputation_power.R
```

Notes:

- The Python commands intentionally call `.venv/bin/python` directly so you do not have to rely on an already activated shell.
- The revision R drivers are intended to be launched with `Rscript` from the repository root.
- If a revision notebook is missing expected tables, it almost always means one of the commands above was skipped or failed.

## Step 4: Run the Revision Notebooks

Once all upstream and revision scripts have completed, run the notebooks in `Revisions/reports/notebooks/`.

You can run them interactively in Jupyter Lab:

```bash
.venv/bin/jupyter lab Revisions/reports/notebooks
```

Then open each notebook and use Run All from top to bottom.

If you prefer a non-interactive rerun, use:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Revisions/reports/notebooks/01-PreModificationDayComparison.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Revisions/reports/notebooks/02-BenchmarkStats.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Revisions/reports/notebooks/03-SimulationStats.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Revisions/reports/notebooks/04-ARIBenchmark.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Revisions/reports/notebooks/05-FDRCalibration.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Revisions/reports/notebooks/06-MNARCalibration.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Revisions/reports/notebooks/07-ImputationBenchmark.ipynb
```

## Notebook Guide

Each notebook is a reporting layer over outputs generated earlier in the pipeline.

### 01-PreModificationDayComparison.ipynb

Use this notebook to inspect the benchmark data before synthetic perturbations are applied. It is mainly a diagnostic check on day-level structure in the benchmark input.

### 02-BenchmarkStats.ipynb

This notebook summarizes revision benchmark results, mainly the benchmark-side identification and grouping statistics generated from the benchmark pipeline and revision classification scripts.

### 03-SimulationStats.ipynb

This notebook summarizes revision simulation results, including identification and grouping performance across the simulation scenarios.

### 04-ARIBenchmark.ipynb

This notebook reports ARI-based grouping quality for the revision analyses. It relies on the ARI drivers from `Revisions/logic/python/benchmark/ari/` and `Revisions/logic/python/simulation/ari/`.

### 05-FDRCalibration.ipynb

This notebook summarizes the standard null calibration workflow across ProteoForge, COPF, and PeCorA. Run the baseline null Python and R drivers before opening it.

### 06-MNARCalibration.ipynb

This notebook focuses on MNAR-null calibration and its comparison against the standard null baseline. It expects both the standard null and MNAR-null revision scripts to have completed.

### 07-ImputationBenchmark.ipynb

This notebook summarizes the imputation sensitivity analyses. It expects both the imputation benchmark and imputation power workflows to have completed first.

## Minimal Re-Execution Checklist

If you only want the revision summaries in the intended order, use this checklist:

1. Run all Benchmark scripts.
2. Run all Simulation scripts, including simulation dataset generation if needed.
3. Run all revision Python and R drivers.
4. Run all notebooks in `Revisions/reports/notebooks/`.

## Result Orientation

The main results from this folder are the revision summary notebooks and the tables they assemble. In practice, the scripts do the heavy lifting and the notebooks consolidate the final reviewer-facing summaries. For reruns, treat the notebooks as the last step, not the first.
