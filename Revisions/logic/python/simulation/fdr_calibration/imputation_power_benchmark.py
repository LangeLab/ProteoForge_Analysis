#!/usr/bin/env python3
"""
imputation_power_benchmark.py

Non-null power simulation companion for imputation_benchmark.py.

For each imputation strategy and null run the benchmark produces TPR (sensitivity)
alongside the already-computed FPR, enabling FDR-vs-Power (ROC-style) curves.

Design
------
Uses the same null-data dimensions as imputation_benchmark.py but injects
K_PERT perturbed proteins per run (peptide-specific shifts in a subset of
conditions).  PF is then run on the imputed data (with corrected adjIntensity)
and the TPR at each alpha threshold is computed from the perturbed peptides.

Only ProteoForge is evaluated here — R methods (COPF/PeCorA) power evaluation
would require running the full R driver which is handled separately.

Usage
-----
From project root:

    PF_SIM_K=5 \\
    PF_STRATEGIES=downshift,minprob,zero_fill,global_mean,knn,hybrid_downshift_knn,no_imputation \\
    PF_OVERWRITE=0 \\
    .venv/bin/python Revisions/logic/python/simulation/fdr_calibration/imputation_power_benchmark.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
REVISION_PYTHON_ROOT = CURRENT_FILE.parents[2]
REPO_ROOT = CURRENT_FILE.parents[5]

for path in (REVISION_PYTHON_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.chdir(REPO_ROOT)

from Simulation import sims
from revisionlib.imputation import (
    global_mean_fill,
    hybrid_mnar_impute,
    knn_impute,
    minprob_imputation,
    zero_fill,
)
from revisionlib.paths import ensure_directory, revision_output_dir
from revisionlib.proteoforge_pipeline import run_proteoforge_pipeline, recommend_revision_n_jobs

# Re-use helpers from null benchmark
from imputation_benchmark import (
    ALLOWED_STRATEGIES,
    alpha_grid,
    apply_strategy,
    fmt_time,
    generate_null_wide,
    getenv_bool,
    getenv_float,
    getenv_int,
    parse_strategies,
)


def compute_tpr(summary_df: pd.DataFrame, alpha_values: np.ndarray) -> pd.DataFrame:
    """Compute per-alpha TPR from a summary with pertPeptide / adj_pval columns."""
    pert_mask = summary_df["pertPeptide"].astype(bool)
    n_total = int(pert_mask.sum())
    if n_total == 0:
        rows = [{"alpha": float(a), "tpr": np.nan, "n_total": 0, "n_detected": 0}
                for a in alpha_values]
        return pd.DataFrame(rows)
    pvals_pert = summary_df.loc[pert_mask, "adj_pval"].to_numpy(dtype=float)
    rows = []
    for alpha in alpha_values:
        n_det = int(np.sum(pvals_pert < alpha))
        rows.append({
            "alpha": float(alpha),
            "tpr": float(n_det / n_total),
            "n_total": n_total,
            "n_detected": n_det,
        })
    return pd.DataFrame(rows)


def build_power_test_data(
    missing_data: pd.DataFrame,
    imputed_data: pd.DataFrame,
    condition_sample_map: dict,
    perturbation_map: dict,
    proteins_to_perturb: np.ndarray,
) -> pd.DataFrame:
    """Same adjIntensity-from-imputed logic as imputation_benchmark, but includes perturbation truth."""
    test_data = sims.build_test_data(
        data=imputed_data,
        condition_sample_map=condition_sample_map,
        perturbation_map=perturbation_map,
        proteins_to_perturb=proteins_to_perturb,
        missing_data=missing_data,
    )
    for col in ("pertCondition", "pertShift"):
        if col in test_data.columns:
            test_data[col] = test_data[col].apply(
                lambda v: str(v.tolist()) if isinstance(v, np.ndarray) else str(v)
            )
    return test_data


def main() -> None:
    start = time.time()

    K = getenv_int("PF_SIM_K", 5)
    base_seed = getenv_int("PF_BASE_SEED", 42)
    overwrite = getenv_bool("PF_OVERWRITE", False)
    strategies = parse_strategies()
    n_jobs = getenv_int("PF_N_JOBS", recommend_revision_n_jobs())

    # Simulation design (mirrors null benchmark)
    n_proteins = getenv_int("PF_N_PROTEINS", 500)
    n_condition = getenv_int("PF_N_CONDITIONS", 3)
    n_replicates = getenv_int("PF_N_REPLICATES", 10)
    min_pep = getenv_int("PF_MIN_PEPTIDES", 5)
    max_pep = getenv_int("PF_MAX_PEPTIDES", 50)
    protein_mu_mean = getenv_float("PF_PROTEIN_MU_MEAN", 20.0)
    protein_mu_sd = getenv_float("PF_PROTEIN_MU_SD", 2.0)
    peptide_offset_sd = getenv_float("PF_PEPTIDE_OFFSET_SD", 0.5)
    condition_shift_sd = getenv_float("PF_CONDITION_SHIFT_SD", 0.10)
    noise_sd = getenv_float("PF_NOISE_SD", 0.30)

    n_amp_sparse = min(getenv_int("PF_N_AMP_SPARSE", 100), n_proteins)
    n_amp_compmiss = min(getenv_int("PF_N_AMP_COMPMISS", 100), max(0, n_proteins - n_amp_sparse))
    sparse_miss_rate = getenv_float("PF_SPARSE_MISS_RATE", 0.35)

    ds_shift = getenv_float("PF_DOWNSHIFT_MAG", 2.0)
    ds_low_pct = getenv_float("PF_DOWNSHIFT_LOW_PCT", 0.15)
    ds_min_value = getenv_float("PF_DOWNSHIFT_MIN", 8.0)
    sparse_n_neighbors = getenv_int("PF_KNN_NEIGHBORS", 5)
    pf_model_type = os.environ.get("PF_MODEL_TYPE", "rlm").strip().lower()

    # Power-specific: how many proteins to perturb, shift magnitude
    n_pert = getenv_int("PF_POWER_N_PERTURB", 50)       # proteins with proteoform-like shifts
    pert_shift = getenv_float("PF_POWER_SHIFT", 2.0)     # log2 shift magnitude

    alpha_values = alpha_grid()
    out_root = revision_output_dir("imputation_benchmark")
    power_dir = ensure_directory(out_root / "power")
    power_input_dir = ensure_directory(out_root / "power_inputs")
    table_dir = ensure_directory(out_root / "tables")

    def _rel(p):
        try: return str(Path(p).relative_to(REPO_ROOT))
        except ValueError: return str(p)

    print("=" * 88)
    print("Imputation power benchmark (MNAR-non-null with PF)")
    print(f"K={K} | strategies={strategies} | n_pert={n_pert} | pert_shift={pert_shift}")
    print("=" * 88)

    tpr_rows: list[pd.DataFrame] = []

    for run_idx in range(K):
        rng = np.random.default_rng(base_seed + 7000 + run_idx)

        wide_data, csm, unique_proteins = generate_null_wide(
            run_idx=run_idx,
            base_seed=base_seed + 5000,  # different seeds from null benchmark
            n_proteins=n_proteins,
            n_peptides=(min_pep, max_pep),
            n_condition=n_condition,
            n_replicates=n_replicates,
            protein_mu_mean=protein_mu_mean,
            protein_mu_sd=protein_mu_sd,
            peptide_offset_sd=peptide_offset_sd,
            condition_shift_sd=condition_shift_sd,
            noise_sd=noise_sd,
        )

        # ── Build a synthetic perturbation map ────────────────────────────────
        perturb_proteins = rng.choice(unique_proteins, size=min(n_pert, len(unique_proteins)),
                                       replace=False)
        # For each perturbed protein, shift a *random* number of its peptides in cond1.
        # The number of shifted peptides is drawn uniformly from [1, n_pep) per protein,
        # making the benchmark harder for single-peptide tests (PeCorA) and more
        # realistic: real proteoform events affect a variable subset of peptides.
        pep_index = wide_data.index.get_level_values("Peptide").unique()
        perturbation_map: dict[str, dict] = {}
        for prot in perturb_proteins:
            prot_peps = [p for p in pep_index if p.startswith(f"{prot}_")]
            if not prot_peps:
                continue
            n_pep = len(prot_peps)
            n_shift = int(rng.integers(1, max(2, n_pep)))  # [1, n_pep)
            shifted_peps = rng.choice(prot_peps, size=n_shift, replace=False)
            for pep in shifted_peps:
                perturbation_map[(prot, pep)] = {
                    "Protein": prot,
                    "Peptide": pep,
                    "pertCondition": ["cond1"],
                    "pertShift": [float(pert_shift)],
                }

        # Apply perturbation to wide_data (in place of sims.apply_perturbations)
        pert_wide = wide_data.copy()
        for (prot, pep), info in perturbation_map.items():
            for cond, shift in zip(info["pertCondition"], info["pertShift"]):
                for sample in csm.get(cond, []):
                    if sample in pert_wide.columns:
                        pert_wide.loc[(prot, pep), sample] *= (2.0 ** shift)

        # Ampute the perturbed data (MNAR)
        cond_shifts = {cond: 0.0 for cond in csm if cond != "control"}
        missing_data = sims.amputation(
            data=pert_wide,
            unique_proteins=unique_proteins,
            proteins_to_perturb=perturb_proteins,
            condition_shifts=cond_shifts,
            condition_sample_map=csm,
            n_amputate_1=n_amp_sparse,
            n_amputate_2=n_amp_compmiss,
            n_amputate_3=0,
            missing_rate=sparse_miss_rate,
            seed=base_seed + 6000 + run_idx,
        )

        # ── also handle no_imputation ─────────────────────────────────────────
        imputation_strategies = [s for s in strategies if s != "no_imputation"]

        if "no_imputation" in strategies:
            strat = "no_imputation"
            cache = power_dir / f"power_{strat}_run_{run_idx:02d}_summary.feather"
            input_path = power_input_dir / f"power_{strat}_run_{run_idx:02d}_InputData.feather"
            if (not overwrite) and cache.exists():
                summary_df = pd.read_feather(cache)
            else:
                test_data = build_power_test_data(
                    missing_data=pert_wide,
                    imputed_data=pert_wide,
                    condition_sample_map=csm,
                    perturbation_map=perturbation_map,
                    proteins_to_perturb=perturb_proteins,
                )
                if not input_path.exists() or overwrite:
                    test_data.to_feather(input_path)
                summary_df, _ = run_proteoforge_pipeline(test_data, pf_model_type, n_jobs=n_jobs)
                summary_df.to_feather(cache)
            curve = compute_tpr(summary_df, alpha_values)
            curve["run"] = run_idx
            curve["strategy"] = strat
            curve["method"] = "ProteoForge"
            tpr_rows.append(curve)
            print(f"Run {run_idx+1}/{K} | {'no_imputation':<24} done")

        for strategy in imputation_strategies:
            cache = power_dir / f"power_{strategy}_run_{run_idx:02d}_summary.feather"
            input_path = power_input_dir / f"power_{strategy}_run_{run_idx:02d}_InputData.feather"
            if (not overwrite) and cache.exists():
                summary_df = pd.read_feather(cache)
            else:
                imputed = apply_strategy(
                    strategy=strategy,
                    missing_data=missing_data,
                    condition_sample_map=csm,
                    seed=base_seed + 8000 + run_idx,
                    ds_shift=ds_shift,
                    ds_low_pct=ds_low_pct,
                    ds_min_value=ds_min_value,
                    sparse_n_neighbors=sparse_n_neighbors,
                )
                test_data = build_power_test_data(
                    missing_data=missing_data,
                    imputed_data=imputed,
                    condition_sample_map=csm,
                    perturbation_map=perturbation_map,
                    proteins_to_perturb=perturb_proteins,
                )
                if not input_path.exists() or overwrite:
                    test_data.to_feather(input_path)
                summary_df, _ = run_proteoforge_pipeline(test_data, pf_model_type, n_jobs=n_jobs)
                summary_df.to_feather(cache)
            curve = compute_tpr(summary_df, alpha_values)
            curve["run"] = run_idx
            curve["strategy"] = strategy
            curve["method"] = "ProteoForge"
            tpr_rows.append(curve)
            print(f"Run {run_idx+1}/{K} | {strategy:<24} done")

        print(f"Run {run_idx+1}/{K} completed in {fmt_time(time.time() - start)}")

    tpr_df = pd.concat(tpr_rows, ignore_index=True)
    tpr_df.to_csv(table_dir / "proteoforge_identification_tpr_curves.csv", index=False)

    tpr_summary = (
        tpr_df.groupby(["method", "strategy", "alpha"], as_index=False)["tpr"]
        .agg(tpr_mean="mean", tpr_sd="std")
        .fillna({"tpr_sd": 0.0})
    )
    tpr_summary.to_csv(table_dir / "proteoforge_identification_tpr_summary.csv", index=False)

    print(f"\nSaved TPR tables to {_rel(table_dir)}")
    print(f"Total elapsed: {fmt_time(time.time() - start)}")


if __name__ == "__main__":
    main()
