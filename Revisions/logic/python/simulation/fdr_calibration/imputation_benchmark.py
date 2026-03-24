#!/usr/bin/env python3
"""
Imputation benchmark under the MNAR-null design with ProteoForge-specific handling.

What this script does
---------------------
1. Generates K null datasets with the same MNAR amputation logic used in
   generate_mnar_null_data.py.
2. Applies alternative imputation strategies (QRILC intentionally excluded):
   - downshift
   - minprob
   - zero_fill
   - global_mean
   - knn
3. Builds test_data using the MNAR split required for fair ProteoForge behavior:
   - adjIntensity path is derived from missing_data (NaN retained)
   - Intensity/log10Intensity path is overwritten with imputed values
4. Runs ProteoForge (default: WLS) and exports null FPR curves for
   identification and grouping.
5. Writes strategy-specific InputData feather files for companion R methods
   (COPF/PeCorA) to consume.

Speed design
------------
- Cache-first execution: all major artifacts are reused unless overwrite is set.
- One MNAR amputation per run; all strategies reuse the same missing_data.
- No redundant bootstrap work in this stage (null FPR only).
- Parallelism delegated to ProteoForge internals with controlled n_jobs.

Usage
-----
From project root:

    PF_SIM_K=20 \
    PF_STRATEGIES=downshift,minprob,zero_fill,global_mean,knn \
    PF_OVERWRITE=0 \
    .venv/bin/python Revisions/logic/python/simulation/fdr_calibration/imputation_benchmark.py
"""

from __future__ import annotations

import math
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
from revisionlib.performance_stats import grouping_calls_proteoforge
from revisionlib.proteoforge_pipeline import run_proteoforge_pipeline, recommend_revision_n_jobs


ALLOWED_STRATEGIES = (
    "downshift",
    "minprob",
    "zero_fill",
    "global_mean",
    "knn",
    "hybrid_downshift_knn",
    "no_imputation",
)


def getenv_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def getenv_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}") from exc


def getenv_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_strategies() -> list[str]:
    raw = os.environ.get(
        "PF_STRATEGIES",
        "downshift,minprob,zero_fill,global_mean,knn,hybrid_downshift_knn,no_imputation",
    )
    strategies = [token.strip() for token in raw.split(",") if token.strip()]
    invalid = [strategy for strategy in strategies if strategy not in ALLOWED_STRATEGIES]
    if invalid:
        raise ValueError(
            "Unsupported strategy in PF_STRATEGIES: "
            f"{invalid}. Allowed: {ALLOWED_STRATEGIES}"
        )
    if not strategies:
        raise ValueError("PF_STRATEGIES produced an empty strategy list.")
    return strategies


def alpha_grid() -> np.ndarray:
    alpha_log = np.exp(np.linspace(np.log(1e-15), np.log(0.01), 25))
    alpha_mid = np.arange(0.01, 0.1001, 0.005)
    alpha_coarse = np.arange(0.10, 0.901, 0.10)
    return np.unique(np.round(np.concatenate([alpha_log, alpha_mid, alpha_coarse]), 15))


def generate_peptide_counts(n_proteins: int, min_pep: int, max_pep: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.beta(0.5, 3.0, size=n_proteins)
    counts = np.round(raw * (max_pep - min_pep) + min_pep).astype(int)
    return np.clip(counts, min_pep, max_pep)


def generate_null_wide(
    run_idx: int,
    base_seed: int,
    n_proteins: int,
    n_peptides: tuple[int, int],
    n_condition: int,
    n_replicates: int,
    protein_mu_mean: float,
    protein_mu_sd: float,
    peptide_offset_sd: float,
    condition_shift_sd: float,
    noise_sd: float,
) -> tuple[pd.DataFrame, dict[str, list[str]], np.ndarray]:
    rng = np.random.default_rng(base_seed + run_idx)

    min_pep, max_pep = n_peptides
    pep_counts = generate_peptide_counts(n_proteins, min_pep, max_pep, rng)

    conditions = ["control"] + [f"cond{i}" for i in range(1, n_condition)]
    sample_names = {
        cond: [f"{cond}-{rep}" for rep in range(1, n_replicates + 1)]
        for cond in conditions
    }
    all_samples: list[str] = []
    for cond in conditions:
        all_samples.extend(sample_names[cond])

    rows_index: list[tuple[str, str]] = []
    rows_data: list[dict[str, float]] = []

    for protein_idx in range(n_proteins):
        protein = f"Protein_{protein_idx}"
        n_pep = int(pep_counts[protein_idx])
        mu_p = rng.normal(protein_mu_mean, protein_mu_sd)

        delta_j = rng.normal(0.0, peptide_offset_sd, size=n_pep)
        beta_c = np.zeros(n_condition)
        beta_c[1:] = rng.normal(0.0, condition_shift_sd, size=n_condition - 1)

        for pep_idx in range(n_pep):
            peptide = f"{protein}_Peptide_{pep_idx}"
            row: dict[str, float] = {}
            for cond_idx, cond in enumerate(conditions):
                noise = rng.normal(0.0, noise_sd, size=n_replicates)
                log2_intensity = mu_p + delta_j[pep_idx] + beta_c[cond_idx] + noise
                intensity = np.power(2.0, log2_intensity)
                for rep_idx, sample in enumerate(sample_names[cond]):
                    row[sample] = float(intensity[rep_idx])
            rows_index.append((protein, peptide))
            rows_data.append(row)

    wide_data = pd.DataFrame(rows_data, columns=all_samples)
    wide_data.index = pd.MultiIndex.from_tuples(rows_index, names=["Protein", "Peptide"])
    unique_proteins = np.array([f"Protein_{idx}" for idx in range(n_proteins)])
    return wide_data, sample_names, unique_proteins


def apply_strategy(
    strategy: str,
    missing_data: pd.DataFrame,
    condition_sample_map: dict[str, list[str]],
    seed: int,
    ds_shift: float,
    ds_low_pct: float,
    ds_min_value: float,
    sparse_n_neighbors: int,
) -> pd.DataFrame:
    # Standalone strategies should each own the full missing-value pattern.
    # Only the explicit hybrid strategy mixes downshift for complete-condition
    # blocks with kNN for sparse gaps.
    if strategy == "downshift":
        return sims.downshifted_imputation(
            data=missing_data,
            condition_sample_map=condition_sample_map,
            is_log2=False,
            shiftMag=ds_shift,
            lowPct=ds_low_pct,
            minValue=ds_min_value,
            impute_all=True,
            seed=seed,
        )

    if strategy == "minprob":
        return minprob_imputation(
            data=missing_data,
            condition_sample_map=condition_sample_map,
            is_log2=False,
            impute_all=True,
            seed=seed,
        )

    if strategy == "zero_fill":
        return zero_fill(
            data=missing_data,
            condition_sample_map=condition_sample_map,
            is_log2=False,
            impute_all=True,
        )

    if strategy == "global_mean":
        return global_mean_fill(
            data=missing_data,
            condition_sample_map=condition_sample_map,
            is_log2=False,
            impute_all=True,
        )

    if strategy == "knn":
        return knn_impute(
            data=missing_data,
            condition_sample_map=condition_sample_map,
            n_neighbors=sparse_n_neighbors,
            is_log2=False,
        )

    if strategy == "hybrid_downshift_knn":
        # Downshift for complete-condition missing blocks; kNN for sparse positions.
        return hybrid_mnar_impute(
            data=missing_data,
            condition_sample_map=condition_sample_map,
            complete_condition_imputer=lambda frame, csm: sims.downshifted_imputation(
                data=frame,
                condition_sample_map=csm,
                is_log2=False,
                shiftMag=ds_shift,
                lowPct=ds_low_pct,
                minValue=ds_min_value,
                impute_all=True,
                seed=seed,
            ),
            sparse_n_neighbors=sparse_n_neighbors,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def build_test_data_with_pf_split(
    missing_data: pd.DataFrame,
    imputed_data: pd.DataFrame,
    condition_sample_map: dict[str, list[str]],
) -> pd.DataFrame:
    # Build test_data using imputed_data so adjIntensity reflects the specific
    # imputation strategy.  isReal / isCompMiss flags are still derived from
    # missing_data so that PF's W_Impute weight correctly down-weights imputed
    # positions — but the normalised intensities used by the statistical model
    # now vary across strategies, making the benchmark meaningful.
    test_data = sims.build_test_data(
        data=imputed_data,
        condition_sample_map=condition_sample_map,
        perturbation_map={},
        proteins_to_perturb=[],
        missing_data=missing_data,   # ← isReal/isCompMiss flags from true missingness
    )

    for col in ("pertCondition", "pertShift"):
        if col in test_data.columns:
            test_data[col] = test_data[col].apply(
                lambda value: str(value.tolist()) if isinstance(value, np.ndarray) else str(value)
            )

    nan_left = int(test_data["Intensity"].isna().sum())
    if nan_left:
        raise RuntimeError(f"Found {nan_left} Intensity NaN values after imputation merge.")

    return test_data


def compute_null_identification_fpr(summary_df: pd.DataFrame, alpha_values: np.ndarray) -> pd.DataFrame:
    total = max(1, int(summary_df.shape[0]))
    pvals = summary_df["adj_pval"].to_numpy(dtype=float)
    rows: list[dict[str, float | int]] = []
    for alpha in alpha_values:
        n_reject = int(np.sum(pvals < alpha))
        rows.append({
            "alpha": float(alpha),
            "fpr": float(n_reject / total),
            "n_total": total,
            "n_reject": n_reject,
        })
    return pd.DataFrame(rows)


def compute_null_grouping_fpr(summary_df: pd.DataFrame, alpha_values: np.ndarray) -> pd.DataFrame:
    required_cols = ["Protein", "PeptideID", "pertProtein", "adj_pval", "ClusterID"]
    missing = [col for col in required_cols if col not in summary_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for grouping FPR: {missing}")

    group_df = summary_df[required_cols].drop_duplicates().copy()
    n_total = max(1, int(group_df["Protein"].nunique()))

    rows: list[dict[str, float | int]] = []
    for alpha in alpha_values:
        calls = grouping_calls_proteoforge(
            data=group_df,
            threshold=float(alpha),
            protein_col="Protein",
            peptide_col="PeptideID",
            truth_col="pertProtein",
            pvalue_col="adj_pval",
            cluster_col="ClusterID",
        )
        n_fp = int(calls["with_proteoform"].sum())
        rows.append({
            "alpha": float(alpha),
            "fpr": float(n_fp / n_total),
            "n_total": n_total,
            "n_fp": n_fp,
        })
    return pd.DataFrame(rows)


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def main() -> None:
    start = time.time()

    K = getenv_int("PF_SIM_K", 10)
    base_seed = getenv_int("PF_BASE_SEED", 42)
    overwrite = getenv_bool("PF_OVERWRITE", False)
    strategies = parse_strategies()
    n_jobs = getenv_int("PF_N_JOBS", recommend_revision_n_jobs())

    # Simulation design mirrors generate_mnar_null_data.py
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

    n_amp_sparse = getenv_int("PF_N_AMP_SPARSE", 100)
    n_amp_compmiss = getenv_int("PF_N_AMP_COMPMISS", 100)
    sparse_miss_rate = getenv_float("PF_SPARSE_MISS_RATE", 0.35)

    # Keep amputation sizes valid for small smoke configurations.
    n_amp_sparse = min(n_amp_sparse, n_proteins)
    n_amp_compmiss = min(n_amp_compmiss, max(0, n_proteins - n_amp_sparse))

    ds_shift = getenv_float("PF_DOWNSHIFT_MAG", 2.0)
    ds_low_pct = getenv_float("PF_DOWNSHIFT_LOW_PCT", 0.15)
    ds_min_value = getenv_float("PF_DOWNSHIFT_MIN", 8.0)
    sparse_n_neighbors = getenv_int("PF_KNN_NEIGHBORS", 5)

    pf_model_type = os.environ.get("PF_MODEL_TYPE", "rlm").strip().lower()
    if pf_model_type not in {"wls", "rlm", "ols", "glm", "quantile"}:
        raise ValueError(f"Unsupported PF_MODEL_TYPE: {pf_model_type}")

    alpha_values = alpha_grid()

    out_root = revision_output_dir("imputation_benchmark")
    input_dir = ensure_directory(out_root / "inputs")
    pf_dir = ensure_directory(out_root / "proteoforge")
    table_dir = ensure_directory(out_root / "tables")

    def _rel(p):
        try: return str(Path(p).relative_to(REPO_ROOT))
        except ValueError: return str(p)

    print("=" * 88)
    print("Imputation benchmark (MNAR-null with PF-specific split)")
    print("=" * 88)
    print(f"K={K} | strategies={strategies} | n_jobs={n_jobs} | overwrite={overwrite}")
    print(f"PF model={pf_model_type} | proteins={n_proteins} | peptides=({min_pep},{max_pep})")
    print(f"Amputation sizes: sparse={n_amp_sparse}, complete-condition={n_amp_compmiss}")

    id_rows: list[pd.DataFrame] = []
    grp_rows: list[pd.DataFrame] = []
    timing_rows: list[dict[str, object]] = []

    for run_idx in range(K):
        run_start = time.time()

        np.random.seed(base_seed + 1000 + run_idx)
        wide_data, csm, unique_proteins = generate_null_wide(
            run_idx=run_idx,
            base_seed=base_seed,
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

        # ── no_imputation baseline: run ProteoForge on complete pre-amputation data ──
        if "no_imputation" in strategies:
            _strat = "no_imputation"
            strategy_start = time.time()
            input_path = input_dir / f"imputation_{_strat}_run_{run_idx:02d}_InputData.feather"
            summary_path = pf_dir / f"proteoforge_{_strat}_run_{run_idx:02d}_summary.feather"

            if (not overwrite) and input_path.exists() and summary_path.exists():
                summary_df = pd.read_feather(summary_path)
                elapsed = time.time() - strategy_start
                timing_rows.append({"run": run_idx, "strategy": _strat, "stage": "cache_hit", "seconds": elapsed})
            else:
                # Wide data has no missing values — use it as both missing_data and imputed_data.
                test_data = build_test_data_with_pf_split(
                    missing_data=wide_data,
                    imputed_data=wide_data,
                    condition_sample_map=csm,
                )
                test_data.to_feather(input_path)
                summary_df, _ = run_proteoforge_pipeline(
                    test_data=test_data,
                    model_type=pf_model_type,
                    n_jobs=n_jobs,
                )
                summary_df.to_feather(summary_path)
                elapsed = time.time() - strategy_start
                timing_rows.append({"run": run_idx, "strategy": _strat, "stage": "recomputed", "seconds": elapsed})

            id_curve = compute_null_identification_fpr(summary_df, alpha_values)
            id_curve["run"] = run_idx
            id_curve["strategy"] = _strat
            id_curve["method"] = "ProteoForge"
            id_rows.append(id_curve)

            grp_curve = compute_null_grouping_fpr(summary_df, alpha_values)
            grp_curve["run"] = run_idx
            grp_curve["strategy"] = _strat
            grp_curve["method"] = "ProteoForge"
            grp_rows.append(grp_curve)

            print(
                f"Run {run_idx + 1:>2}/{K} | {'no_imputation':<22} done in {fmt_time(elapsed)} "
                f"({timing_rows[-1]['stage']})"
            )

        # ── Amputation (only needed for actual imputation strategies) ──────────
        imputation_strategies = [s for s in strategies if s != "no_imputation"]
        if imputation_strategies:
            cond_shifts = {cond: 0.0 for cond in csm if cond != "control"}
            missing_data = sims.amputation(
                data=wide_data,
                unique_proteins=unique_proteins,
                proteins_to_perturb=np.array([]),
                condition_shifts=cond_shifts,
                condition_sample_map=csm,
                n_amputate_1=n_amp_sparse,
                n_amputate_2=n_amp_compmiss,
                n_amputate_3=0,
                missing_rate=sparse_miss_rate,
                seed=base_seed + 1000 + run_idx,
            )

            for strategy in imputation_strategies:
                strategy_start = time.time()
                input_path = input_dir / f"imputation_{strategy}_run_{run_idx:02d}_InputData.feather"
                summary_path = pf_dir / f"proteoforge_{strategy}_run_{run_idx:02d}_summary.feather"

                if (not overwrite) and input_path.exists() and summary_path.exists():
                    summary_df = pd.read_feather(summary_path)
                    elapsed = time.time() - strategy_start
                    timing_rows.append({
                        "run": run_idx,
                        "strategy": strategy,
                        "stage": "cache_hit",
                        "seconds": elapsed,
                    })
                else:
                    imputed = apply_strategy(
                        strategy=strategy,
                        missing_data=missing_data,
                        condition_sample_map=csm,
                        seed=base_seed + 2000 + run_idx,
                        ds_shift=ds_shift,
                        ds_low_pct=ds_low_pct,
                        ds_min_value=ds_min_value,
                        sparse_n_neighbors=sparse_n_neighbors,
                    )

                    test_data = build_test_data_with_pf_split(
                        missing_data=missing_data,
                        imputed_data=imputed,
                        condition_sample_map=csm,
                    )
                    test_data.to_feather(input_path)

                    summary_df, _ = run_proteoforge_pipeline(
                        test_data=test_data,
                        model_type=pf_model_type,
                        n_jobs=n_jobs,
                    )
                    summary_df.to_feather(summary_path)

                    elapsed = time.time() - strategy_start
                    timing_rows.append({
                        "run": run_idx,
                        "strategy": strategy,
                        "stage": "recomputed",
                        "seconds": elapsed,
                    })

                id_curve = compute_null_identification_fpr(summary_df, alpha_values)
                id_curve["run"] = run_idx
                id_curve["strategy"] = strategy
                id_curve["method"] = "ProteoForge"
                id_rows.append(id_curve)

                grp_curve = compute_null_grouping_fpr(summary_df, alpha_values)
                grp_curve["run"] = run_idx
                grp_curve["strategy"] = strategy
                grp_curve["method"] = "ProteoForge"
                grp_rows.append(grp_curve)

                print(
                    f"Run {run_idx + 1:>2}/{K} | {strategy:<22} done in {fmt_time(elapsed)} "
                    f"({timing_rows[-1]['stage']})"
                )

        print(f"Run {run_idx + 1:>2}/{K} completed in {fmt_time(time.time() - run_start)}")

    id_df = pd.concat(id_rows, ignore_index=True)
    grp_df = pd.concat(grp_rows, ignore_index=True)

    id_df.to_csv(table_dir / "proteoforge_identification_fpr_curves.csv", index=False)
    grp_df.to_csv(table_dir / "proteoforge_grouping_fpr_curves.csv", index=False)

    id_summary = (
        id_df.groupby(["method", "strategy", "alpha"], as_index=False)["fpr"]
        .agg(fpr_mean="mean", fpr_sd="std")
        .fillna({"fpr_sd": 0.0})
    )
    grp_summary = (
        grp_df.groupby(["method", "strategy", "alpha"], as_index=False)["fpr"]
        .agg(fpr_mean="mean", fpr_sd="std")
        .fillna({"fpr_sd": 0.0})
    )

    id_summary.to_csv(table_dir / "proteoforge_identification_fpr_summary.csv", index=False)
    grp_summary.to_csv(table_dir / "proteoforge_grouping_fpr_summary.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(table_dir / "runtime_timing.csv", index=False)

    print("\nSaved:")
    print(f"  {_rel(table_dir / 'proteoforge_identification_fpr_curves.csv')}")
    print(f"  {_rel(table_dir / 'proteoforge_grouping_fpr_curves.csv')}")
    print(f"  {_rel(table_dir / 'runtime_timing.csv')}")
    print(f"\nTotal elapsed: {fmt_time(time.time() - start)}")


if __name__ == "__main__":
    main()
