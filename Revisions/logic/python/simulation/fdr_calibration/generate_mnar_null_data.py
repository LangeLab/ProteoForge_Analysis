#!/usr/bin/env python3
"""
Generate K null datasets WITH MNAR-pattern missingness for FDR calibration.

Data generation
---------------
Uses the SAME null design as generate_null_data.py (properly calibrated null):
  - Independent replicates per condition (not shared-base)
  - Small condition shifts: β_c ~ N(0, 0.10) per protein, applied to ALL
    peptides → no interaction → true null
  - log2(I) = μ_p + δ_j + β_c + ε,  ε ~ N(0, 0.30)

MNAR overlay (matching Sim2 amputation pattern)
------------------------------------------------
  1. Wide-format data is amputated using sims.amputation():
     - 100 proteins: sparse random missingness (35% per-value rate)
     - 100 proteins: complete-condition blocks (one peptide per protein)
     - 300 proteins: completely clean
  2. sims.downshifted_imputation(impute_all=True): fills ALL missing values
     (sparse + complete-condition blocks) with realistic downshifted noise.
     COPF/PeCorA receive these values as real observations and are corrupted.
     ProteoForge drops sparse-imputed rows via is_sparse filter (isReal=0,
     isCompMiss=0), achieving correct FDR control.

Output
------
<project_root>/Revisions/outputs/simulation/mnar_fdr_calibration/null_input/
    mnar_null_run_{k}_InputData.feather

Usage (from project root):
    PF_SIM_K=50 .venv/bin/python Revisions/logic/python/simulation/fdr_calibration/generate_mnar_null_data.py
    .venv/bin/python Revisions/logic/python/simulation/fdr_calibration/generate_mnar_null_data.py
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Resolve project root (5 levels up from this script) ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from Simulation import sims


def getenv_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"Environment variable {name} must be >= 1, got {value}")
    return value

# ── Simulation parameters (matching non-MNAR generate_null_data.py) ─────────
K = getenv_int("PF_SIM_K", 50)
base_seed = 42
n_proteins = 500
n_peptides = (5, 50)  # min, max peptides per protein
n_condition = 3  # 1 control + 2 treatment conditions
n_replicates = 10  # replicates per condition

# Distribution parameters (log2 scale) — match generate_null_data.py
protein_mu_mean = 20.0
protein_mu_sd = 2.0
peptide_offset_sd = 0.5
condition_shift_sd = 0.10  # small shifts → no spurious interactions after build_test_data
noise_sd = 0.30

# ── MNAR parameters (match Sim2 amputation pattern) ─────────────────────────
n_amp_sparse   = 100   # proteins with sparse random missingness
n_amp_compmiss = 100   # proteins with complete-condition blocks
sparse_miss_rate = 0.35  # per-value missing probability for sparse proteins

# ── Downshifted imputation parameters (match main simulations) ──────────────
ds_shiftMag = 2.0
ds_lowPct   = 0.15
ds_minValue = 8

output_dir = os.path.join(
    PROJECT_ROOT, "Revisions", "outputs", "simulation", "mnar_fdr_calibration", "null_input"
)
os.makedirs(output_dir, exist_ok=True)

print(
    f"Generating K={K} MNAR null datasets  "
    f"(n_proteins={n_proteins}, n_peptides={n_peptides}, "
    f"n_condition={n_condition}, n_replicates={n_replicates})"
)
print(f"  PF_SIM_K={K}")
print(
    f"  Data: independent replicates, condition_shift_sd={condition_shift_sd}, "
    f"noise_sd={noise_sd}"
)
print(
    f"  MNAR: n_amp_sparse={n_amp_sparse}, n_amp_compmiss={n_amp_compmiss}, "
    f"sparse_miss_rate={sparse_miss_rate}"
)
print(
    f"  Imputation: downshifted (shift={ds_shiftMag}, lowPct={ds_lowPct}) "
    f"for ALL missing (impute_all=True); ProteoForge drops sparse via is_sparse filter\n"
)


def generate_peptide_counts(n_proteins, min_pep, max_pep, rng):
    """Draw peptide counts from a beta-like distribution skewed toward low counts."""
    raw = rng.beta(0.5, 3.0, size=n_proteins)
    counts = np.round(raw * (max_pep - min_pep) + min_pep).astype(int)
    return np.clip(counts, min_pep, max_pep)


def generate_null_wide(run_idx, base_seed):
    """
    Generate one null dataset with independent replicates per condition,
    matching the non-MNAR generate_null_data.py design.
    Returns wide-format DataFrame (Protein, Peptide) x Sample in raw intensity scale,
    condition_sample_map, and unique_proteins.
    """
    seed = base_seed + run_idx
    rng = np.random.default_rng(seed)

    min_pep, max_pep = n_peptides
    pep_counts = generate_peptide_counts(n_proteins, min_pep, max_pep, rng)

    # Condition and sample names
    conditions = ["control"] + [f"cond{i}" for i in range(1, n_condition)]
    sample_names = {
        c: [f"{c}-{r}" for r in range(1, n_replicates + 1)] for c in conditions
    }
    all_samples = []
    for c in conditions:
        all_samples.extend(sample_names[c])

    condition_sample_map = sample_names

    rows_index = []  # (Protein, Peptide) tuples
    rows_data = []   # list of dicts: sample → intensity

    for pi in range(n_proteins):
        prot_name = f"Protein_{pi}"
        n_pep = pep_counts[pi]
        mu_p = rng.normal(protein_mu_mean, protein_mu_sd)

        # Peptide offsets (same across conditions → no interaction)
        delta_j = rng.normal(0, peptide_offset_sd, size=n_pep)

        # Condition shifts: ONE per protein, applied to ALL peptides → true null
        beta_c = np.zeros(n_condition)
        beta_c[1:] = rng.normal(0, condition_shift_sd, size=n_condition - 1)

        for ji in range(n_pep):
            pep_name = f"{prot_name}_Peptide_{ji}"
            row = {}
            for ci, cond in enumerate(conditions):
                # Independent noise for each replicate
                noise = rng.normal(0, noise_sd, size=n_replicates)
                log2_intensities = mu_p + delta_j[ji] + beta_c[ci] + noise
                intensities = np.power(2, log2_intensities)
                for ri, samp in enumerate(sample_names[cond]):
                    row[samp] = intensities[ri]
            rows_index.append((prot_name, pep_name))
            rows_data.append(row)

    wide_data = pd.DataFrame(rows_data, columns=all_samples)
    wide_data.index = pd.MultiIndex.from_tuples(rows_index, names=['Protein', 'Peptide'])
    unique_proteins = np.array([f"Protein_{i}" for i in range(n_proteins)])

    return wide_data, condition_sample_map, unique_proteins


# ── Main loop ─────────────────────────────────────────────────────────────────
for k in range(K):
    out_path = os.path.join(output_dir, f"mnar_null_run_{k}_InputData.feather")
    if os.path.exists(out_path):
        try:
            cached_intensity = pd.read_feather(out_path, columns=["Intensity"])
            n_cached_na = int(cached_intensity["Intensity"].isna().sum())
        except Exception as exc:
            n_cached_na = -1
            print(f"  Run {k:>2}: cached file unreadable ({exc}) → regenerating")
        else:
            if n_cached_na == 0:
                print(f"  Run {k:>2}: already exists → skipping")
                continue
            print(
                f"  Run {k:>2}: cached file has {n_cached_na} residual Intensity NaNs "
                f"→ regenerating"
            )

    np.random.seed(base_seed + 1000 + k)

    # Step 1: Generate complete null data (independent replicates, tiny shifts)
    wide_data, csm, unique_proteins = generate_null_wide(k, base_seed)
    n_peptides_total = len(wide_data)

    # Build condition_shifts dict for sims.amputation
    cond_shifts = {}
    for c in csm:
        if c != 'control':
            cond_shifts[c] = 0.0  # amputation just needs the dict keys

    # Step 2: Apply MNAR amputation using sims.amputation()
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
        seed=base_seed + 1000 + k,
    )

    n_missing = missing_data.isna().sum().sum()
    n_total = missing_data.size
    pct_miss = 100 * n_missing / n_total

    # Step 3: Downshifted imputation for ALL missing (sparse + complete-condition blocks).
    # impute_all=True fills every NaN with realistic downshifted noise.
    # COPF and PeCorA consume the Intensity column (imputed values look real → corrupted).
    # ProteoForge computes adjIntensity from missing_data (NaN retained) so cntrPepMean
    # is based only on real observations → unbiased normalization → then drops sparse
    # rows before WLS via is_sparse filter.  This is the critical split:
    #   adjIntensity ← missing_data (NaN → cntrPepMean mean ≈ 0, harmless)
    #   Intensity    ← imputed_data (downshifted noise → corrupts COPF / PeCorA)
    imputed_data = sims.downshifted_imputation(
        data=missing_data,
        condition_sample_map=csm,
        is_log2=False,
        shiftMag=ds_shiftMag,
        lowPct=ds_lowPct,
        minValue=ds_minValue,
        impute_all=True,
        seed=base_seed + 2000 + k,
    )

    n_sparse_remaining = imputed_data.isna().sum().sum()

    # Step 4a: Build adjIntensity from missing_data (NaN retained).
    # This keeps ProteoForge's normalization uncontaminated by imputed control values.
    test_data = sims.build_test_data(
        data=missing_data,          # ← NaN-containing original; cntrPepMean = real-obs only
        condition_sample_map=csm,
        perturbation_map={},
        proteins_to_perturb=[],
        missing_data=missing_data,  # ← sets isReal / isCompMiss correctly
    )

    # Step 4b: Override Intensity / log10Intensity with the downshifted imputed values so
    # COPF and PeCorA receive realistic-looking low intensities and are corrupted.
    imputed_long = imputed_data.reset_index().melt(
        id_vars=["Protein", "Peptide"], var_name="Sample", value_name="Intensity_imp"
    )
    test_data = test_data.merge(
        imputed_long, on=["Protein", "Peptide", "Sample"], how="left"
    )
    test_data["Intensity"]      = test_data["Intensity_imp"]
    test_data["log10Intensity"] = np.log10(test_data["Intensity"])
    test_data.drop(columns=["Intensity_imp"], inplace=True)

    # Fix array-like columns for feather serialization
    for col in ['pertCondition', 'pertShift']:
        if col in test_data.columns:
            test_data[col] = test_data[col].apply(
                lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x))

    n_remaining_na = int(test_data["Intensity"].isna().sum())
    if n_remaining_na:
        raise RuntimeError(
            f"Run {k} still contains {n_remaining_na} Intensity NaNs after imputation."
        )

    # Save
    test_data.to_feather(out_path)

    print(
        f"  Run {k:>2}: {n_peptides_total:,} peptides | "
        f"{n_missing:,}/{n_total:,} values missing ({pct_miss:.1f}%) | "
        f"NaN after impute_all: {n_sparse_remaining:,} | "
        f"→ {os.path.basename(out_path)}"
    )

print("\nDone. MNAR null input datasets are ready.")
print("Next: run run_r_methods_mnar.R, then open MNARCalibration.ipynb")
