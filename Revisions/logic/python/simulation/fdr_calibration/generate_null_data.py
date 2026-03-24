#!/usr/bin/env python3
"""
Generate K null datasets for FDR calibration — standalone, no sims.py dependency.

Null design
-----------
Each protein has:
  - a protein-level log2 mean  μ_p  ~ N(20, 2)
  - n_peptides peptides, each with an offset  δ_j  ~ N(0, σ_pep)
  - n_condition conditions, each with a shift  β_c  ~ N(0, σ_cond)
    applied identically to ALL peptides in the protein  (→ no interaction → true null)
  - For every (peptide, condition, replicate) triple the intensity is drawn
    INDEPENDENTLY:
        log2(I) = μ_p + δ_j + β_c + ε          ε ~ N(0, σ_noise)

Because replicates are independent across conditions (not shared-base), the
OLS/WLS/RLM interaction test is properly calibrated and raw p-values are
Uniform(0,1) under this null.

Output
------
<project_root>/Revisions/outputs/simulation/fdr_calibration/null_input/
    null_run_{k}_InputData.feather

Columns: Protein, Peptide, Intensity, Condition, Sample,
         pertProtein, pertPeptide, pertPFG

Usage (from project root):
    PF_SIM_K=50 .venv/bin/python Revisions/logic/python/simulation/fdr_calibration/generate_null_data.py
    .venv/bin/python Revisions/logic/python/simulation/fdr_calibration/generate_null_data.py
"""

import os
import numpy as np
import pandas as pd

# ── Resolve project root (4 levels up from this script) ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", ".."))
os.chdir(PROJECT_ROOT)


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

# ── Simulation parameters ────────────────────────────────────────────────────
K = getenv_int("PF_SIM_K", 50)
base_seed = 42
n_proteins = 500
n_peptides = (5, 50)  # min, max peptides per protein
n_condition = 3  # 1 control  + 2 treatment conditions
n_replicates = 10  # replicates per condition

# Distribution parameters (log2 scale)
protein_mu_mean = 20.0  # mean of protein-level log2 means
protein_mu_sd = 2.0  # sd of protein-level log2 means
peptide_offset_sd = 0.5  # sd of peptide offsets within a protein
condition_shift_sd = 0.10  # sd of protein-level condition shifts
noise_sd = 0.30  # replicate-level noise (log2 scale)

output_dir = os.path.join(
    PROJECT_ROOT, "Revisions", "outputs", "simulation", "fdr_calibration", "null_input"
)
os.makedirs(output_dir, exist_ok=True)

print(
    f"Generating K={K} null datasets  "
    f"(n_proteins={n_proteins}, n_peptides={n_peptides}, "
    f"n_condition={n_condition}, n_replicates={n_replicates})"
)
print(f"  PF_SIM_K={K}")
print(
    f"  noise_sd={noise_sd}, condition_shift_sd={condition_shift_sd}, "
    f"peptide_offset_sd={peptide_offset_sd}\n"
)


def generate_peptide_counts(n_proteins, min_pep, max_pep, rng):
    """Draw peptide counts from a beta-like distribution skewed toward low counts."""
    raw = rng.beta(0.5, 3.0, size=n_proteins)
    counts = np.round(raw * (max_pep - min_pep) + min_pep).astype(int)
    return np.clip(counts, min_pep, max_pep)


def generate_null_run(
    run_idx, base_seed, n_proteins, n_peptides, n_condition, n_replicates
):
    """Generate one null dataset with independent replicates per condition."""
    seed = base_seed + run_idx
    rng = np.random.default_rng(seed)

    min_pep, max_pep = n_peptides
    pep_counts = generate_peptide_counts(n_proteins, min_pep, max_pep, rng)

    # Condition and sample names
    conditions = ["control"] + [f"cond{i}" for i in range(1, n_condition)]
    sample_names = {
        c: [f"{c}-{r}" for r in range(1, n_replicates + 1)] for c in conditions
    }

    rows = []
    for pi in range(n_proteins):
        prot_name = f"Protein_{pi}"
        n_pep = pep_counts[pi]
        mu_p = rng.normal(protein_mu_mean, protein_mu_sd)

        # Peptide offsets (shared across conditions — this is the protein structure)
        delta_j = rng.normal(0, peptide_offset_sd, size=n_pep)

        # Condition shifts: ONE per protein, applied to ALL its peptides → true null
        beta_c = np.zeros(n_condition)
        beta_c[1:] = rng.normal(0, condition_shift_sd, size=n_condition - 1)

        for ji in range(n_pep):
            pep_name = f"{prot_name}_Peptide_{ji}"
            for ci, cond in enumerate(conditions):
                # Independent noise for each replicate — NOT shared-base
                noise = rng.normal(0, noise_sd, size=n_replicates)
                log2_intensities = mu_p + delta_j[ji] + beta_c[ci] + noise
                intensities = np.power(2, log2_intensities)

                for ri, samp in enumerate(sample_names[cond]):
                    rows.append(
                        {
                            "Protein": prot_name,
                            "Peptide": pep_name,
                            "Intensity": intensities[ri],
                            "Condition": cond,
                            "Sample": samp,
                            "pertProtein": False,
                            "pertPeptide": False,
                            "pertPFG": -1,
                        }
                    )

    df = pd.DataFrame(rows)
    df["pertPFG"] = df["pertPFG"].astype("int32")
    df["pertProtein"] = df["pertProtein"].astype(bool)
    df["pertPeptide"] = df["pertPeptide"].astype(bool)
    return df


# ── Main loop ─────────────────────────────────────────────────────────────────
for k in range(K):
    out_path = os.path.join(output_dir, f"null_run_{k}_InputData.feather")
    if os.path.exists(out_path):
        print(f"  Run {k:>2}: already exists → skipping")
        continue

    df = generate_null_run(
        k, base_seed, n_proteins, n_peptides, n_condition, n_replicates
    )
    df.to_feather(out_path)

    n_unique_peps = df[["Protein", "Peptide"]].drop_duplicates().shape[0]
    print(
        f"  Run {k:>2}: {n_unique_peps:,} unique peptides "
        f"× {df['Sample'].nunique()} samples → {os.path.basename(out_path)}"
    )

print("\nDone. Null input datasets are ready for run_r_methods.R")
