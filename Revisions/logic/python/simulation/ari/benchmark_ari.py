#!/usr/bin/env python3
"""
benchmark_ari.py
================
Compute Adjusted Rand Index (ARI) for peptide grouping quality across
Sim1–Sim4 for ProteoForge and COPF (Phase 4a from the revision plan).

PeCorA is excluded — it produces no grouping output.

Three reporting modes (per the plan § 4a):
  1. Per-protein mean ARI (perturbed only) + bootstrap 95% CI
  2. Per-protein mean ARI (all proteins, unperturbed scored 1.0 if 1 cluster)
  3. Concatenated ARI (all peptides pooled) + CI

Prerequisites
-------------
- Existing result feather files from 04-runProteoForge.py and R methods
  (02-runCOPF.R, 03-runPeCorA.R) in ./data/Sim{1-4}/.

Outputs
-------
Figures  → ./Revisions/outputs/simulation/ari_benchmark/figures/
Tables   → ./Revisions/outputs/simulation/ari_benchmark/

Usage (from repo root):
    ./.venv/bin/python Revisions/logic/python/simulation/ari/benchmark_ari.py
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

CURRENT_FILE = Path(__file__).resolve()
REVISION_PYTHON_ROOT = CURRENT_FILE.parents[2]
REPO_ROOT = CURRENT_FILE.parents[5]
SIMULATION_ROOT = REPO_ROOT / 'Simulation'

for path in (REVISION_PYTHON_ROOT, REPO_ROOT, SIMULATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.chdir(REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from revisionlib.metrics import bootstrap_ari_ci
from revisionlib.paths import SIMULATION_DATA_ROOT, ensure_directory, revision_output_dir
from revisionlib.truth import apply_biological_absence_truth

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = revision_output_dir('ari_benchmark')
FIGURE_DIR = ensure_directory(OUTPUT_DIR / 'figures')

N_BOOTSTRAP = 1000
SEED = 42

DEF_COLORS = [
    '#139593', '#fca311', '#e54f2a',
    '#c3c3c3', '#555555',
    '#690000', '#5f4a00', '#004549',
]
METHOD_COLORS = {
    'COPF': '#139593',
    'ProteoForge': '#e54f2a',
}
METHOD_MARKERS = {
    'COPF': 'o',
    'ProteoForge': '^',
}
METHOD_LINESTYLES = {
    'COPF': '--',
    'ProteoForge': ':',
}
METHOD_ORDER = ['COPF', 'ProteoForge']

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
})

sns.set_theme(
    style='white',
    context='paper',
    palette=DEF_COLORS,
    font_scale=1,
    rc={
        'figure.figsize': (6, 4),
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Ubuntu Mono', 'DejaVu Sans'],
    },
)

SIM1_EXPERIMENT_ORDER = ['twoPep', 'randomPep', 'halfPep', 'halfPlusPep']
SIM1_EXPERIMENT_LABELS = {
    'twoPep': 'Two Peptides',
    'randomPep': '2>50% Peptides',
    'halfPep': '50% Peptides',
    'halfPlusPep': '>50% Peptides',
}
SIM4_OVERLAP_ORDER = {'NonOverlap': 0, 'Overlap': 1}
SIM4_DIRECTION_ORDER = {'same': 0, 'random': 1}


def order_methods(methods) -> list[str]:
    ordered = [method for method in METHOD_ORDER if method in methods]
    extras = sorted(method for method in methods if method not in ordered)
    return ordered + extras


def format_rate(value: float) -> str:
    return f'{int(round(value * 100))}%'


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURE_DIR / filename, bbox_inches='tight', transparent=True)


def style_axis(ax: plt.Axes, grid_axis: str = 'y') -> None:
    ax.grid(grid_axis, linestyle='--', linewidth=0.75, alpha=0.5, color='lightgrey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_note(fig: plt.Figure, text: str, y: float = 0.985) -> None:
    fig.text(
        0.99,
        y,
        text,
        ha='right',
        va='top',
        fontsize=10,
        fontstyle='italic',
        color='gray',
    )


def scenario_sort_key(sim_id: str, scenario: str) -> tuple:
    if sim_id == 'Sim1':
        parts = scenario.split('_')
        return (
            SIM1_EXPERIMENT_ORDER.index(parts[1]),
            0 if parts[2] == 'complete' else 1,
        )

    if sim_id == 'Sim2':
        parts = scenario.split('_')
        return (
            float(parts[1].replace('Pro', '')),
            float(parts[2].replace('Pep', '')),
        )

    if sim_id == 'Sim3':
        parts = scenario.split('_')
        return (float(parts[1]), float(parts[2]))

    if sim_id == 'Sim4':
        parts = scenario.split('_')
        return (
            int(parts[1].replace('Cond', '')),
            SIM4_OVERLAP_ORDER[parts[2]],
            SIM4_DIRECTION_ORDER[parts[3].replace('Dir', '')],
        )

    return (scenario,)


def scenario_display_label(sim_id: str, scenario: str) -> str:
    parts = scenario.split('_')

    if sim_id == 'Sim1':
        experiment = SIM1_EXPERIMENT_LABELS.get(parts[1], parts[1])
        return f'{experiment}\n{parts[2].capitalize()}'

    if sim_id == 'Sim2':
        return f'Pro {format_rate(float(parts[1].replace("Pro", "")))}\nPep {format_rate(float(parts[2].replace("Pep", "")))}'

    if sim_id == 'Sim3':
        return f'{float(parts[1]):.2f}-{float(parts[2]):.2f}'

    if sim_id == 'Sim4':
        n_cond = parts[1].replace('Cond', 'C')
        overlap = 'Overlap' if parts[2] == 'Overlap' else 'Non-overlap'
        direction = parts[3].replace('Dir', '').capitalize()
        return f'{n_cond}\n{overlap}\n{direction}'

    return scenario

# ── Scenario catalogue ──────────────────────────────────────────────────────
# Each entry: (sim_id, data_dir, file_prefix, scenario_label, has_copf)
# We discover files automatically but need to know the naming conventions.

def discover_scenarios(data_root=SIMULATION_DATA_ROOT):
    """Auto-discover all Sim1–Sim4 result files for ProteoForge and COPF."""
    scenarios = []

    for sim_id in ["Sim1", "Sim2", "Sim3", "Sim4"]:
        sim_dir = Path(data_root) / sim_id
        if not sim_dir.is_dir():
            continue

        pf_files = sorted([
            path.name for path in sim_dir.iterdir()
            if path.name.endswith("_ProteoForge_ResultData.feather")
            # Exclude model-benchmark variants (glm, mqr, ols, wls, quantile, rlm_*)
            and not any(x in path.name for x in [
                "_glm_", "_mqr_", "_ols_", "_wls_", "_quantile_",
                "_rlm_default_", "_rlm_impute_", "_rlm_no_weight_", "_rlm_revtech_",
            ])
        ])

        for pf_file in pf_files:
            prefix = pf_file.replace("_ProteoForge_ResultData.feather", "")
            copf_file = f"{prefix}_COPF_ResultData.feather"
            has_copf = (sim_dir / copf_file).exists()

            scenarios.append({
                "sim_id": sim_id,
                "prefix": prefix,
                "label": f"{sim_id}/{prefix}",
                "sim_dir": sim_dir,
                "pf_path": sim_dir / pf_file,
                "copf_path": (sim_dir / copf_file) if has_copf else None,
            })

    return scenarios


def load_input_truth(sim_dir: Path, prefix: str) -> pd.DataFrame | None:
    input_path = sim_dir / f'{prefix}_InputData.feather'
    if not input_path.exists():
        return None

    truth = pd.read_feather(input_path)
    truth = truth[[
        column for column in ['Protein', 'Peptide', 'PeptideID', 'pertPFG', 'pertPeptide', 'isCompMiss', 'Reason']
        if column in truth.columns
    ]].drop_duplicates().copy()
    truth = apply_biological_absence_truth(truth)
    return truth


def prepare_proteoforge_frame(df: pd.DataFrame, truth: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if truth is not None and 'PeptideID' in out.columns and 'PeptideID' in truth.columns:
        merged = out.merge(
            truth[['Protein', 'PeptideID', 'pertPFG']],
            on=['Protein', 'PeptideID'],
            how='left',
            suffixes=('', '_truth'),
        )
        if 'pertPFG_truth' in merged.columns:
            merged['ariTruthPFG'] = merged['pertPFG_truth'].fillna(merged['pertPFG'])
            return merged
    out['ariTruthPFG'] = out['pertPFG']
    return out


def prepare_copf_frame(df: pd.DataFrame, truth: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if 'id' in out.columns:
        out['Peptide'] = out['id'].astype(str).str.split('-', n=1).str[1]
    if truth is not None and 'Peptide' in out.columns and 'Peptide' in truth.columns:
        merged = out.merge(
            truth[['Protein', 'Peptide', 'pertPFG']].rename(columns={'Protein': 'protein_id'}),
            on=['protein_id', 'Peptide'],
            how='left',
            suffixes=('', '_truth'),
        )
        if 'pertPFG_truth' in merged.columns:
            merged['ariTruthPFG'] = merged['pertPFG_truth'].fillna(merged['pertPFG'])
            return merged
    out['ariTruthPFG'] = out['pertPFG']
    return out


def compute_ari_for_proteoforge(df):
    results = {}

    # Mode 1: Per-protein, perturbed only
    r1 = bootstrap_ari_ci(
        df, true_col='ariTruthPFG', pred_col='ClusterID', protein_col='Protein',
        mode="per_protein", include_unperturbed=False,
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )
    results["perturbed_per_protein"] = r1

    # Mode 2: Per-protein, all proteins
    r2 = bootstrap_ari_ci(
        df, true_col='ariTruthPFG', pred_col='ClusterID', protein_col='Protein',
        mode="per_protein", include_unperturbed=True,
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )
    results["all_per_protein"] = r2

    # Mode 3: Concatenated, all proteins
    r3 = bootstrap_ari_ci(
        df, true_col='ariTruthPFG', pred_col='ClusterID', protein_col='Protein',
        mode="concatenated", include_unperturbed=True,
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )
    results["concatenated_all"] = r3

    return results


def compute_ari_for_copf(df):
    results = {}

    r1 = bootstrap_ari_ci(
        df, true_col='ariTruthPFG', pred_col='cluster', protein_col='protein_id',
        mode="per_protein", include_unperturbed=False,
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )
    results["perturbed_per_protein"] = r1

    r2 = bootstrap_ari_ci(
        df, true_col='ariTruthPFG', pred_col='cluster', protein_col='protein_id',
        mode="per_protein", include_unperturbed=True,
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )
    results["all_per_protein"] = r2

    r3 = bootstrap_ari_ci(
        df, true_col='ariTruthPFG', pred_col='cluster', protein_col='protein_id',
        mode="concatenated", include_unperturbed=True,
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )
    results["concatenated_all"] = r3

    return results


# ==============================================================================
# Main
# ==============================================================================
print("=" * 70)
print("  12-ARIBenchmark: Adjusted Rand Index for Peptide Grouping")
print("=" * 70)

scenarios = discover_scenarios()
print(f"  Discovered {len(scenarios)} scenarios across Sim1–Sim4\n")

all_rows = []

for sc in scenarios:
    print(f"  {sc['label']}...")

    # ── ProteoForge ──────────────────────────────────────────────────────────
    try:
        truth_df = load_input_truth(sc['sim_dir'], sc['prefix'])
        pf_df = prepare_proteoforge_frame(pd.read_feather(sc["pf_path"]), truth_df)
        pf_ari = compute_ari_for_proteoforge(pf_df)
        for mode_name, ari_dict in pf_ari.items():
            all_rows.append({
                "sim_id": sc["sim_id"],
                "scenario": sc["prefix"],
                "method": "ProteoForge",
                "mode": mode_name,
                "ARI": ari_dict["ARI"],
                "CI_lower": ari_dict["CI_lower"],
                "CI_upper": ari_dict["CI_upper"],
                "n_proteins": ari_dict["n_proteins"],
            })
        print(f"    PF:  perturbed ARI={pf_ari['perturbed_per_protein']['ARI']:.3f} "
              f"[{pf_ari['perturbed_per_protein']['CI_lower']:.3f}, "
              f"{pf_ari['perturbed_per_protein']['CI_upper']:.3f}]")
    except Exception as e:
        print(f"    PF:  FAILED — {e}")

    # ── COPF ─────────────────────────────────────────────────────────────────
    if sc["copf_path"] is not None:
        try:
            copf_df = prepare_copf_frame(pd.read_feather(sc["copf_path"]), truth_df)
            copf_ari = compute_ari_for_copf(copf_df)
            for mode_name, ari_dict in copf_ari.items():
                all_rows.append({
                    "sim_id": sc["sim_id"],
                    "scenario": sc["prefix"],
                    "method": "COPF",
                    "mode": mode_name,
                    "ARI": ari_dict["ARI"],
                    "CI_lower": ari_dict["CI_lower"],
                    "CI_upper": ari_dict["CI_upper"],
                    "n_proteins": ari_dict["n_proteins"],
                })
            print(f"    COPF: perturbed ARI={copf_ari['perturbed_per_protein']['ARI']:.3f} "
                  f"[{copf_ari['perturbed_per_protein']['CI_lower']:.3f}, "
                  f"{copf_ari['perturbed_per_protein']['CI_upper']:.3f}]")
        except Exception as e:
            print(f"    COPF: FAILED — {e}")

results_df = pd.DataFrame(all_rows)
results_csv = OUTPUT_DIR / 'ari_summary_all.csv'
results_df.to_csv(results_csv, index=False)
print(f"\n  Saved: {results_csv} ({len(results_df)} rows)")


# ==============================================================================
# Figures
# ==============================================================================
print("\n" + "=" * 70)
print("  Generating Figures")
print("=" * 70)

# ── Figure 1: Per-Sim grouped bar chart (perturbed-only ARI) ─────────────────
for sim_id in ["Sim1", "Sim2", "Sim3", "Sim4"]:
    sub = results_df[
        (results_df["sim_id"] == sim_id) &
        (results_df["mode"] == "perturbed_per_protein")
    ].copy()
    if sub.empty:
        continue

    ordered_scenarios = sorted(sub['scenario'].unique(), key=lambda name: scenario_sort_key(sim_id, name))
    x_labels = [scenario_display_label(sim_id, name) for name in ordered_scenarios]
    methods = order_methods(sub['method'].unique())

    fig_width = max(10, min(20, len(ordered_scenarios) * 0.95))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    x = np.arange(len(ordered_scenarios))
    width = 0.78 / max(1, len(methods))

    for i, method in enumerate(methods):
        m_sub = sub[sub["method"] == method].set_index("scenario")
        vals = [m_sub.loc[s, 'ARI'] if s in m_sub.index else np.nan for s in ordered_scenarios]
        ci_lo = [m_sub.loc[s, 'CI_lower'] if s in m_sub.index else np.nan for s in ordered_scenarios]
        ci_hi = [m_sub.loc[s, 'CI_upper'] if s in m_sub.index else np.nan for s in ordered_scenarios]
        yerr_lo = [v - lo for v, lo in zip(vals, ci_lo)]
        yerr_hi = [hi - v for v, hi in zip(vals, ci_hi)]

        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width, label=method,
            color=METHOD_COLORS.get(method, "#999"),
            edgecolor="black", linewidth=0.5,
            yerr=[yerr_lo, yerr_hi], capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title(f'{sim_id}: Peptide Grouping ARI on Perturbed Proteins')
    ax.legend(frameon=False, title='Method')
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    style_axis(ax, grid_axis='y')
    add_note(fig, 'Mode: per-protein mean ARI on perturbed proteins only; whiskers show bootstrap 95% CI.')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, f'ari_perturbed_{sim_id}.png')
    plt.close(fig)
    print(f"  Saved: {FIGURE_DIR}/ari_perturbed_{sim_id}.png")


# ── Figure 2: Three-mode comparison for a representative scenario ────────────
# Pick Sim2 Pro0.2_Pep0.2 as the representative
rep_scenario = "2_Pro0.2_Pep0.2_imputed"
rep_sub = results_df[results_df["scenario"] == rep_scenario].copy()

if not rep_sub.empty:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    modes = ["perturbed_per_protein", "all_per_protein", "concatenated_all"]
    mode_labels = [
        "Perturbed only\n(per-protein)",
        "All proteins\n(per-protein)",
        "All proteins\n(concatenated)",
    ]
    methods = order_methods(rep_sub['method'].unique())
    x = np.arange(len(modes))
    width = 0.78 / max(1, len(methods))

    for i, method in enumerate(methods):
        m_sub = rep_sub[rep_sub["method"] == method].set_index("mode")
        vals, lo, hi = [], [], []
        for mode in modes:
            if mode in m_sub.index:
                vals.append(m_sub.loc[mode, "ARI"])
                lo.append(max(0, m_sub.loc[mode, "ARI"] - m_sub.loc[mode, "CI_lower"]))
                hi.append(max(0, m_sub.loc[mode, "CI_upper"] - m_sub.loc[mode, "ARI"]))
            else:
                vals.append(0)
                lo.append(0)
                hi.append(0)

        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width, label=method,
            color=METHOD_COLORS.get(method, "#999"),
            edgecolor="black", linewidth=0.5,
            yerr=[lo, hi], capsize=4,
        )

        for bar, value in zip(bars, vals, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.03,
                f'{value:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color=METHOD_COLORS.get(method, '#999'),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=10)
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title('Sim2 Pro0.2/Pep0.2: ARI Reporting Modes')
    ax.legend(frameon=False, title='Method')
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    style_axis(ax, grid_axis='y')
    add_note(fig, 'Per-sim benchmark figures report the first mode: perturbed-only, per-protein mean ARI.', y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, 'ari_three_modes.png')
    plt.close(fig)
    print(f"  Saved: {FIGURE_DIR}/ari_three_modes.png")


# ── Figure 3: Sim2 missingness sweep — ARI vs missingness level ─────────────
sim2_sub = results_df[
    (results_df["sim_id"] == "Sim2") &
    (results_df["mode"] == "perturbed_per_protein")
].copy()

if not sim2_sub.empty:
    # Extract Pro and Pep rates from scenario name
    def parse_miss(name):
        parts = name.split("_")
        pro = float(parts[1].replace("Pro", ""))
        pep = float(parts[2].replace("Pep", ""))
        return pro, pep

    sim2_sub[["pro_miss", "pep_miss"]] = sim2_sub["scenario"].apply(
        lambda x: pd.Series(parse_miss(x))
    )

    heatmap_vmin = np.floor(sim2_sub['ARI'].min() * 20) / 20
    heatmap_vmax = np.ceil(sim2_sub['ARI'].max() * 20) / 20

    for method in order_methods(sim2_sub['method'].unique()):
        m_sub = sim2_sub[sim2_sub["method"] == method].copy()
        pivot = m_sub.pivot(index='pro_miss', columns='pep_miss', values='ARI').sort_index().sort_index(axis=1)
        pivot = pivot.iloc[::-1]

        fig, ax = plt.subplots(figsize=(6.5, 5.25))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            vmin=heatmap_vmin,
            vmax=heatmap_vmax,
            linewidths=0.5,
            linecolor='white',
            square=True,
            cbar_kws={'label': 'Adjusted Rand Index'},
            annot_kws={'size': 11, 'weight': 'bold'},
            ax=ax,
        )
        ax.set_title(f'{method}: ARI Across Missingness Levels')
        ax.set_xlabel('Peptide Missingness Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Protein Missingness Rate', fontsize=12, fontweight='bold')
        ax.set_xticklabels([format_rate(value) for value in pivot.columns], rotation=0)
        ax.set_yticklabels([format_rate(value) for value in pivot.index], rotation=0)
        add_note(fig, 'Mode: per-protein mean ARI on perturbed proteins only.', y=0.97)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fname = f"ari_sim2_heatmap_{method.lower().replace(' ', '_')}.png"
        save_figure(fig, fname)
        plt.close(fig)
        print(f"  Saved: {FIGURE_DIR}/{fname}")


# ── Figure 4: Sim3 perturbation magnitude effect on ARI ─────────────────────
sim3_sub = results_df[
    (results_df["sim_id"] == "Sim3") &
    (results_df["mode"] == "perturbed_per_protein")
].copy()

if not sim3_sub.empty:
    ordered_scenarios = sorted(sim3_sub['scenario'].unique(), key=lambda name: scenario_sort_key('Sim3', name))
    x = np.arange(len(ordered_scenarios))
    x_labels = [scenario_display_label('Sim3', name) for name in ordered_scenarios]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in order_methods(sim3_sub['method'].unique()):
        m_sub = sim3_sub[sim3_sub['method'] == method].set_index('scenario').loc[ordered_scenarios].reset_index()
        color = METHOD_COLORS.get(method, '#999')
        ax.plot(
            x,
            m_sub['ARI'],
            label=method,
            color=color,
            linestyle=METHOD_LINESTYLES.get(method, '-'),
            linewidth=2.5,
            marker=METHOD_MARKERS.get(method, 'o'),
            markersize=9,
            markeredgewidth=0.5,
            markeredgecolor='black',
        )
        ax.fill_between(
            x,
            m_sub['CI_lower'],
            m_sub['CI_upper'],
            color=color,
            alpha=0.15,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel('Perturbation Range (log2 fold-change)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Adjusted Rand Index', fontsize=12, fontweight='bold')
    ax.set_title('Sim3: ARI Sensitivity Across Perturbation Magnitudes')
    ax.legend(frameon=False, title='Method')
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    style_axis(ax, grid_axis='y')
    add_note(fig, 'Lines show perturbed-only, per-protein mean ARI; shaded bands show bootstrap 95% CI.', y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, 'ari_sim3_magnitude.png')
    plt.close(fig)
    print(f"  Saved: {FIGURE_DIR}/ari_sim3_magnitude.png")


print("\n" + "=" * 70)
print("  12-ARIBenchmark complete.")
print("=" * 70)
