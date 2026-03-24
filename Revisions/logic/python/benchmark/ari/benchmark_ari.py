#!/usr/bin/env python3
"""
benchmark_ari.py
================
Compute Adjusted Rand Index (ARI) for peptide grouping quality on the
SWATH-MS InterLab Benchmark datasets.

This mirrors the Simulation revision ARI analysis, but adapts the truth
definition to the Benchmark data structure:
  - truth labels come from `perturbed_peptide` in `bench_*_input.feather`
  - ProteoForge benchmark outputs are sample-level, so they are collapsed to
    one row per peptide before scoring
  - the `1pep` scenario is excluded because the Benchmark grouping notebook
    treats it as non-informative for grouping quality

Reporting modes:
  1. Per-protein mean ARI (perturbed proteins only) + bootstrap 95% CI
  2. Per-protein mean ARI (all proteins, unperturbed scored 1.0 if 1 cluster)
  3. Concatenated ARI (all peptides pooled) + CI

Usage (from repo root):
    ./.venv/bin/python Revisions/logic/python/benchmark/ari/benchmark_ari.py
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

CURRENT_FILE = Path(__file__).resolve()
REVISION_PYTHON_ROOT = CURRENT_FILE.parents[2]
REPO_ROOT = CURRENT_FILE.parents[5]
BENCHMARK_ROOT = REPO_ROOT / 'Benchmark'

for path in (REVISION_PYTHON_ROOT, REPO_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.chdir(REPO_ROOT)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from revisionlib.metrics import bootstrap_ari_ci
from revisionlib.paths import (
    BENCHMARK_DATA_ROOT,
    BENCHMARK_RESULTS_ROOT,
    ensure_directory,
    revision_benchmark_output_dir,
)

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = revision_benchmark_output_dir('ari_benchmark')
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
METHOD_ORDER = ['COPF', 'ProteoForge']

SCENARIO_ORDER = ['2pep', 'random', '050pep']
SCENARIO_LABELS = {
    '2pep': '2 Peptides',
    'random': 'Random (2 to %50) Peptides',
    '050pep': '%50 Peptides',
}
MODE_ORDER = ['perturbed_per_protein', 'all_per_protein', 'concatenated_all']
MODE_LABELS = {
    'perturbed_per_protein': 'Perturbed only\n(per-protein)',
    'all_per_protein': 'All proteins\n(per-protein)',
    'concatenated_all': 'All proteins\n(concatenated)',
}

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


def order_methods(methods) -> list[str]:
    ordered = [method for method in METHOD_ORDER if method in methods]
    extras = sorted(method for method in methods if method not in ordered)
    return ordered + extras


def style_axis(ax: plt.Axes, grid_axis: str = 'y') -> None:
    ax.grid(grid_axis, linestyle='--', linewidth=0.75, alpha=0.5, color='lightgrey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURE_DIR / filename, bbox_inches='tight', transparent=True)


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


def discover_scenarios(data_root: Path = BENCHMARK_DATA_ROOT) -> list[dict[str, Path | str | None]]:
    prepared_dir = Path(data_root) / 'prepared'
    results_dir = Path(data_root) / 'results'
    scenarios: list[dict[str, Path | str | None]] = []

    for data_id in SCENARIO_ORDER:
        pf_path = results_dir / f'ProteoForge_{data_id}_result.feather'
        if not pf_path.exists():
            continue

        input_path = prepared_dir / f'bench_{data_id}_input.feather'
        copf_path = results_dir / f'COPF_{data_id}_result.feather'
        scenarios.append({
            'scenario': data_id,
            'label': SCENARIO_LABELS[data_id],
            'input_path': input_path if input_path.exists() else None,
            'pf_path': pf_path,
            'copf_path': copf_path if copf_path.exists() else None,
        })

    return scenarios


def load_input_truth(input_path: Path | None) -> pd.DataFrame | None:
    if input_path is None or not input_path.exists():
        return None

    truth = pd.read_feather(input_path)
    keep_cols = [
        column for column in [
            'protein_id',
            'peptide_id',
            'perturbed_protein',
            'perturbed_peptide',
            'n_perturbed_peptides',
        ]
        if column in truth.columns
    ]
    truth = truth[keep_cols].drop_duplicates().copy()
    truth['ariTruthPFG'] = truth['perturbed_peptide'].fillna(False).astype(int)
    return truth


def collapse_to_peptide_level(
        data: pd.DataFrame,
        protein_col: str,
        peptide_col: str,
        true_col: str,
        pred_col: str,
) -> pd.DataFrame:
    uniqueness = data.groupby([protein_col, peptide_col], observed=True).agg({
        true_col: 'nunique',
        pred_col: 'nunique',
    })
    bad = uniqueness[(uniqueness[true_col] > 1) | (uniqueness[pred_col] > 1)]
    if not bad.empty:
        raise ValueError(
            f'Found inconsistent peptide-level labels for {len(bad)} peptides when collapsing {pred_col}.'
        )

    return data[[protein_col, peptide_col, true_col, pred_col]].drop_duplicates().copy()


def prepare_proteoforge_frame(df: pd.DataFrame, truth: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if truth is not None:
        out = out.merge(
            truth[['protein_id', 'peptide_id', 'ariTruthPFG']],
            on=['protein_id', 'peptide_id'],
            how='left',
            suffixes=('', '_truth'),
        )

    if 'ariTruthPFG' not in out.columns:
        if 'perturbed_peptide' not in out.columns:
            raise ValueError('ProteoForge benchmark results are missing truth columns and no prepared input was found.')
        out['ariTruthPFG'] = out['perturbed_peptide'].fillna(False).astype(int)
    else:
        out['ariTruthPFG'] = out['ariTruthPFG'].fillna(out['perturbed_peptide'].fillna(False).astype(int)).astype(int)

    return collapse_to_peptide_level(
        out,
        protein_col='protein_id',
        peptide_col='peptide_id',
        true_col='ariTruthPFG',
        pred_col='ClusterID',
    )


def prepare_copf_frame(df: pd.DataFrame, truth: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if 'peptide_id' not in out.columns and 'id' in out.columns:
        out['peptide_id'] = out['id']

    if truth is not None:
        out = out.merge(
            truth[['protein_id', 'peptide_id', 'ariTruthPFG']],
            on=['protein_id', 'peptide_id'],
            how='left',
            suffixes=('', '_truth'),
        )

    if 'ariTruthPFG' not in out.columns:
        if 'perturbed_peptide' not in out.columns:
            raise ValueError('COPF benchmark results are missing truth columns and no prepared input was found.')
        out['ariTruthPFG'] = out['perturbed_peptide'].fillna(False).astype(int)
    else:
        out['ariTruthPFG'] = out['ariTruthPFG'].fillna(out['perturbed_peptide'].fillna(False).astype(int)).astype(int)

    return collapse_to_peptide_level(
        out,
        protein_col='protein_id',
        peptide_col='peptide_id',
        true_col='ariTruthPFG',
        pred_col='cluster',
    )


def compute_ari(df: pd.DataFrame, pred_col: str) -> dict[str, dict[str, float]]:
    results = {}

    results['perturbed_per_protein'] = bootstrap_ari_ci(
        df,
        true_col='ariTruthPFG',
        pred_col=pred_col,
        protein_col='protein_id',
        mode='per_protein',
        include_unperturbed=False,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )
    results['all_per_protein'] = bootstrap_ari_ci(
        df,
        true_col='ariTruthPFG',
        pred_col=pred_col,
        protein_col='protein_id',
        mode='per_protein',
        include_unperturbed=True,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )
    results['concatenated_all'] = bootstrap_ari_ci(
        df,
        true_col='ariTruthPFG',
        pred_col=pred_col,
        protein_col='protein_id',
        mode='concatenated',
        include_unperturbed=True,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )

    return results


print('=' * 70)
print('  Benchmark ARI: Adjusted Rand Index for Peptide Grouping')
print('=' * 70)

scenarios = discover_scenarios()
print(f'  Discovered {len(scenarios)} Benchmark scenarios\n')

all_rows: list[dict[str, object]] = []

for sc in scenarios:
    print(f"  Benchmark/{sc['scenario']}...")

    truth_df = load_input_truth(sc['input_path'])

    try:
        pf_df = prepare_proteoforge_frame(pd.read_feather(sc['pf_path']), truth_df)
        pf_ari = compute_ari(pf_df, pred_col='ClusterID')
        for mode_name, ari_dict in pf_ari.items():
            all_rows.append({
                'scenario': sc['scenario'],
                'scenario_label': sc['label'],
                'method': 'ProteoForge',
                'mode': mode_name,
                'ARI': ari_dict['ARI'],
                'CI_lower': ari_dict['CI_lower'],
                'CI_upper': ari_dict['CI_upper'],
                'n_proteins': ari_dict['n_proteins'],
            })
        print(
            f"    PF:  perturbed ARI={pf_ari['perturbed_per_protein']['ARI']:.3f} "
            f"[{pf_ari['perturbed_per_protein']['CI_lower']:.3f}, "
            f"{pf_ari['perturbed_per_protein']['CI_upper']:.3f}]"
        )
    except Exception as exc:
        print(f'    PF:  FAILED — {exc}')

    if sc['copf_path'] is not None:
        try:
            copf_df = prepare_copf_frame(pd.read_feather(sc['copf_path']), truth_df)
            copf_ari = compute_ari(copf_df, pred_col='cluster')
            for mode_name, ari_dict in copf_ari.items():
                all_rows.append({
                    'scenario': sc['scenario'],
                    'scenario_label': sc['label'],
                    'method': 'COPF',
                    'mode': mode_name,
                    'ARI': ari_dict['ARI'],
                    'CI_lower': ari_dict['CI_lower'],
                    'CI_upper': ari_dict['CI_upper'],
                    'n_proteins': ari_dict['n_proteins'],
                })
            print(
                f"    COPF: perturbed ARI={copf_ari['perturbed_per_protein']['ARI']:.3f} "
                f"[{copf_ari['perturbed_per_protein']['CI_lower']:.3f}, "
                f"{copf_ari['perturbed_per_protein']['CI_upper']:.3f}]"
            )
        except Exception as exc:
            print(f'    COPF: FAILED — {exc}')

results_df = pd.DataFrame(all_rows)
results_csv = OUTPUT_DIR / 'ari_summary_all.csv'
results_df.to_csv(results_csv, index=False)
print(f'\n  Saved: {results_csv} ({len(results_df)} rows)')


print('\n' + '=' * 70)
print('  Generating Figures')
print('=' * 70)

sub = results_df[results_df['mode'] == 'perturbed_per_protein'].copy()
if not sub.empty:
    ordered_scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in sub['scenario'].unique()]
    methods = order_methods(sub['method'].unique())
    x = np.arange(len(ordered_scenarios))
    width = 0.78 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for i, method in enumerate(methods):
        m_sub = sub[sub['method'] == method].set_index('scenario')
        vals = [m_sub.loc[s, 'ARI'] if s in m_sub.index else np.nan for s in ordered_scenarios]
        ci_lo = [m_sub.loc[s, 'CI_lower'] if s in m_sub.index else np.nan for s in ordered_scenarios]
        ci_hi = [m_sub.loc[s, 'CI_upper'] if s in m_sub.index else np.nan for s in ordered_scenarios]
        yerr_lo = [v - lo for v, lo in zip(vals, ci_lo)]
        yerr_hi = [hi - v for v, hi in zip(vals, ci_hi)]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=method,
            color=METHOD_COLORS.get(method, '#999999'),
            edgecolor='black',
            linewidth=0.5,
            yerr=[yerr_lo, yerr_hi],
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in ordered_scenarios], rotation=0)
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title('Benchmark: Peptide Grouping ARI on Perturbed Proteins')
    ax.legend(frameon=False, title='Method')
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    style_axis(ax, grid_axis='y')
    add_note(fig, 'Mode: per-protein mean ARI on perturbed proteins only; whiskers show bootstrap 95% CI.')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, 'ari_perturbed_benchmark.png')
    plt.close(fig)
    print(f'  Saved: {FIGURE_DIR}/ari_perturbed_benchmark.png')


if not results_df.empty:
    ordered_scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in results_df['scenario'].unique()]
    methods = order_methods(results_df['method'].unique())
    fig, axes = plt.subplots(1, len(ordered_scenarios), figsize=(4.5 * len(ordered_scenarios), 5.5), sharey=True)
    if len(ordered_scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, ordered_scenarios, strict=False):
        scenario_sub = results_df[results_df['scenario'] == scenario].copy()
        x = np.arange(len(MODE_ORDER))
        width = 0.78 / max(1, len(methods))

        for i, method in enumerate(methods):
            m_sub = scenario_sub[scenario_sub['method'] == method].set_index('mode')
            vals, lo, hi = [], [], []
            for mode in MODE_ORDER:
                if mode in m_sub.index:
                    vals.append(m_sub.loc[mode, 'ARI'])
                    lo.append(max(0, m_sub.loc[mode, 'ARI'] - m_sub.loc[mode, 'CI_lower']))
                    hi.append(max(0, m_sub.loc[mode, 'CI_upper'] - m_sub.loc[mode, 'ARI']))
                else:
                    vals.append(0)
                    lo.append(0)
                    hi.append(0)

                offset = (i - len(methods) / 2 + 0.5) * width

            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=method,
                color=METHOD_COLORS.get(method, '#999999'),
                edgecolor='black',
                linewidth=0.5,
                yerr=[lo, hi],
                capsize=4,
            )

            for bar, value in zip(bars, vals, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.03,
                    f'{value:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color=METHOD_COLORS.get(method, '#999999'),
                )

        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[mode] for mode in MODE_ORDER], fontsize=9)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        style_axis(ax, grid_axis='y')
        if ax is axes[0]:
            ax.set_ylabel('Adjusted Rand Index')
        else:
            ax.set_ylabel('')

    axes[0].legend(frameon=False, title='Method')
    add_note(fig, 'Three reporting modes match the Simulation ARI benchmark; labels come from Benchmark perturbed_peptide truth.', y=0.975)
    fig.suptitle('Benchmark: ARI Reporting Modes by Perturbation Scenario', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, 'ari_modes_benchmark.png')
    plt.close(fig)
    print(f'  Saved: {FIGURE_DIR}/ari_modes_benchmark.png')


print('\n' + '=' * 70)
print('  Benchmark ARI complete.')
print('=' * 70)