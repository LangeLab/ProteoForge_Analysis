#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

CURRENT_FILE = Path(__file__).resolve()
REVISION_PYTHON_ROOT = CURRENT_FILE.parents[1]
REPO_ROOT = CURRENT_FILE.parents[4]

for path in (REVISION_PYTHON_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.chdir(REPO_ROOT)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

from revisionlib.paths import ensure_directory


OUTPUT_ROOT = ensure_directory(REPO_ROOT / 'Revisions' / 'outputs' / 'panels' / 'supplementary_drafts')
SIMULATION_DIR = ensure_directory(OUTPUT_ROOT / 'simulation_design')
MODULE4_DIR = ensure_directory(OUTPUT_ROOT / 'module4_toy_example')
DEFAULT_FORMATS = ('png', 'svg')

STATE_COLORS = {
    'stable': '#c3c3c3',
    'perturbed': '#e54f2a',
    'observed': '#139593',
    'missing': '#fca311',
}
CLUSTER_COLORS = {
    0: '#8d99ae',
    1: '#e54f2a',
    2: '#fca311',
    3: '#139593',
    4: '#555555',
    5: '#690000',
    6: '#5f4a00',
}
DPF_COLORS = {
    'dPF_0': '#8d99ae',
    'dPF_1': '#e54f2a',
    'dPF_2': '#139593',
    'dPF_-1': '#fca311',
}
TOY_CONDITIONS = ['control', 'cond-1', 'cond-2', 'cond-3']
SIM1_CONDITION_BLOCKS = ['Cond 1', 'Cond 2', 'Cond 3']
SIM1_PATTERN_SPECS = [
    {'label': 'twoPep', 'positions': [3, 9], 'summary': '2 peptides'},
    {'label': 'randomPep', 'positions': [2, 4, 7, 10], 'summary': '10-50%'},
    {'label': 'halfPep', 'positions': [2, 3, 6, 7, 10, 11], 'summary': '50%'},
    {'label': 'halfPlusPep', 'positions': [1, 2, 3, 5, 6, 7, 9, 10], 'summary': '70%'},
]


sns.set_theme(
    style='white',
    context='paper',
    palette=['#139593', '#fca311', '#e54f2a', '#c3c3c3', '#555555'],
    font_scale=1,
    rc={
        'figure.figsize': (6, 4),
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Ubuntu Mono', 'DejaVu Sans'],
    },
)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
})


def style_axis(ax: plt.Axes, grid_axis: str = 'y') -> None:
    ax.grid(grid_axis, linestyle='--', linewidth=0.75, alpha=0.5, color='lightgrey')
    sns.despine(ax=ax)


def add_caption(fig: plt.Figure, text: str, width: int, y: float = 0.02, fontsize: float = 8.8) -> None:
    fig.text(
        0.02,
        y,
        textwrap.fill(text, width=width),
        ha='left',
        va='bottom',
        fontsize=fontsize,
        color='#555555',
        linespacing=1.35,
    )


def save_figure(fig: plt.Figure, stem: str, output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    saved_paths: list[Path] = []
    for fmt in formats:
        out_path = output_dir / f'{stem}.{fmt}'
        fig.savefig(out_path, bbox_inches='tight', transparent=True, pad_inches=0.02)
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def clear_outputs(output_dir: Path) -> None:
    for path in output_dir.glob('*'):
        if path.is_file():
            path.unlink()


def draw_stage_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    facecolor: str,
    edgecolor: str,
    accent_color: str,
    subtitle_width: int,
    title_fontsize: float,
    subtitle_fontsize: float,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle='round,pad=0.012,rounding_size=0.02',
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.add_patch(Rectangle((x, y + h - 0.04), w, 0.04, transform=ax.transAxes, facecolor=accent_color, edgecolor='none'))
    ax.text(x + w / 2, y + h * 0.63, title, transform=ax.transAxes, ha='center', va='center', fontsize=title_fontsize, weight='bold', color='#111111')
    ax.text(
        x + w / 2,
        y + h * 0.30,
        textwrap.fill(subtitle, width=subtitle_width),
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=subtitle_fontsize,
        color='#444444',
        linespacing=1.25,
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle='-|>',
            mutation_scale=12,
            linewidth=1.3,
            color='#555555',
            transform=ax.transAxes,
        )
    )


def add_process_split(ax: plt.Axes, x: float, y: float) -> None:
    ax.scatter([x], [y], transform=ax.transAxes, s=26, color='#555555', zorder=5)
    ax.plot([x, x], [y - 0.24, y + 0.20], transform=ax.transAxes, color='#555555', linewidth=1.2)


def sim1_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=STATE_COLORS['stable'], edgecolor='white', label='Unchanged peptide'),
        Patch(facecolor=STATE_COLORS['perturbed'], edgecolor='white', label='Perturbed peptide'),
        Patch(facecolor='#f8fbfb', edgecolor='#d2d2d2', label='Observed entry'),
        Patch(facecolor=STATE_COLORS['missing'], edgecolor='white', label='Missing entry'),
    ]


def draw_missingness_tile(ax: plt.Axes, title: str, note: str, method: str, missing_cells: set[tuple[int, int]]) -> None:
    n_rows = 8
    n_cols = 9
    ax.set_xlim(0, n_cols)
    ax.set_ylim(-0.92, n_rows + 1.45)
    ax.axis('off')
    ax.text(0, n_rows + 1.14, title, ha='left', va='top', fontsize=8.5, weight='bold')
    ax.text(0, n_rows + 0.60, textwrap.fill(note, width=34), ha='left', va='top', fontsize=7.2, color='#555555', linespacing=1.2)
    for row in range(n_rows):
        for col in range(n_cols):
            peptide_idx = n_rows - row
            facecolor = '#f8fbfb'
            edgecolor = '#d2d2d2'
            if (peptide_idx, col + 1) in missing_cells:
                facecolor = STATE_COLORS['missing']
                edgecolor = '#ffffff'
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.7))
    for boundary in [3, 6]:
        ax.plot([boundary, boundary], [0, n_rows], color='#888888', linewidth=1.05)
    for idx, label in enumerate(SIM1_CONDITION_BLOCKS):
        ax.text(idx * 3 + 1.5, n_rows + 0.08, label, ha='center', va='bottom', fontsize=7.4, color='#333333')
    for rep in range(3):
        for cond_idx in range(3):
            ax.text(cond_idx * 3 + rep + 0.5, -0.10, f'R{rep + 1}', ha='center', va='top', fontsize=6.9, color='#666666')
    for row in range(n_rows):
        ax.text(-0.15, row + 0.5, f'P{n_rows - row}', ha='right', va='center', fontsize=7.2, color='#666666')
    ax.text(0, -0.66, method, ha='left', va='top', fontsize=7.4, color='#333333', weight='bold')


def plot_sim1_patterns(ax: plt.Axes) -> None:
    ax.set_title('A. Peptide perturbation patterns', loc='left')
    ax.set_xlim(-1.35, 12.85)
    ax.set_ylim(0.35, len(SIM1_PATTERN_SPECS) + 0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(i) for i in range(1, 13)], fontsize=7.8)
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True, length=0, pad=2)
    ax.set_yticks([])
    ax.text(5.9, len(SIM1_PATTERN_SPECS) + 0.43, 'Ordered peptides within one perturbed protein', ha='center', va='bottom', fontsize=7.9, color='#555555')
    for row_idx, spec in enumerate(SIM1_PATTERN_SPECS):
        y = len(SIM1_PATTERN_SPECS) - row_idx
        if row_idx % 2 == 0:
            ax.axhspan(y - 0.42, y + 0.42, color='#fbfbfb', zorder=0)
        ax.text(-0.55, y, spec['label'], ha='right', va='center', fontsize=9.1, weight='bold', color='#222222')
        for peptide_idx in range(1, 13):
            facecolor = STATE_COLORS['perturbed'] if peptide_idx in spec['positions'] else STATE_COLORS['stable']
            ax.add_patch(Rectangle((peptide_idx - 0.44, y - 0.22), 0.88, 0.44, facecolor=facecolor, edgecolor='white', linewidth=0.8))
        ax.text(12.75, y, spec['summary'], ha='right', va='center', fontsize=7.8, color='#666666')
    sns.despine(ax=ax, left=True, bottom=True)


def plot_sim1_workflow(ax: plt.Axes) -> None:
    ax.axis('off')
    ax.set_title('B. Complete and imputed branches', loc='left')
    draw_stage_box(ax, 0.04, 0.26, 0.22, 0.50, 'Peptide matrix', '500 proteins | 3 conditions | 10 replicates | 5-50 peptides per protein', '#fbfbfb', '#777777', '#d9d9d9', 27, 9.8, 8.0)
    draw_stage_box(ax, 0.34, 0.18, 0.24, 0.66, 'Perturb 250 proteins', '0.5-1.5 log2 shifts under twoPep, randomPep, halfPep, or halfPlusPep', '#fff6f1', STATE_COLORS['perturbed'], STATE_COLORS['perturbed'], 26, 9.8, 8.0)
    draw_stage_box(ax, 0.71, 0.56, 0.23, 0.21, 'Complete dataset', 'written directly after perturbation', '#f3fbfb', STATE_COLORS['observed'], STATE_COLORS['observed'], 18, 9.5, 7.7)
    draw_stage_box(ax, 0.71, 0.14, 0.23, 0.26, 'Imputed dataset', 'same perturbed matrix after missingness correction', '#fff8ee', STATE_COLORS['missing'], STATE_COLORS['missing'], 18, 9.5, 7.6)
    draw_arrow(ax, (0.26, 0.51), (0.34, 0.51))
    draw_arrow(ax, (0.58, 0.51), (0.64, 0.51))
    add_process_split(ax, 0.66, 0.51)
    draw_arrow(ax, (0.66, 0.66), (0.71, 0.66))
    draw_arrow(ax, (0.66, 0.27), (0.71, 0.27))


def plot_sim1_missingness(ax: plt.Axes) -> None:
    ax.axis('off')
    ax.set_title('C. Missingness added only in the imputed branch', loc='left')
    tile_specs = [
        {
            'title': 'Random missingness',
            'note': '100 non-perturbed proteins | up to 35% missing entries',
            'method': 'kNN imputation',
            'missing_cells': {(8, 2), (7, 5), (6, 7), (5, 3), (4, 9), (3, 1), (2, 6), (1, 4)},
        },
        {
            'title': 'Condition-complete missingness',
            'note': '100 non-perturbed proteins | one peptide absent across one condition',
            'method': 'Downshifted low-value imputation',
            'missing_cells': {(4, 4), (4, 5), (4, 6)},
        },
    ]
    for spec, xpos in zip(tile_specs, [0.03, 0.53]):
        inset = ax.inset_axes([xpos, 0.05, 0.42, 0.83])
        draw_missingness_tile(inset, spec['title'], spec['note'], spec['method'], spec['missing_cells'])


def build_sim1_figure() -> tuple[plt.Figure, str]:
    fig = plt.figure(figsize=(10.8, 8.35))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.19)
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[0.82, 0.90, 1.16], hspace=0.16)
    plot_sim1_patterns(fig.add_subplot(gs[0, 0]))
    plot_sim1_workflow(fig.add_subplot(gs[1, 0]))
    plot_sim1_missingness(fig.add_subplot(gs[2, 0]))
    fig.legend(handles=sim1_legend_handles(), loc='lower center', bbox_to_anchor=(0.5, 0.102), ncol=4, frameon=False, fontsize=7.9, handlelength=1.4, columnspacing=1.2)
    add_caption(
        fig,
        'Simulation 1 schematic. A shared peptide matrix is first perturbed in 250 proteins under four peptide-fraction patterns. That same perturbed matrix is then written directly as the complete dataset or passed into the imputed branch, where the manuscript-described missingness modes are introduced before correction. For visual clarity this panel emphasizes the two missingness modes described in the manuscript; the simulation code also applies additional sparse dropout to perturbed proteins within the imputed branch.',
        width=150,
    )
    return fig, 'sim1_design_schematic'


def infer_dpf_labels(assignments: pd.DataFrame) -> dict[int, str]:
    labels: dict[int, str] = {}
    next_label = 1
    grouped = assignments.groupby('ClusterID', sort=True)
    for cluster_id, cluster_df in grouped:
        peptide_count = cluster_df['Peptide'].nunique()
        has_sig = bool(cluster_df['isSignificant'].any())
        if not has_sig:
            labels[int(cluster_id)] = 'dPF_0'
        elif peptide_count == 1:
            labels[int(cluster_id)] = 'dPF_-1'
        else:
            labels[int(cluster_id)] = f'dPF_{next_label}'
            next_label += 1
    return labels


def build_module4_dataset() -> pd.DataFrame:
    base_pattern = {'control': 0.00, 'cond-1': -0.14, 'cond-2': 0.10, 'cond-3': 0.22}
    cluster_specs = [
        {'cluster_id': 1, 'start': 1, 'end': 3, 'offsets': {}, 'significant': set(), 'weak_support': set()},
        {'cluster_id': 2, 'start': 4, 'end': 7, 'offsets': {'cond-1': 0.80}, 'significant': {4, 5}, 'weak_support': {6, 7}},
        {'cluster_id': 3, 'start': 8, 'end': 8, 'offsets': {'cond-2': 0.92}, 'significant': {8}, 'weak_support': set()},
        {'cluster_id': 4, 'start': 9, 'end': 11, 'offsets': {'cond-1': -0.04}, 'significant': set(), 'weak_support': set()},
        {'cluster_id': 5, 'start': 12, 'end': 15, 'offsets': {'cond-3': -0.86}, 'significant': {12, 13}, 'weak_support': {14, 15}},
        {'cluster_id': 6, 'start': 16, 'end': 18, 'offsets': {'cond-2': 0.04}, 'significant': set(), 'weak_support': set()},
    ]
    rows: list[dict[str, object]] = []
    for cluster in cluster_specs:
        for peptide_id in range(cluster['start'], cluster['end'] + 1):
            wave = 0.018 * np.sin(peptide_id / 1.7)
            peptide_adjust = (peptide_id - 9) * 0.003
            is_significant = peptide_id in cluster['significant']
            is_weak_support = peptide_id in cluster['weak_support']
            for condition in TOY_CONDITIONS:
                offset = cluster['offsets'].get(condition, 0.0)
                if is_weak_support:
                    offset *= 0.45
                    offset += 0.03 * np.cos(peptide_id + TOY_CONDITIONS.index(condition))
                value = base_pattern[condition] + wave + peptide_adjust + offset
                if is_significant and condition in cluster['offsets']:
                    value += 0.03
                rows.append({
                    'Peptide': f'Pep{peptide_id}',
                    'PeptideID': peptide_id,
                    'Condition': condition,
                    'adjIntensity': value,
                    'ClusterID': int(cluster['cluster_id']),
                    'isSignificant': is_significant,
                    'hasWeakClusterSupport': is_weak_support,
                })
    df = pd.DataFrame(rows)
    dpf_map = infer_dpf_labels(df[['Peptide', 'ClusterID', 'isSignificant']].drop_duplicates())
    df['dPF'] = df['ClusterID'].map(dpf_map)
    df['module2Call'] = np.where(
        df['dPF'].eq('dPF_-1'),
        'singleton_sig',
        np.where(df['isSignificant'], 'significant_discordant', np.where(df['hasWeakClusterSupport'], 'clustered_nonsig', 'background')),
    )
    return df


def module4_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='#e54f2a', markeredgecolor='#222222', label='Significant discordant'),
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='white', markeredgecolor='#e54f2a', label='Weak but cluster-consistent'),
        Line2D([0], [0], marker='D', linestyle='None', markerfacecolor='#fca311', markeredgecolor='#222222', label='Singleton significant peptide'),
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='#c3c3c3', markeredgecolor='#777777', label='Background peptide'),
    ]


def add_module2_track(ax: plt.Axes, df: pd.DataFrame) -> None:
    transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    calls = df[['PeptideID', 'module2Call']].drop_duplicates().sort_values('PeptideID')
    style_map = {
        'significant_discordant': {'marker': 'o', 'facecolor': '#e54f2a', 'edgecolor': '#222222', 'size': 34},
        'clustered_nonsig': {'marker': 'o', 'facecolor': 'white', 'edgecolor': '#e54f2a', 'size': 34},
        'singleton_sig': {'marker': 'D', 'facecolor': '#fca311', 'edgecolor': '#222222', 'size': 34},
        'background': {'marker': 'o', 'facecolor': '#c3c3c3', 'edgecolor': '#777777', 'size': 26},
    }
    y = -0.24
    ax.text(0.35, y, 'Module 2 evidence', transform=transform, ha='right', va='center', fontsize=8.2, color='#555555', clip_on=False)
    for row in calls.itertuples(index=False):
        style = style_map[row.module2Call]
        ax.scatter(row.PeptideID, y, transform=transform, marker=style['marker'], s=style['size'], facecolors=style['facecolor'], edgecolors=style['edgecolor'], linewidth=1.0, clip_on=False, zorder=5)


def add_cluster_band(ax: plt.Axes, df: pd.DataFrame) -> None:
    transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    cluster_summary = df[['PeptideID', 'ClusterID']].drop_duplicates().groupby('ClusterID', as_index=False).agg(start=('PeptideID', 'min'), end=('PeptideID', 'max')).sort_values('ClusterID')
    y0 = -0.38
    ax.text(0.35, y0 + 0.035, 'Cluster layer', transform=transform, ha='right', va='center', fontsize=8.2, color='#555555', clip_on=False)
    for row in cluster_summary.itertuples(index=False):
        width = row.end - row.start + 1
        ax.add_patch(Rectangle((row.start - 0.5, y0), width, 0.07, transform=transform, facecolor=CLUSTER_COLORS[int(row.ClusterID)], edgecolor='none', alpha=0.18, clip_on=False))
        ax.text(row.start - 0.5 + width / 2, y0 + 0.035, f'C{row.ClusterID}', transform=transform, ha='center', va='center', fontsize=7.8, color='#333333', clip_on=False)


def add_dpf_band(ax: plt.Axes, df: pd.DataFrame) -> None:
    transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    cluster_summary = df[['PeptideID', 'ClusterID', 'dPF']].drop_duplicates().groupby(['ClusterID', 'dPF'], as_index=False).agg(start=('PeptideID', 'min'), end=('PeptideID', 'max')).sort_values('ClusterID')
    y0 = -0.52
    ax.text(0.35, y0 + 0.035, 'Module 4 assembly', transform=transform, ha='right', va='center', fontsize=8.2, color='#555555', clip_on=False)
    for row in cluster_summary.itertuples(index=False):
        width = row.end - row.start + 1
        ax.add_patch(Rectangle((row.start - 0.5, y0), width, 0.07, transform=transform, facecolor=DPF_COLORS.get(row.dPF, '#c3c3c3'), edgecolor='none', alpha=0.24, clip_on=False))
        ax.text(row.start - 0.5 + width / 2, y0 + 0.035, row.dPF, transform=transform, ha='center', va='center', fontsize=7.8, color='#333333', clip_on=False)


def build_module4_figure() -> tuple[plt.Figure, str]:
    df = build_module4_dataset()
    condition_colors = {'control': '#8d99ae', 'cond-1': '#48cae4', 'cond-2': '#0096c7', 'cond-3': '#023e8a'}
    fig, ax = plt.subplots(figsize=(12.8, 5.7))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.52)
    for condition in TOY_CONDITIONS:
        cond_df = df[df['Condition'] == condition].sort_values('PeptideID')
        ax.plot(cond_df['PeptideID'], cond_df['adjIntensity'], color=condition_colors[condition], marker='o', markersize=4.5, linewidth=1.2, alpha=0.9)
    add_module2_track(ax, df)
    add_cluster_band(ax, df)
    add_dpf_band(ax, df)
    ax.set_title('Large-protein toy example for Module 4 assembly', loc='left')
    ax.set_xlabel('Ordered Peptides (PeptideID)')
    ax.set_ylabel('Control Adjusted Intensity')
    ax.set_xticks(sorted(df['PeptideID'].unique()))
    ax.set_xlim(0.5, df['PeptideID'].max() + 0.5)
    style_axis(ax, grid_axis='both')
    ax.legend(handles=module4_legend_handles(), frameon=False, title='Module 2 evidence', ncol=2, loc='upper right', fontsize=8.2, title_fontsize=9)
    add_caption(
        fig,
        'Module 4 toy example. Peptides 4-7 and 12-15 form two discordant cluster patterns across conditions. Only a subset of those peptides is individually significant, while weaker but cluster-consistent peptides remain below the peptide-level threshold. The additional clustering layer groups those weaker peptides with the same discordant pattern, allowing Module 4 to assemble dPF_1 and dPF_2. Peptide 8 is a singleton significant event and is retained as dPF_-1, while the remaining clusters stay in dPF_0.',
        width=185,
        y=0.03,
    )
    return fig, 'module4_toy_example'


def export_panel(panel: str, formats: tuple[str, ...], clean: bool) -> list[Path]:
    panel_map = {
        'sim1': (SIMULATION_DIR, build_sim1_figure),
        'module4': (MODULE4_DIR, build_module4_figure),
    }
    output_dir, builder = panel_map[panel]
    if clean:
        clear_outputs(output_dir)
    figure, stem = builder()
    return save_figure(figure, stem, output_dir, formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export the retained supplementary panel figures.')
    parser.add_argument('--panel', choices=['sim1', 'module4', 'all'], default='all', help='Panel to export.')
    parser.add_argument('--clean', action='store_true', help='Remove existing files from the selected output folder before exporting.')
    parser.add_argument('--formats', nargs='+', choices=['png', 'svg'], default=list(DEFAULT_FORMATS), help='Output formats to write.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_panels = ['sim1', 'module4'] if args.panel == 'all' else [args.panel]
    formats = tuple(args.formats)
    exported_paths: list[Path] = []
    for panel in selected_panels:
        exported_paths.extend(export_panel(panel=panel, formats=formats, clean=args.clean))
    print(f'Exported {len(exported_paths)} files:')
    for path in exported_paths:
        print(f'- {path.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()