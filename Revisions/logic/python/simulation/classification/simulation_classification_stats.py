#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

CURRENT_FILE = Path(__file__).resolve()
REVISION_PYTHON_ROOT = CURRENT_FILE.parents[2]
REPO_ROOT = CURRENT_FILE.parents[5]

for path in (REVISION_PYTHON_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.chdir(REPO_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from revisionlib.paths import SIMULATION_DATA_ROOT, ensure_directory, revision_output_dir
from revisionlib.performance_stats import (
    DEFAULT_GROUPING_METHOD_PAIRS,
    DEFAULT_IDENTIFICATION_METHOD_PAIRS,
    FIXED_THRESHOLD,
    MATCHED_FPR,
    STANDARD_THRESHOLDS,
    best_mcc_statistic,
    build_identification_score_frame,
    combine_item_key,
    compute_delong_table,
    compute_paired_metric_table,
    grouping_calls_copf,
    grouping_calls_proteoforge,
    grouping_curve_and_activation,
    matched_sensitivity_statistic,
    peptide_position_identifier,
    operating_point_table,
    prepare_delong_display_table,
    resolve_worker_count,
    summarize_binary_scores,
)


OUTPUT_DIR = revision_output_dir("classification_stats")
FIGURE_DIR = ensure_directory(OUTPUT_DIR / "figures")
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
SIMULATION_ORDER = ["Sim1", "Sim2", "Sim3", "Sim4"]
SIMULATION_TITLES = {
    "Sim1": "Sim1: Complete vs Imputed",
    "Sim2": "Sim2: Missingness Levels",
    "Sim3": "Sim3: Perturbation Magnitude",
    "Sim4": "Sim4: Experimental Complexity",
}
IDENTIFICATION_METHOD_ORDER = ["ProteoForge", "COPF", "PeCorA"]
GROUPING_METHOD_ORDER = ["ProteoForge", "COPF"]
METHOD_COLORS = {
    "ProteoForge": "#e54f2a",
    "COPF": "#139593",
    "PeCorA": "#fca311",
}
DELONG_COMPARISON_ORDER = ["ProteoForge vs PeCorA", "ProteoForge vs COPF", "PeCorA vs COPF"]
SIM1_EXPERIMENT_LABELS = {
    "twoPep": "Two Peptides",
    "halfPlusPep": ">50% Peptides",
    "halfPep": "50% Peptides",
    "randomPep": "2>50% Peptides",
}
SIM1_EXPERIMENT_ORDER = ["twoPep", "randomPep", "halfPep", "halfPlusPep"]
SIM1_PATTERN = re.compile(
    r"^2_(?P<experiment>[^_]+)_(?P<data_type>complete|imputed)_(?P<method>ProteoForge|COPF|PeCorA)_ResultData\.feather$"
)
SIM2_PATTERN = re.compile(
    r"^2_Pro(?P<protein>[\d.]+)_Pep(?P<peptide>[\d.]+)_imputed_(?P<method>ProteoForge|COPF|PeCorA)_ResultData\.feather$"
)
SIM3_PATTERN = re.compile(
    r"^2_(?P<low>[\d.]+)_(?P<high>[\d.]+)_(?P<method>ProteoForge|COPF|PeCorA)_ResultData\.feather$"
)
SIM4_PATTERN = re.compile(
    r"^2_(?P<n_cond>\d+)Cond_(?P<overlap>Overlap|NonOverlap)_(?P<direction>same|random)Dir_(?P<method>ProteoForge|COPF|PeCorA)_ResultData\.feather$"
)


def resolve_input_data_path(result_path: Path) -> Path:
    input_name = re.sub(
        r"_(ProteoForge|COPF|PeCorA)_ResultData\.feather$",
        "_InputData.feather",
        result_path.name,
    )
    input_path = result_path.with_name(input_name)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing simulation input file for {result_path.name}: {input_path}")
    return input_path


def load_proteoforge_identification_truth(result_path: Path) -> pd.DataFrame:
    input_path = resolve_input_data_path(result_path)
    return (
        pd.read_feather(input_path)[["Protein", "PeptideID", "pertPeptide"]]
        .drop_duplicates()
        .copy()
    )


def _simulation_best_mcc_metric(y_true: np.ndarray, y_pval: np.ndarray) -> float:
    return best_mcc_statistic(y_true, y_pval, thresholds=STANDARD_THRESHOLDS)


def _simulation_matched_sensitivity_metric(y_true: np.ndarray, y_pval: np.ndarray) -> float:
    return matched_sensitivity_statistic(y_true, y_pval, matched_fpr=MATCHED_FPR)


def _run_parallel_jobs(tasks: list[dict[str, object]], worker_func, n_workers: int, desc: str, unit: str):
    if n_workers <= 1:
        return [worker_func(task) for task in tqdm(tasks, desc=desc, unit=unit)]

    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {
            executor.submit(worker_func, task): index
            for index, task in enumerate(tasks)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=desc, unit=unit):
            results[future_to_index[future]] = future.result()
    return results


def _simulation_identification_task(task: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame]:
    metadata = task["metadata"]
    n_bootstrap = int(task["n_bootstrap"])
    data = pd.read_feather(metadata["path"])
    method = metadata["method"]

    if method == "ProteoForge":
        truth = load_proteoforge_identification_truth(metadata["path"])
        data = (
            data[["Protein", "PeptideID", "adj_pval"]]
            .drop_duplicates()
            .merge(truth, on=["Protein", "PeptideID"], how="left", validate="one_to_one")
        )
        y_true = data["pertPeptide"].astype(bool).to_numpy()
        y_pval = data["adj_pval"].astype(float).to_numpy()
    elif method == "COPF":
        data = data[["protein_id", "id", "pertPeptide", "proteoform_score_pval"]].drop_duplicates().copy()
        y_true = data["pertPeptide"].astype(bool).to_numpy()
        y_pval = data["proteoform_score_pval"].astype(float).to_numpy()
    else:
        data = data[["Protein", "Peptide", "pertPeptide", "adj_pval"]].drop_duplicates().copy()
        y_true = data["pertPeptide"].astype(bool).to_numpy()
        y_pval = data["adj_pval"].astype(float).to_numpy()

    summary = summarize_binary_scores(
        y_true,
        y_pval,
        thresholds=STANDARD_THRESHOLDS,
        fixed_threshold=FIXED_THRESHOLD,
        n_bootstrap=n_bootstrap,
    )
    summary_row = {
        "simulation": metadata["simulation"],
        "level": "identification",
        "scenario_key": metadata["scenario_key"],
        "scenario_label": metadata["scenario_label"],
        "sort_1": metadata["sort_1"],
        "sort_2": metadata["sort_2"],
        "sort_3": metadata["sort_3"],
        "method": method,
        **summary,
    }

    if method == "ProteoForge":
        item_ids = data.apply(
            lambda row: combine_item_key(row["Protein"], peptide_position_identifier(row["PeptideID"])),
            axis=1,
        )
    elif method == "PeCorA":
        item_ids = data.apply(
            lambda row: combine_item_key(row["Protein"], peptide_position_identifier(row["Peptide"])),
            axis=1,
        )
    else:
        item_ids = data.apply(
            lambda row: combine_item_key(row["protein_id"], peptide_position_identifier(row["id"])),
            axis=1,
        )

    score_frame = build_identification_score_frame(
        item_ids=item_ids,
        truth_labels=y_true,
        pvalues=y_pval,
        method=method,
        metadata={
            "simulation": metadata["simulation"],
            "level": "identification",
            "scenario_key": metadata["scenario_key"],
            "scenario_label": metadata["scenario_label"],
            "sort_1": metadata["sort_1"],
            "sort_2": metadata["sort_2"],
            "sort_3": metadata["sort_3"],
        },
    )
    return summary_row, score_frame


def _simulation_grouping_task(task: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    metadata = task["metadata"]
    n_bootstrap = int(task["n_bootstrap"])
    data = pd.read_feather(metadata["path"])
    method = metadata["method"]

    if method == "ProteoForge":
        data = data[["Protein", "PeptideID", "pertProtein", "adj_pval", "ClusterID"]].drop_duplicates().copy()
        curve_df, activation_df = grouping_curve_and_activation(
            data=data,
            thresholds=STANDARD_THRESHOLDS,
            builder=lambda frame, thr: grouping_calls_proteoforge(
                frame,
                thr,
                protein_col="Protein",
                peptide_col="PeptideID",
                truth_col="pertProtein",
                pvalue_col="adj_pval",
                cluster_col="ClusterID",
            ),
            protein_col="Protein",
            truth_col="pertProtein",
        )
        activation_df = activation_df.rename(columns={"Protein": "item_id"})
    else:
        data = data[["protein_id", "id", "pertProtein", "proteoform_score_pval", "proteoform_score", "cluster"]].drop_duplicates().copy()
        curve_df, activation_df = grouping_curve_and_activation(
            data=data,
            thresholds=STANDARD_THRESHOLDS,
            builder=lambda frame, thr: grouping_calls_copf(
                frame,
                thr,
                protein_col="protein_id",
                truth_col="pertProtein",
                pvalue_col="proteoform_score_pval",
                score_col="proteoform_score",
                cluster_col="cluster",
            ),
            protein_col="protein_id",
            truth_col="pertProtein",
        )
        activation_df = activation_df.rename(columns={"protein_id": "item_id"})

    summary = summarize_binary_scores(
        activation_df["true_label"].to_numpy(),
        activation_df["score_pvalue"].to_numpy(),
        thresholds=STANDARD_THRESHOLDS,
        fixed_threshold=FIXED_THRESHOLD,
        n_bootstrap=n_bootstrap,
    )
    summary_row = {
        "simulation": metadata["simulation"],
        "level": "grouping",
        "scenario_key": metadata["scenario_key"],
        "scenario_label": metadata["scenario_label"],
        "sort_1": metadata["sort_1"],
        "sort_2": metadata["sort_2"],
        "sort_3": metadata["sort_3"],
        "method": method,
        **summary,
    }

    curve_df["simulation"] = metadata["simulation"]
    curve_df["level"] = "grouping"
    curve_df["scenario_key"] = metadata["scenario_key"]
    curve_df["scenario_label"] = metadata["scenario_label"]
    curve_df["sort_1"] = metadata["sort_1"]
    curve_df["sort_2"] = metadata["sort_2"]
    curve_df["sort_3"] = metadata["sort_3"]
    curve_df["method"] = method

    activation_df["simulation"] = metadata["simulation"]
    activation_df["level"] = "grouping"
    activation_df["scenario_key"] = metadata["scenario_key"]
    activation_df["scenario_label"] = metadata["scenario_label"]
    activation_df["sort_1"] = metadata["sort_1"]
    activation_df["sort_2"] = metadata["sort_2"]
    activation_df["sort_3"] = metadata["sort_3"]
    activation_df["method"] = method
    activation_df = activation_df[[
        "item_id", "true_label", "score_pvalue", "simulation", "level",
        "scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3", "method",
    ]].copy()
    return summary_row, curve_df, activation_df


def parse_result_metadata(path: Path) -> dict[str, object] | None:
    filename = path.name

    match = SIM1_PATTERN.match(filename)
    if match:
        experiment = match.group("experiment")
        data_type = match.group("data_type")
        return {
            "simulation": "Sim1",
            "method": match.group("method"),
            "path": path,
            "scenario_key": f"{data_type}::{experiment}",
            "scenario_label": f"{data_type.title()} | {SIM1_EXPERIMENT_LABELS[experiment]}",
            "sort_1": 0 if data_type == "complete" else 1,
            "sort_2": SIM1_EXPERIMENT_ORDER.index(experiment),
            "sort_3": 0,
        }

    match = SIM2_PATTERN.match(filename)
    if match:
        protein = float(match.group("protein"))
        peptide = float(match.group("peptide"))
        return {
            "simulation": "Sim2",
            "method": match.group("method"),
            "path": path,
            "scenario_key": f"P{int(round(protein * 100))}_Q{int(round(peptide * 100))}",
            "scenario_label": f"P{int(round(protein * 100))}% | pep {int(round(peptide * 100))}%",
            "sort_1": protein,
            "sort_2": peptide,
            "sort_3": 0,
        }

    match = SIM3_PATTERN.match(filename)
    if match:
        low = float(match.group("low"))
        high = float(match.group("high"))
        return {
            "simulation": "Sim3",
            "method": match.group("method"),
            "path": path,
            "scenario_key": f"{low:.2f}-{high:.2f}",
            "scenario_label": f"{low:.2f}-{high:.2f}",
            "sort_1": low,
            "sort_2": high,
            "sort_3": 0,
        }

    match = SIM4_PATTERN.match(filename)
    if match:
        n_cond = int(match.group("n_cond"))
        overlap = match.group("overlap")
        direction = match.group("direction")
        overlap_label = "Yes" if overlap == "Overlap" else "No"
        direction_label = "Same" if direction == "same" else "Random"
        return {
            "simulation": "Sim4",
            "method": match.group("method"),
            "path": path,
            "scenario_key": f"{n_cond}::{overlap_label}::{direction_label}",
            "scenario_label": f"{n_cond}C | {overlap_label} | {direction_label}",
            "sort_1": n_cond,
            "sort_2": 0 if overlap_label == "Yes" else 1,
            "sort_3": 0 if direction_label == "Same" else 1,
        }

    return None


def collect_result_index() -> pd.DataFrame:
    rows = []
    for simulation in SIMULATION_ORDER:
        folder = SIMULATION_DATA_ROOT / simulation
        paths = sorted(folder.glob("*_ResultData.feather"))
        for path in paths:
            metadata = parse_result_metadata(path)
            if metadata is not None:
                rows.append(metadata)
    df = pd.DataFrame(rows)
    df = df.sort_values(["simulation", "sort_1", "sort_2", "sort_3", "method"]).reset_index(drop=True)
    return df


def save_faceted_summary_plot(
        summary_df: pd.DataFrame,
        level: str,
        method_order: list[str],
        y_col: str,
        lower_col: str,
        upper_col: str,
        filename: str,
        title: str,
) -> None:
    subset = summary_df[summary_df["level"] == level].copy()
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    for axis, simulation in zip(axes, SIMULATION_ORDER, strict=False):
        sim_df = subset[subset["simulation"] == simulation].copy()
        if sim_df.empty:
            axis.set_visible(False)
            continue

        scenarios = (
            sim_df[["scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3"]]
            .drop_duplicates()
            .sort_values(["sort_1", "sort_2", "sort_3", "scenario_label"])
            .reset_index(drop=True)
        )
        x = np.arange(len(scenarios), dtype=float)

        for method in method_order:
            method_df = sim_df[sim_df["method"] == method].set_index("scenario_key")
            if method_df.empty:
                continue
            values = [method_df.loc[key, y_col] if key in method_df.index else np.nan for key in scenarios["scenario_key"]]
            lower = [
                max(values[idx] - method_df.loc[key, lower_col], 0.0)
                if key in method_df.index and np.isfinite(values[idx])
                else 0.0
                for idx, key in enumerate(scenarios["scenario_key"])
            ]
            upper = [
                max(method_df.loc[key, upper_col] - values[idx], 0.0)
                if key in method_df.index and np.isfinite(values[idx])
                else 0.0
                for idx, key in enumerate(scenarios["scenario_key"])
            ]
            axis.errorbar(
                x,
                values,
                yerr=[lower, upper],
                fmt="o-",
                linewidth=1.75,
                markersize=4,
                capsize=3,
                color=METHOD_COLORS[method],
                label=method,
            )

        axis.set_title(SIMULATION_TITLES[simulation])
        axis.set_xticks(x)
        axis.set_xticklabels(scenarios["scenario_label"], rotation=45, ha="right", fontsize=8)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", linestyle="--", linewidth=0.75, alpha=0.35)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(title)
    fig.supylabel(y_col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_faceted_operating_point_plot(
        summary_df: pd.DataFrame,
        level: str,
        method_order: list[str],
        filename: str,
        title: str,
) -> None:
    metric_specs = [
        ("matched_sensitivity", "matched_sensitivity_ci_lower", "matched_sensitivity_ci_upper", "Sensitivity"),
        ("matched_specificity", "matched_specificity_ci_lower", "matched_specificity_ci_upper", "Specificity"),
        ("matched_precision", "matched_precision_ci_lower", "matched_precision_ci_upper", "Precision"),
    ]
    subset = summary_df[summary_df["level"] == level].copy()
    fig, axes = plt.subplots(len(metric_specs), len(SIMULATION_ORDER), figsize=(22, 11), sharey="row")

    for col_idx, simulation in enumerate(SIMULATION_ORDER):
        sim_df = subset[subset["simulation"] == simulation].copy()
        scenarios = (
            sim_df[["scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3"]]
            .drop_duplicates()
            .sort_values(["sort_1", "sort_2", "sort_3", "scenario_label"])
            .reset_index(drop=True)
        )
        x = np.arange(len(scenarios), dtype=float)

        for row_idx, (value_col, lower_col, upper_col, ylabel) in enumerate(metric_specs):
            axis = axes[row_idx, col_idx]
            if sim_df.empty:
                axis.set_visible(False)
                continue

            for method in method_order:
                method_df = sim_df[sim_df["method"] == method].set_index("scenario_key")
                if method_df.empty:
                    continue
                values = [method_df.loc[key, value_col] if key in method_df.index else np.nan for key in scenarios["scenario_key"]]
                lower = [
                    max(values[idx] - method_df.loc[key, lower_col], 0.0) if key in method_df.index and np.isfinite(values[idx]) else 0.0
                    for idx, key in enumerate(scenarios["scenario_key"])
                ]
                upper = [
                    max(method_df.loc[key, upper_col] - values[idx], 0.0) if key in method_df.index and np.isfinite(values[idx]) else 0.0
                    for idx, key in enumerate(scenarios["scenario_key"])
                ]
                axis.errorbar(
                    x,
                    values,
                    yerr=[lower, upper],
                    fmt="o-",
                    linewidth=1.75,
                    markersize=4,
                    capsize=3,
                    color=METHOD_COLORS[method],
                    label=method,
                )

            if row_idx == 0:
                axis.set_title(SIMULATION_TITLES[simulation])
            if col_idx == 0:
                axis.set_ylabel(ylabel)
            axis.set_xticks(x)
            axis.set_xticklabels(scenarios["scenario_label"], rotation=45, ha="right", fontsize=8)
            axis.set_ylim(0.0, 1.0)
            axis.grid(axis="y", linestyle="--", linewidth=0.75, alpha=0.35)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncols=len(handles))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_faceted_delong_plot(
        delong_df: pd.DataFrame,
        filename: str,
        title: str,
) -> None:
    if delong_df.empty:
        return

    frame = prepare_delong_display_table(delong_df, ["simulation", "scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3"]).copy()
    frame["comparison"] = pd.Categorical(frame["comparison"], categories=DELONG_COMPARISON_ORDER, ordered=True)
    fig, axes = plt.subplots(2, 2, figsize=(20, 9), constrained_layout=True)
    axes = axes.flatten()

    for axis, simulation in zip(axes, SIMULATION_ORDER, strict=False):
        sim_df = frame[frame["simulation"] == simulation].copy()
        if sim_df.empty:
            axis.set_visible(False)
            continue

        scenarios = (
            sim_df[["scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3"]]
            .drop_duplicates()
            .sort_values(["sort_1", "sort_2", "sort_3", "scenario_label"])
        )
        sim_df["scenario_label"] = pd.Categorical(
            sim_df["scenario_label"],
            categories=scenarios["scenario_label"].tolist(),
            ordered=True,
        )
        sim_df = sim_df.sort_values(["comparison", "scenario_label"]).reset_index(drop=True)
        sim_df["annotation"] = sim_df.apply(
            lambda row: f"{row['delta_auroc']:+.2f}\nq={row['pvalue_bh']:.1e}",
            axis=1,
        )

        heatmap = sim_df.pivot(index="comparison", columns="scenario_label", values="delta_auroc")
        annot = sim_df.pivot(index="comparison", columns="scenario_label", values="annotation")

        sns.heatmap(
            heatmap,
            annot=annot,
            fmt="",
            cmap="RdBu_r",
            center=0.0,
            linewidths=0.5,
            linecolor="white",
            cbar=(simulation == SIMULATION_ORDER[-1]),
            cbar_kws={"label": "Delta AUROC (method A - method B)"},
            ax=axis,
        )
        axis.set_title(SIMULATION_TITLES[simulation])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=45, labelsize=8)
        axis.tick_params(axis="y", labelsize=9)

    fig.suptitle(title)
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def build_simulation_identification(result_index: pd.DataFrame, n_bootstrap: int, n_workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    identification_index = result_index[result_index["method"].isin(IDENTIFICATION_METHOD_ORDER)].copy()
    tasks = [
        {
            "metadata": metadata,
            "n_bootstrap": n_bootstrap,
        }
        for metadata in identification_index.to_dict("records")
    ]
    results = _run_parallel_jobs(tasks, _simulation_identification_task, n_workers, "Simulation identification jobs", "job")
    summary_rows = [summary_row for summary_row, _ in results]
    score_frames = [score_frame for _, score_frame in results]
    return pd.DataFrame(summary_rows), pd.concat(score_frames, ignore_index=True)


def build_simulation_grouping(result_index: pd.DataFrame, n_bootstrap: int, n_workers: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouping_index = result_index[result_index["method"].isin(GROUPING_METHOD_ORDER)].copy()
    tasks = [
        {
            "metadata": metadata,
            "n_bootstrap": n_bootstrap,
        }
        for metadata in grouping_index.to_dict("records")
    ]
    results = _run_parallel_jobs(tasks, _simulation_grouping_task, n_workers, "Simulation grouping jobs", "job")
    summary_rows = [summary_row for summary_row, _, _ in results]
    curve_rows = [curve_df for _, curve_df, _ in results]
    activation_rows = [activation_df for _, _, activation_df in results]
    summary_df = pd.DataFrame(summary_rows)
    curve_df = pd.concat(curve_rows, ignore_index=True)
    activation_df = pd.concat(activation_rows, ignore_index=True)
    return summary_df, curve_df, activation_df


def main(save_figures: bool = False, n_bootstrap: int = N_BOOTSTRAP, n_permutations: int = N_PERMUTATIONS, n_workers: int | None = None) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    start_time = time.time()
    worker_count = resolve_worker_count(n_workers)
    result_index = collect_result_index()
    total_steps = 19 if save_figures else 11
    pipeline_bar = tqdm(total=total_steps, desc="Simulation classification pipeline", unit="step")

    print(f"Using {worker_count} worker(s)")

    id_summary, id_scores = build_simulation_identification(result_index, n_bootstrap, worker_count)
    pipeline_bar.update(1)
    grp_summary, grp_curves, grp_scores = build_simulation_grouping(result_index, n_bootstrap, worker_count)
    pipeline_bar.update(1)
    id_operating = operating_point_table(
        id_summary,
        ["simulation", "level", "scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3", "method"],
    )
    pipeline_bar.update(1)
    grp_operating = operating_point_table(
        grp_summary,
        ["simulation", "level", "scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3", "method"],
    )
    pipeline_bar.update(1)
    id_delong = compute_delong_table(
        id_scores,
        group_cols=["simulation", "scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3"],
        method_pairs=DEFAULT_IDENTIFICATION_METHOD_PAIRS,
        n_workers=worker_count,
        progress_desc="Simulation identification DeLong comparisons",
    )
    pipeline_bar.update(1)
    grp_delong = compute_delong_table(
        grp_scores,
        group_cols=["simulation", "scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3"],
        method_pairs=DEFAULT_GROUPING_METHOD_PAIRS,
        n_workers=worker_count,
        progress_desc="Simulation grouping DeLong comparisons",
    )
    pipeline_bar.update(1)
    id_best_mcc_tests = compute_paired_metric_table(
        id_scores,
        group_cols=["simulation", "scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3"],
        method_pairs=DEFAULT_IDENTIFICATION_METHOD_PAIRS,
        metric_name="best_mcc",
        metric_func=_simulation_best_mcc_metric,
        n_permutations=n_permutations,
        progress_desc="Simulation identification best MCC permutations",
        n_workers=worker_count,
    )
    pipeline_bar.update(1)
    grp_best_mcc_tests = compute_paired_metric_table(
        grp_scores,
        group_cols=["simulation", "scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3"],
        method_pairs=DEFAULT_GROUPING_METHOD_PAIRS,
        metric_name="best_mcc",
        metric_func=_simulation_best_mcc_metric,
        n_permutations=n_permutations,
        progress_desc="Simulation grouping best MCC permutations",
        n_workers=worker_count,
    )
    pipeline_bar.update(1)
    id_matched_sens_tests = compute_paired_metric_table(
        id_scores,
        group_cols=["simulation", "scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3"],
        method_pairs=DEFAULT_IDENTIFICATION_METHOD_PAIRS,
        metric_name="matched_sensitivity",
        metric_func=_simulation_matched_sensitivity_metric,
        n_permutations=n_permutations,
        progress_desc="Simulation identification matched-sensitivity permutations",
        n_workers=worker_count,
    )
    pipeline_bar.update(1)
    grp_matched_sens_tests = compute_paired_metric_table(
        grp_scores,
        group_cols=["simulation", "scenario_key", "scenario_label", "sort_1", "sort_2", "sort_3"],
        method_pairs=DEFAULT_GROUPING_METHOD_PAIRS,
        metric_name="matched_sensitivity",
        metric_func=_simulation_matched_sensitivity_metric,
        n_permutations=n_permutations,
        progress_desc="Simulation grouping matched-sensitivity permutations",
        n_workers=worker_count,
    )
    pipeline_bar.update(1)

    id_summary.to_csv(OUTPUT_DIR / "identification_summary.csv", index=False)
    grp_summary.to_csv(OUTPUT_DIR / "grouping_summary.csv", index=False)
    id_operating.to_csv(OUTPUT_DIR / "identification_operating_point_table.csv", index=False)
    grp_operating.to_csv(OUTPUT_DIR / "grouping_operating_point_table.csv", index=False)
    grp_curves.to_csv(OUTPUT_DIR / "grouping_threshold_curves.csv", index=False)
    id_delong.to_csv(OUTPUT_DIR / "identification_delong.csv", index=False)
    grp_delong.to_csv(OUTPUT_DIR / "grouping_delong.csv", index=False)
    id_best_mcc_tests.to_csv(OUTPUT_DIR / "identification_best_mcc_tests.csv", index=False)
    grp_best_mcc_tests.to_csv(OUTPUT_DIR / "grouping_best_mcc_tests.csv", index=False)
    id_matched_sens_tests.to_csv(OUTPUT_DIR / "identification_matched_sensitivity_tests.csv", index=False)
    grp_matched_sens_tests.to_csv(OUTPUT_DIR / "grouping_matched_sensitivity_tests.csv", index=False)
    pipeline_bar.update(1)

    if save_figures:
        save_faceted_summary_plot(
            summary_df=id_summary,
            level="identification",
            method_order=IDENTIFICATION_METHOD_ORDER,
            y_col="auroc",
            lower_col="auroc_ci_lower",
            upper_col="auroc_ci_upper",
            filename="simulation_identification_auroc_ci.png",
            title="Simulation Identification AUROC with 95% Bootstrap CI",
        )
        pipeline_bar.update(1)
        save_faceted_summary_plot(
            summary_df=id_summary,
            level="identification",
            method_order=IDENTIFICATION_METHOD_ORDER,
            y_col="best_mcc",
            lower_col="best_mcc_ci_lower",
            upper_col="best_mcc_ci_upper",
            filename="simulation_identification_best_mcc_ci.png",
            title="Simulation Identification Best MCC with 95% Bootstrap CI",
        )
        pipeline_bar.update(1)
        save_faceted_summary_plot(
            summary_df=grp_summary,
            level="grouping",
            method_order=GROUPING_METHOD_ORDER,
            y_col="auroc",
            lower_col="auroc_ci_lower",
            upper_col="auroc_ci_upper",
            filename="simulation_grouping_auroc_ci.png",
            title="Simulation Grouping AUROC with 95% Bootstrap CI",
        )
        pipeline_bar.update(1)
        save_faceted_summary_plot(
            summary_df=grp_summary,
            level="grouping",
            method_order=GROUPING_METHOD_ORDER,
            y_col="best_mcc",
            lower_col="best_mcc_ci_lower",
            upper_col="best_mcc_ci_upper",
            filename="simulation_grouping_best_mcc_ci.png",
            title="Simulation Grouping Best MCC with 95% Bootstrap CI",
        )
        pipeline_bar.update(1)
        save_faceted_operating_point_plot(
            summary_df=id_summary,
            level="identification",
            method_order=IDENTIFICATION_METHOD_ORDER,
            filename="simulation_identification_matched_fpr_ci.png",
            title="Simulation Identification Operating Point at FPR = 5%",
        )
        pipeline_bar.update(1)
        save_faceted_operating_point_plot(
            summary_df=grp_summary,
            level="grouping",
            method_order=GROUPING_METHOD_ORDER,
            filename="simulation_grouping_matched_fpr_ci.png",
            title="Simulation Grouping Operating Point at FPR = 5%",
        )
        pipeline_bar.update(1)
    if save_figures:
        save_faceted_delong_plot(
            delong_df=id_delong,
            filename="simulation_identification_delong_heatmap.png",
            title="Simulation Identification Paired DeLong Comparisons",
        )
        pipeline_bar.update(1)
        save_faceted_delong_plot(
            delong_df=grp_delong,
            filename="simulation_grouping_delong_heatmap.png",
            title="Simulation Grouping Paired DeLong Comparisons",
        )
        pipeline_bar.update(1)

    pipeline_bar.close()

    print("Saved simulation classification summaries to", OUTPUT_DIR)
    print(f"Runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation classification stats exporter")
    parser.add_argument("--save-figures", action="store_true", help="Save figure outputs in addition to CSVs")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help="Number of bootstrap samples for CI calculations")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS, help="Paired permutation resamples for non-AUROC tests")
    parser.add_argument("--n-workers", type=int, default=1, help="Maximum worker processes to use (capped at 28)")
    args = parser.parse_args()
    main(
        save_figures=args.save_figures,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        n_workers=args.n_workers,
    )
