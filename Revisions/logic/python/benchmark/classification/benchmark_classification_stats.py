#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import time
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
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

from revisionlib.paths import BENCHMARK_RESULTS_ROOT, ensure_directory, revision_benchmark_output_dir
from revisionlib.performance_stats import (
    DEFAULT_GROUPING_METHOD_PAIRS,
    DEFAULT_IDENTIFICATION_METHOD_PAIRS,
    FIXED_THRESHOLD,
    MATCHED_FPR,
    STANDARD_THRESHOLDS,
    best_mcc_statistic,
    build_identification_score_frame,
    combine_item_key,
    compute_paired_metric_table,
    compute_delong_table,
    grouping_calls_copf,
    grouping_calls_proteoforge,
    grouping_curve_and_activation,
    matched_sensitivity_statistic,
    normalize_peptide_identifier,
    operating_point_table,
    prepare_delong_display_table,
    resolve_worker_count,
    summarize_binary_scores,
)


OUTPUT_DIR = revision_benchmark_output_dir("classification_stats")
FIGURE_DIR = ensure_directory(OUTPUT_DIR / "figures")
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000

IDENTIFICATION_ORDER = ["1pep", "2pep", "random", "050pep"]
IDENTIFICATION_LABELS = {
    "1pep": "1 Peptide",
    "2pep": "2 Peptides",
    "random": "Random (1 to 50%)",
    "050pep": "50% Peptides",
}
GROUPING_ORDER = ["2pep", "random", "050pep"]
GROUPING_LABELS = {
    "2pep": "2 Peptides",
    "random": "Random (2 to 50%)",
    "050pep": "50% Peptides",
}
METHOD_COLORS = {
    "COPF": "#139593",
    "PeCorA": "#fca311",
    "ProteoForge": "#e54f2a",
}
METHOD_ORDER = ["ProteoForge", "COPF", "PeCorA"]
DELONG_COMPARISON_ORDER = ["ProteoForge vs PeCorA", "ProteoForge vs COPF", "PeCorA vs COPF"]


def _benchmark_best_mcc_metric(y_true: np.ndarray, y_pval: np.ndarray) -> float:
    return best_mcc_statistic(y_true, y_pval, thresholds=STANDARD_THRESHOLDS)


def _benchmark_matched_sensitivity_metric(y_true: np.ndarray, y_pval: np.ndarray) -> float:
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


def _benchmark_identification_task(task: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame]:
    data_id = str(task["scenario"])
    method = str(task["method"])
    pvalue_col = str(task["pvalue_col"])
    item_col = str(task["item_col"])

    path = BENCHMARK_RESULTS_ROOT / f"{method}_{data_id}_result.feather"
    data = pd.read_feather(path)

    if method == "ProteoForge":
        data = data[["protein_id", "peptide_id", "perturbed_peptide", pvalue_col]].drop_duplicates().copy()
    elif method == "PeCorA":
        data = data[["protein", item_col, "perturbed_peptide", pvalue_col]].drop_duplicates().copy()
    else:
        data = data[["protein_id", item_col, "perturbed_peptide", pvalue_col]].drop_duplicates().copy()

    y_true = data["perturbed_peptide"].astype(bool).to_numpy()
    y_pval = data[pvalue_col].astype(float).to_numpy()
    summary = summarize_binary_scores(
        y_true,
        y_pval,
        thresholds=STANDARD_THRESHOLDS,
        fixed_threshold=FIXED_THRESHOLD,
        n_bootstrap=int(task["n_bootstrap"]),
    )
    summary_row = {
        "level": "identification",
        "scenario": data_id,
        "method": method,
        **summary,
    }

    if method == "ProteoForge":
        item_ids = data.apply(
            lambda row: combine_item_key(row["protein_id"], normalize_peptide_identifier(row["peptide_id"])),
            axis=1,
        )
    elif method == "PeCorA":
        item_ids = data.apply(
            lambda row: combine_item_key(row["protein"], normalize_peptide_identifier(row["peptide_id"])),
            axis=1,
        )
    else:
        item_ids = data.apply(
            lambda row: combine_item_key(row["protein_id"], normalize_peptide_identifier(row["id"])),
            axis=1,
        )

    score_frame = build_identification_score_frame(
        item_ids=item_ids,
        truth_labels=y_true,
        pvalues=y_pval,
        method=method,
        metadata={
            "level": "identification",
            "scenario": data_id,
        },
    )
    return summary_row, score_frame


def _benchmark_grouping_task(task: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    data_id = str(task["scenario"])
    method = str(task["method"])

    path = BENCHMARK_RESULTS_ROOT / f"{method}_{data_id}_result.feather"
    data = pd.read_feather(path)

    if method == "COPF":
        data = data[[
            "protein_id", "id", "perturbed_protein", "proteoform_score_pval", "proteoform_score", "cluster",
        ]].drop_duplicates().copy()
        curve_df, activation_df = grouping_curve_and_activation(
            data=data,
            thresholds=STANDARD_THRESHOLDS,
            builder=lambda frame, thr: grouping_calls_copf(
                frame,
                thr,
                protein_col="protein_id",
                truth_col="perturbed_protein",
                pvalue_col="proteoform_score_pval",
                score_col="proteoform_score",
                cluster_col="cluster",
            ),
            protein_col="protein_id",
            truth_col="perturbed_protein",
        )
    else:
        data = data[[
            "protein_id", "peptide_id", "perturbed_protein", "perturbed_peptide", "adj_pval", "ClusterID",
        ]].drop_duplicates().copy()
        curve_df, activation_df = grouping_curve_and_activation(
            data=data,
            thresholds=STANDARD_THRESHOLDS,
            builder=lambda frame, thr: grouping_calls_proteoforge(
                frame,
                thr,
                protein_col="protein_id",
                peptide_col="peptide_id",
                truth_col="perturbed_protein",
                pvalue_col="adj_pval",
                cluster_col="ClusterID",
            ),
            protein_col="protein_id",
            truth_col="perturbed_protein",
        )

    summary = summarize_binary_scores(
        activation_df["true_label"].to_numpy(),
        activation_df["score_pvalue"].to_numpy(),
        thresholds=STANDARD_THRESHOLDS,
        fixed_threshold=FIXED_THRESHOLD,
        n_bootstrap=int(task["n_bootstrap"]),
    )
    summary_row = {
        "level": "grouping",
        "scenario": data_id,
        "method": method,
        **summary,
    }

    curve_df = curve_df.copy()
    curve_df["level"] = "grouping"
    curve_df["scenario"] = data_id
    curve_df["method"] = method

    activation_df = activation_df.rename(columns={"protein_id": "item_id"}).copy()
    activation_df["level"] = "grouping"
    activation_df["scenario"] = data_id
    activation_df["method"] = method
    activation_df = activation_df[["item_id", "true_label", "score_pvalue", "level", "scenario", "method"]].copy()
    return summary_row, curve_df, activation_df


def save_summary_plot(
        summary_df: pd.DataFrame,
        x_col: str,
        x_order: list[str],
        x_labels: dict[str, str],
        y_col: str,
        lower_col: str,
        upper_col: str,
        filename: str,
        title: str,
) -> None:
    methods = [method for method in METHOD_ORDER if method in summary_df["method"].unique()]
    x = np.arange(len(x_order), dtype=float)
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, method in enumerate(methods):
        method_df = summary_df[summary_df["method"] == method].set_index(x_col)
        values = [method_df.loc[item, y_col] if item in method_df.index else np.nan for item in x_order]
        lower = [
            max(values[i] - method_df.loc[item, lower_col], 0.0) if item in method_df.index and np.isfinite(values[i]) else 0.0
            for i, item in enumerate(x_order)
        ]
        upper = [
            max(method_df.loc[item, upper_col] - values[i], 0.0) if item in method_df.index and np.isfinite(values[i]) else 0.0
            for i, item in enumerate(x_order)
        ]
        ax.bar(
            x + idx * width,
            values,
            width,
            yerr=[lower, upper],
            capsize=3,
            color=METHOD_COLORS.get(method, "#999999"),
            edgecolor="black",
            linewidth=0.5,
            label=method,
        )

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([x_labels[item] for item in x_order], rotation=20, ha="right")
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.75, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_operating_point_plot(
        summary_df: pd.DataFrame,
        x_col: str,
        x_order: list[str],
        x_labels: dict[str, str],
        filename: str,
        title: str,
) -> None:
    metric_specs = [
        ("matched_sensitivity", "matched_sensitivity_ci_lower", "matched_sensitivity_ci_upper", "Sensitivity"),
        ("matched_specificity", "matched_specificity_ci_lower", "matched_specificity_ci_upper", "Specificity"),
        ("matched_precision", "matched_precision_ci_lower", "matched_precision_ci_upper", "Precision"),
    ]
    methods = [method for method in METHOD_ORDER if method in summary_df["method"].unique()]
    x = np.arange(len(x_order), dtype=float)
    width = 0.8 / max(1, len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for axis, (value_col, lower_col, upper_col, ylabel) in zip(axes, metric_specs, strict=False):
        for idx, method in enumerate(methods):
            method_df = summary_df[summary_df["method"] == method].set_index(x_col)
            values = [method_df.loc[item, value_col] if item in method_df.index else np.nan for item in x_order]
            lower = [
                max(values[i] - method_df.loc[item, lower_col], 0.0) if item in method_df.index and np.isfinite(values[i]) else 0.0
                for i, item in enumerate(x_order)
            ]
            upper = [
                max(method_df.loc[item, upper_col] - values[i], 0.0) if item in method_df.index and np.isfinite(values[i]) else 0.0
                for i, item in enumerate(x_order)
            ]
            axis.bar(
                x + idx * width,
                values,
                width,
                yerr=[lower, upper],
                capsize=3,
                color=METHOD_COLORS.get(method, "#999999"),
                edgecolor="black",
                linewidth=0.5,
                label=method,
            )

        axis.set_xticks(x + width * (len(methods) - 1) / 2)
        axis.set_xticklabels([x_labels[item] for item in x_order], rotation=20, ha="right")
        axis.set_title(ylabel)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", linestyle="--", linewidth=0.75, alpha=0.4)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel(f"Estimate at FPR = {MATCHED_FPR:.0%}")
    axes[0].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_delong_heatmap(
        delong_df: pd.DataFrame,
        scenario_order: list[str],
        scenario_labels: dict[str, str],
        filename: str,
        title: str,
) -> None:
    if delong_df.empty:
        return

    frame = prepare_delong_display_table(delong_df, ["scenario"]).copy()
    frame["comparison"] = pd.Categorical(frame["comparison"], categories=DELONG_COMPARISON_ORDER, ordered=True)
    frame["scenario"] = pd.Categorical(frame["scenario"], categories=scenario_order, ordered=True)
    frame = frame.sort_values(["comparison", "scenario"]).reset_index(drop=True)
    frame["Scenario"] = frame["scenario"].map(scenario_labels)
    frame["annotation"] = frame.apply(
        lambda row: f"{row['delta_auroc']:+.3f}\nq={row['pvalue_bh']:.2e}",
        axis=1,
    )

    heatmap = frame.pivot(index="comparison", columns="Scenario", values="delta_auroc")
    annot = frame.pivot(index="comparison", columns="Scenario", values="annotation")

    fig, ax = plt.subplots(figsize=(1.8 * len(heatmap.columns) + 1.5, 3.4))
    sns.heatmap(
        heatmap,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0.0,
        linewidths=0.75,
        linecolor="white",
        cbar_kws={"label": "Delta AUROC (method A - method B)"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def build_benchmark_identification(n_bootstrap: int, n_workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [
        {
            "scenario": data_id,
            "method": method,
            "pvalue_col": pvalue_col,
            "item_col": item_col,
            "n_bootstrap": n_bootstrap,
        }
        for data_id in IDENTIFICATION_ORDER
        for method, pvalue_col, item_col in [
            ("COPF", "proteoform_score_pval", "id"),
            ("PeCorA", "adj_pval", "peptide_id"),
            ("ProteoForge", "adj_pval", "peptide_id"),
        ]
    ]
    results = _run_parallel_jobs(tasks, _benchmark_identification_task, n_workers, "Benchmark identification jobs", "job")
    summary_rows = [summary_row for summary_row, _ in results]
    score_frames = [score_frame for _, score_frame in results]
    return pd.DataFrame(summary_rows), pd.concat(score_frames, ignore_index=True)


def build_benchmark_grouping(n_bootstrap: int, n_workers: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = [
        {
            "scenario": data_id,
            "method": method,
            "n_bootstrap": n_bootstrap,
        }
        for data_id in GROUPING_ORDER
        for method in ["COPF", "ProteoForge"]
    ]
    results = _run_parallel_jobs(tasks, _benchmark_grouping_task, n_workers, "Benchmark grouping jobs", "job")
    summary_rows = [summary_row for summary_row, _, _ in results]
    curve_rows = [curve_df for _, curve_df, _ in results]
    activation_rows = [activation_df for _, _, activation_df in results]
    summary_df = pd.DataFrame(summary_rows)
    curve_df = pd.concat(curve_rows, ignore_index=True)
    activation_df = pd.concat(activation_rows, ignore_index=True)
    return summary_df, curve_df, activation_df


def main(save_figures: bool = False, n_permutations: int = N_PERMUTATIONS, n_bootstrap: int = N_BOOTSTRAP, n_workers: int | None = None) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    start_time = time.time()
    worker_count = resolve_worker_count(n_workers)
    total_steps = 19 if save_figures else 17
    pipeline_bar = tqdm(total=total_steps, desc="Benchmark classification pipeline", unit="step")

    print(f"Using {worker_count} worker(s)")

    id_summary, id_scores = build_benchmark_identification(n_bootstrap, worker_count)
    pipeline_bar.update(1)

    grp_summary, grp_curves, grp_scores = build_benchmark_grouping(n_bootstrap, worker_count)
    pipeline_bar.update(1)

    id_operating = operating_point_table(id_summary, ["level", "scenario", "method"])
    pipeline_bar.update(1)
    grp_operating = operating_point_table(grp_summary, ["level", "scenario", "method"])
    pipeline_bar.update(1)

    id_delong = compute_delong_table(
        id_scores,
        group_cols=["scenario"],
        method_pairs=DEFAULT_IDENTIFICATION_METHOD_PAIRS,
        n_workers=worker_count,
        progress_desc="Identification DeLong comparisons",
    )
    pipeline_bar.update(1)
    grp_delong = compute_delong_table(
        grp_scores,
        group_cols=["scenario"],
        method_pairs=DEFAULT_GROUPING_METHOD_PAIRS,
        n_workers=worker_count,
        progress_desc="Grouping DeLong comparisons",
    )
    pipeline_bar.update(1)

    id_best_mcc_tests = compute_paired_metric_table(
        id_scores,
        group_cols=["scenario"],
        method_pairs=DEFAULT_IDENTIFICATION_METHOD_PAIRS,
        metric_name="best_mcc",
        metric_func=_benchmark_best_mcc_metric,
        n_permutations=n_permutations,
        progress_desc="Identification best MCC permutations",
        n_workers=worker_count,
    )
    pipeline_bar.update(1)
    grp_best_mcc_tests = compute_paired_metric_table(
        grp_scores,
        group_cols=["scenario"],
        method_pairs=DEFAULT_GROUPING_METHOD_PAIRS,
        metric_name="best_mcc",
        metric_func=_benchmark_best_mcc_metric,
        n_permutations=n_permutations,
        progress_desc="Grouping best MCC permutations",
        n_workers=worker_count,
    )
    pipeline_bar.update(1)
    id_matched_sens_tests = compute_paired_metric_table(
        id_scores,
        group_cols=["scenario"],
        method_pairs=DEFAULT_IDENTIFICATION_METHOD_PAIRS,
        metric_name="matched_sensitivity",
        metric_func=_benchmark_matched_sensitivity_metric,
        n_permutations=n_permutations,
        progress_desc="Identification matched-sensitivity permutations",
        n_workers=worker_count,
    )
    pipeline_bar.update(1)
    grp_matched_sens_tests = compute_paired_metric_table(
        grp_scores,
        group_cols=["scenario"],
        method_pairs=DEFAULT_GROUPING_METHOD_PAIRS,
        metric_name="matched_sensitivity",
        metric_func=_benchmark_matched_sensitivity_metric,
        n_permutations=n_permutations,
        progress_desc="Grouping matched-sensitivity permutations",
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

    save_summary_plot(
        summary_df=id_summary,
        x_col="scenario",
        x_order=IDENTIFICATION_ORDER,
        x_labels=IDENTIFICATION_LABELS,
        y_col="auroc",
        lower_col="auroc_ci_lower",
        upper_col="auroc_ci_upper",
        filename="identification_auroc_ci.png",
        title="Benchmark Identification AUROC with 95% Bootstrap CI",
    )
    pipeline_bar.update(1)
    save_summary_plot(
        summary_df=id_summary,
        x_col="scenario",
        x_order=IDENTIFICATION_ORDER,
        x_labels=IDENTIFICATION_LABELS,
        y_col="best_mcc",
        lower_col="best_mcc_ci_lower",
        upper_col="best_mcc_ci_upper",
        filename="identification_best_mcc_ci.png",
        title="Benchmark Identification Best MCC with 95% Bootstrap CI",
    )
    pipeline_bar.update(1)
    save_summary_plot(
        summary_df=grp_summary,
        x_col="scenario",
        x_order=GROUPING_ORDER,
        x_labels=GROUPING_LABELS,
        y_col="auroc",
        lower_col="auroc_ci_lower",
        upper_col="auroc_ci_upper",
        filename="grouping_auroc_ci.png",
        title="Benchmark Grouping AUROC with 95% Bootstrap CI",
    )
    pipeline_bar.update(1)
    save_summary_plot(
        summary_df=grp_summary,
        x_col="scenario",
        x_order=GROUPING_ORDER,
        x_labels=GROUPING_LABELS,
        y_col="best_mcc",
        lower_col="best_mcc_ci_lower",
        upper_col="best_mcc_ci_upper",
        filename="grouping_best_mcc_ci.png",
        title="Benchmark Grouping Best MCC with 95% Bootstrap CI",
    )
    pipeline_bar.update(1)
    save_operating_point_plot(
        summary_df=id_summary,
        x_col="scenario",
        x_order=IDENTIFICATION_ORDER,
        x_labels=IDENTIFICATION_LABELS,
        filename="identification_matched_fpr_ci.png",
        title="Benchmark Identification Operating Point at FPR = 5%",
    )
    pipeline_bar.update(1)
    save_operating_point_plot(
        summary_df=grp_summary,
        x_col="scenario",
        x_order=GROUPING_ORDER,
        x_labels=GROUPING_LABELS,
        filename="grouping_matched_fpr_ci.png",
        title="Benchmark Grouping Operating Point at FPR = 5%",
    )
    pipeline_bar.update(1)
    if save_figures:
        save_delong_heatmap(
            delong_df=id_delong,
            scenario_order=IDENTIFICATION_ORDER,
            scenario_labels=IDENTIFICATION_LABELS,
            filename="identification_delong_heatmap.png",
            title="Benchmark Identification Paired DeLong Comparisons",
        )
        pipeline_bar.update(1)
        save_delong_heatmap(
            delong_df=grp_delong,
            scenario_order=GROUPING_ORDER,
            scenario_labels=GROUPING_LABELS,
            filename="grouping_delong_heatmap.png",
            title="Benchmark Grouping Paired DeLong Comparisons",
        )
        pipeline_bar.update(1)

    pipeline_bar.close()

    print("Saved benchmark classification summaries to", OUTPUT_DIR)
    print(f"Runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark classification stats exporter")
    parser.add_argument("--save-figures", action="store_true", help="Save figure outputs in addition to CSVs")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help="Number of bootstrap samples for CI calculations")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS, help="Paired permutation resamples for non-AUROC tests")
    parser.add_argument("--n-workers", type=int, default=1, help="Maximum worker processes to use (capped at 28)")
    args = parser.parse_args()
    main(
        save_figures=args.save_figures,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
        n_workers=args.n_workers,
    )

