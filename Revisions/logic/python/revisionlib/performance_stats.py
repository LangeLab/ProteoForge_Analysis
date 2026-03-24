from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[4]
for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src import utils


STANDARD_THRESHOLDS = list(utils.generate_thresholds(10.0, -15, 1, 0, 1, 0.1))
FIXED_THRESHOLD = 1e-3
MATCHED_FPR = 0.05
NEVER_POSITIVE_PVALUE = 1.1
MAX_PARALLEL_WORKERS = 28
DEFAULT_IDENTIFICATION_METHOD_PAIRS = [
    ("ProteoForge", "PeCorA"),
    ("ProteoForge", "COPF"),
    ("PeCorA", "COPF"),
]
DEFAULT_GROUPING_METHOD_PAIRS = [
    ("ProteoForge", "COPF"),
]
PEPTIDE_MODIFICATION_PATTERN = re.compile(r"\(UniMod:\d+\)")
PEPTIDE_INTEGER_PATTERN = re.compile(r"(?:^|[-_])(?:pep_)?(\d+)$", re.IGNORECASE)


def resolve_worker_count(n_workers: int | None) -> int:
    if n_workers is None:
        return 1

    available = os.cpu_count() or 1
    requested = max(1, int(n_workers))
    return min(requested, available, MAX_PARALLEL_WORKERS)


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)

    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return adjusted

    finite_values = values[finite_mask]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    count = ranked.size

    scaled = ranked * count / np.arange(1, count + 1, dtype=float)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)

    restored = np.empty_like(monotone)
    restored[order] = monotone
    adjusted[finite_mask] = restored
    return adjusted


def bonferroni_adjust(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)

    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return adjusted

    count = int(finite_mask.sum())
    adjusted[finite_mask] = np.clip(values[finite_mask] * count, 0.0, 1.0)
    return adjusted


def normalize_peptide_identifier(value: object) -> str:
    text = str(value).strip()
    text = text.replace("_all", "")
    text = PEPTIDE_MODIFICATION_PATTERN.sub("", text)
    return text


def peptide_position_identifier(value: object) -> str:
    text = normalize_peptide_identifier(value)
    match = PEPTIDE_INTEGER_PATTERN.search(text)
    return match.group(1) if match else text


def combine_item_key(*parts: object) -> str:
    return "::".join(str(part) for part in parts)


def build_identification_score_frame(
        item_ids: Iterable[object],
        truth_labels: Iterable[bool] | np.ndarray,
        pvalues: Iterable[float] | np.ndarray,
        method: str,
        metadata: dict[str, object],
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "item_id": list(item_ids),
        "true_label": np.asarray(truth_labels, dtype=bool),
        "score_pvalue": np.asarray(pvalues, dtype=float),
    })
    frame["method"] = method
    for key, value in metadata.items():
        frame[key] = value
    return frame.drop_duplicates(subset=["item_id", "true_label", "method"]).reset_index(drop=True)


def prepare_delong_display_table(
        delong_df: pd.DataFrame,
        group_cols: Sequence[str],
) -> pd.DataFrame:
    if delong_df.empty:
        return delong_df.copy()

    frame = delong_df.copy()
    frame["comparison"] = frame["method_a"] + " vs " + frame["method_b"]
    for column in ["auroc_a", "auroc_b", "delta_auroc", "z_statistic", "pvalue", "pvalue_bh"]:
        if column in frame.columns:
            frame[column] = frame[column].astype(float)
    return frame[[
        *group_cols,
        "comparison",
        "method_a",
        "method_b",
        "n_items",
        "n_positive",
        "n_negative",
        "auroc_a",
        "auroc_b",
        "delta_auroc",
        "z_statistic",
        "pvalue",
        "pvalue_bh",
    ]].copy()


def summarize_binary_scores(
        y_true: Iterable[bool] | np.ndarray,
        y_pval: Iterable[float] | np.ndarray,
        thresholds: Sequence[float] | None = None,
        alpha: float = 0.05,
        fixed_threshold: float = FIXED_THRESHOLD,
        matched_fpr: float = MATCHED_FPR,
        seed: int = 42,
        n_bootstrap: int = 1000,
) -> dict[str, float]:
    thresholds = list(STANDARD_THRESHOLDS if thresholds is None else thresholds)
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_pval_arr = np.asarray(y_pval, dtype=float)

    auroc_ci = utils.bootstrap_auroc_ci(
        y_true_arr.astype(int),
        y_pval_arr,
        n_bootstrap=n_bootstrap,
        seed=seed,
        alpha=alpha,
    )
    max_mcc_ci = utils.bootstrap_max_mcc_ci(
        y_true_arr.astype(int),
        y_pval_arr,
        thresholds=thresholds,
        n_bootstrap=n_bootstrap,
        seed=seed,
        alpha=alpha,
    )
    fixed_mcc_ci = utils.bootstrap_mcc_ci(
        y_true_arr.astype(int),
        y_pval_arr,
        threshold=fixed_threshold,
        n_bootstrap=n_bootstrap,
        seed=seed,
        alpha=alpha,
    )
    matched_fpr_metrics = utils.bootstrap_fixed_fpr_ci(
        y_true_arr.astype(int),
        y_pval_arr,
        target_fpr=matched_fpr,
        n_bootstrap=n_bootstrap,
        seed=seed,
        alpha=alpha,
    )
    curve = utils.create_metric_data(
        pd.DataFrame({"label": y_true_arr, "pval": y_pval_arr}),
        pvalue_thresholds=thresholds,
        label_col="label",
        pvalue_col="pval",
    )
    best_idx = int(curve["MCC"].idxmax())
    best_row = curve.loc[best_idx]
    fixed_metrics = utils.calculate_metrics(
        true_labels=pd.Series(y_true_arr),
        pred_labels=pd.Series(y_pval_arr <= fixed_threshold),
        verbose=False,
        return_metrics=True,
    )
    sensitivity_ci_lower, sensitivity_ci_upper = matched_fpr_metrics["Sens_CI"]
    specificity_ci_lower, specificity_ci_upper = matched_fpr_metrics["Spec_CI"]
    precision_ci_lower, precision_ci_upper = matched_fpr_metrics["Prec_CI"]

    return {
        "n_items": int(y_true_arr.size),
        "n_positive": int(y_true_arr.sum()),
        "n_negative": int((~y_true_arr).sum()),
        "auroc": float(auroc_ci["AUROC"]),
        "auroc_ci_lower": float(auroc_ci["CI_lower"]),
        "auroc_ci_upper": float(auroc_ci["CI_upper"]),
        "auroc_se": np.nan,
        "best_mcc": float(best_row["MCC"]),
        "best_mcc_threshold": float(best_row["threshold"]),
        "best_mcc_ci_lower": float(max_mcc_ci["CI_lower"]),
        "best_mcc_ci_upper": float(max_mcc_ci["CI_upper"]),
        "fixed_mcc": float(fixed_mcc_ci["MCC"]),
        "fixed_mcc_threshold": float(fixed_threshold),
        "fixed_mcc_ci_lower": float(fixed_mcc_ci["CI_lower"]),
        "fixed_mcc_ci_upper": float(fixed_mcc_ci["CI_upper"]),
        "fixed_tpr": float(fixed_metrics["TPR"]),
        "fixed_fpr": float(fixed_metrics["FPR"]),
        "matched_fpr_target": float(matched_fpr),
        "matched_sensitivity": float(matched_fpr_metrics["Sensitivity"]),
        "matched_sensitivity_ci_lower": float(sensitivity_ci_lower),
        "matched_sensitivity_ci_upper": float(sensitivity_ci_upper),
        "matched_specificity": float(matched_fpr_metrics["Specificity"]),
        "matched_specificity_ci_lower": float(specificity_ci_lower),
        "matched_specificity_ci_upper": float(specificity_ci_upper),
        "matched_precision": float(matched_fpr_metrics["Precision"]),
        "matched_precision_ci_lower": float(precision_ci_lower),
        "matched_precision_ci_upper": float(precision_ci_upper),
        "matched_actual_fpr": float(matched_fpr_metrics["ActualFPR"]),
        "matched_threshold": float(matched_fpr_metrics["Threshold"]),
    }


def operating_point_table(
        summary_df: pd.DataFrame,
        leading_columns: Sequence[str],
) -> pd.DataFrame:
    return summary_df[[
        *leading_columns,
        "matched_fpr_target",
        "matched_actual_fpr",
        "matched_threshold",
        "matched_sensitivity",
        "matched_sensitivity_ci_lower",
        "matched_sensitivity_ci_upper",
        "matched_specificity",
        "matched_specificity_ci_lower",
        "matched_specificity_ci_upper",
        "matched_precision",
        "matched_precision_ci_lower",
        "matched_precision_ci_upper",
    ]].copy()


def build_identification_curve(
        data: pd.DataFrame,
        thresholds: Sequence[float],
        label_col: str,
        pvalue_col: str,
) -> pd.DataFrame:
    return utils.create_metric_data(
        data[[label_col, pvalue_col]].copy(),
        pvalue_thresholds=thresholds,
        label_col=label_col,
        pvalue_col=pvalue_col,
    )


def _copf_grouping_assignment(
        data: pd.DataFrame,
        threshold: float,
        protein_col: str,
        cluster_col: str,
        pvalue_col: str,
        score_col: str,
        score_threshold: float | None = None,
        sep: str = "_",
) -> pd.DataFrame:
    out = data.copy()
    if score_threshold is not None:
        condition = (out[score_col] >= score_threshold) & (out[pvalue_col] <= threshold)
    else:
        condition = out[pvalue_col] <= threshold

    out["proteoform_id"] = out[protein_col]
    out.loc[condition, "proteoform_id"] = out.loc[condition].apply(
        lambda row: f"{row[protein_col]}{sep}{int(row[cluster_col])}",
        axis=1,
    )
    out.loc[out[protein_col] == out["proteoform_id"], "proteoform_id"] = out[protein_col] + f"{sep}0"
    out.loc[out[cluster_col] == 100, "proteoform_id"] = out[protein_col] + f"{sep}0"

    n_proteoforms = out.groupby(protein_col)["proteoform_id"].transform("nunique")
    has_zero_cluster = out["proteoform_id"].str.endswith(f"{sep}0")
    n_proteoforms -= has_zero_cluster.groupby(out[protein_col]).transform("sum")
    out.loc[n_proteoforms.isin([0, 1]), "proteoform_id"] = out[protein_col]
    return out


def grouping_calls_proteoforge(
        data: pd.DataFrame,
        threshold: float,
        protein_col: str,
        peptide_col: str,
        truth_col: str,
        pvalue_col: str,
        cluster_col: str,
) -> pd.DataFrame:
    out = data[[protein_col, peptide_col, truth_col, pvalue_col, cluster_col]].drop_duplicates().copy()
    out["proteoform_group"] = -1
    out.loc[out[pvalue_col] < threshold, "proteoform_group"] = out.loc[out[pvalue_col] < threshold, cluster_col]
    out["is_ptm"] = out["proteoform_group"] > 0
    out["is_group"] = out.groupby([protein_col, "proteoform_group"])[peptide_col].transform("count") > 1
    out["with_proteoform"] = (out["is_group"] & out["is_ptm"]).groupby(out[protein_col]).transform("any")
    return out[[protein_col, truth_col, "with_proteoform"]].drop_duplicates().reset_index(drop=True)


def grouping_calls_copf(
        data: pd.DataFrame,
        threshold: float,
        protein_col: str,
        truth_col: str,
        pvalue_col: str,
        score_col: str,
        cluster_col: str,
) -> pd.DataFrame:
    out = _copf_grouping_assignment(
        data=data,
        threshold=threshold,
        protein_col=protein_col,
        cluster_col=cluster_col,
        pvalue_col=pvalue_col,
        score_col=score_col,
        score_threshold=None,
    )
    out["n_proteoforms"] = out.groupby(protein_col)["proteoform_id"].transform("nunique")
    out["with_proteoform"] = out["n_proteoforms"] > 1
    return out[[protein_col, truth_col, "with_proteoform"]].drop_duplicates().reset_index(drop=True)


def grouping_curve_and_activation(
        data: pd.DataFrame,
        thresholds: Sequence[float],
        builder: Callable[[pd.DataFrame, float], pd.DataFrame],
        protein_col: str,
        truth_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds_sorted = sorted(float(threshold) for threshold in thresholds)
    metric_frames: list[pd.DataFrame] = []
    activation_lookup: dict[str, float] = {}
    truth_lookup: dict[str, bool] = (
        data[[protein_col, truth_col]]
        .drop_duplicates()
        .assign(**{truth_col: lambda frame: frame[truth_col].astype(bool)})
        .set_index(protein_col)[truth_col]
        .to_dict()
    )

    for threshold in thresholds_sorted:
        calls = builder(data, threshold)
        metrics = utils.calculate_metrics(
            true_labels=calls[truth_col].astype(bool),
            pred_labels=calls["with_proteoform"].astype(bool),
            verbose=False,
            return_metrics=True,
        )
        metrics["threshold"] = threshold
        metric_frames.append(pd.DataFrame([metrics]))

        positives = calls.loc[calls["with_proteoform"].astype(bool), protein_col].tolist()
        for protein_id in positives:
            activation_lookup.setdefault(str(protein_id), float(threshold))

    activation_rows = []
    for protein_id, truth in truth_lookup.items():
        activation_rows.append({
            protein_col: protein_id,
            "true_label": bool(truth),
            "score_pvalue": float(activation_lookup.get(str(protein_id), NEVER_POSITIVE_PVALUE)),
        })

    return pd.concat(metric_frames, ignore_index=True), pd.DataFrame(activation_rows)


def compute_delong_table(
        score_frame: pd.DataFrame,
        group_cols: Sequence[str],
        method_pairs: Sequence[tuple[str, str]],
        item_col: str = "item_id",
        truth_col: str = "true_label",
        pvalue_col: str = "score_pvalue",
        n_workers: int | None = None,
        progress_desc: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tasks: list[tuple[dict[str, object], str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    worker_count = resolve_worker_count(n_workers)

    for group_values, group_df in score_frame.groupby(list(group_cols), dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_payload = dict(zip(group_cols, group_values))

        for method_a, method_b in method_pairs:
            subset_a = group_df[group_df["method"] == method_a][[item_col, truth_col, pvalue_col]].copy()
            subset_b = group_df[group_df["method"] == method_b][[item_col, truth_col, pvalue_col]].copy()
            if subset_a.empty or subset_b.empty:
                continue

            merged = subset_a.merge(
                subset_b,
                on=[item_col, truth_col],
                how="inner",
                suffixes=("_a", "_b"),
            )
            if merged.empty:
                continue

            tasks.append((
                group_payload,
                method_a,
                method_b,
                merged[truth_col].astype(int).to_numpy(),
                merged[f"{pvalue_col}_a"].to_numpy(dtype=float),
                merged[f"{pvalue_col}_b"].to_numpy(dtype=float),
            ))

    if progress_desc:
        progress = tqdm(total=len(tasks), desc=progress_desc, unit="cmp")
    else:
        progress = None

    if worker_count <= 1:
        for task in tasks:
            rows.append(_compute_delong_task(task))
            if progress is not None:
                progress.update(1)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_compute_delong_task, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()

    delong_df = pd.DataFrame(rows)
    if delong_df.empty:
        return delong_df

    delong_df["pvalue_bh"] = (
        delong_df.groupby(list(group_cols), sort=False)["pvalue"]
        .transform(lambda values: benjamini_hochberg(values.to_numpy()))
    )
    delong_df["pvalue_bonferroni"] = (
        delong_df.groupby(list(group_cols), sort=False)["pvalue"]
        .transform(lambda values: bonferroni_adjust(values.to_numpy()))
    )
    delong_df["proteoforge_better"] = delong_df["delta_auroc"] > 0
    return delong_df


def best_mcc_statistic(
        y_true: Iterable[bool] | np.ndarray,
        y_pval: Iterable[float] | np.ndarray,
        thresholds: Sequence[float] | None = None,
) -> float:
    threshold_grid = np.asarray(STANDARD_THRESHOLDS if thresholds is None else thresholds, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_pval_arr = np.asarray(y_pval, dtype=float)
    finite_mask = np.isfinite(y_pval_arr)
    y_true_arr = y_true_arr[finite_mask]
    y_pval_arr = y_pval_arr[finite_mask]

    if y_true_arr.size == 0 or np.unique(y_true_arr).size < 2 or threshold_grid.size == 0:
        return np.nan

    positive_mask = y_true_arr[:, None]
    negative_mask = ~positive_mask
    predicted_positive = y_pval_arr[:, None] <= threshold_grid[None, :]
    predicted_negative = ~predicted_positive

    tp = np.sum(predicted_positive & positive_mask, axis=0, dtype=float)
    fp = np.sum(predicted_positive & negative_mask, axis=0, dtype=float)
    tn = np.sum(predicted_negative & negative_mask, axis=0, dtype=float)
    fn = np.sum(predicted_negative & positive_mask, axis=0, dtype=float)

    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    numerator = (tp * tn) - (fp * fn)
    mcc = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    return float(np.max(mcc))


def matched_sensitivity_statistic(
        y_true: Iterable[bool] | np.ndarray,
        y_pval: Iterable[float] | np.ndarray,
        matched_fpr: float = MATCHED_FPR,
) -> float:
    metrics = utils.evaluate_at_fixed_fpr(
        np.asarray(y_true, dtype=int),
        np.asarray(y_pval, dtype=float),
        target_fpr=matched_fpr,
    )
    return float(metrics["Sensitivity"])


def compute_paired_metric_table(
        score_frame: pd.DataFrame,
        group_cols: Sequence[str],
        method_pairs: Sequence[tuple[str, str]],
        metric_name: str,
        metric_func: Callable[[np.ndarray, np.ndarray], float],
        n_permutations: int = 1000,
        seed: int = 42,
        item_col: str = "item_id",
        truth_col: str = "true_label",
        pvalue_col: str = "score_pvalue",
        progress_desc: str | None = None,
        n_workers: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped_frames = list(score_frame.groupby(list(group_cols), dropna=False, sort=False))
    tasks: list[dict[str, object]] = []
    worker_count = resolve_worker_count(n_workers)
    progress_bar = None
    task_seed = seed

    for group_values, group_df in grouped_frames:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_payload = dict(zip(group_cols, group_values))

        for method_a, method_b in method_pairs:
            subset_a = group_df[group_df["method"] == method_a][[item_col, truth_col, pvalue_col]].copy()
            subset_b = group_df[group_df["method"] == method_b][[item_col, truth_col, pvalue_col]].copy()
            if subset_a.empty or subset_b.empty:
                continue

            merged = subset_a.merge(
                subset_b,
                on=[item_col, truth_col],
                how="inner",
                suffixes=("_a", "_b"),
            )
            if merged.empty:
                continue

            y_true = merged[truth_col].astype(bool).to_numpy()
            score_a = merged[f"{pvalue_col}_a"].to_numpy(dtype=float)
            score_b = merged[f"{pvalue_col}_b"].to_numpy(dtype=float)
            finite_mask = np.isfinite(score_a) & np.isfinite(score_b)
            y_true = y_true[finite_mask]
            score_a = score_a[finite_mask]
            score_b = score_b[finite_mask]
            if y_true.size == 0 or np.unique(y_true).size < 2:
                continue

            value_a = float(metric_func(y_true, score_a))
            value_b = float(metric_func(y_true, score_b))
            tasks.append({
                "group_payload": group_payload,
                "metric": metric_name,
                "method_a": method_a,
                "method_b": method_b,
                "y_true": y_true,
                "score_a": score_a,
                "score_b": score_b,
                "value_a": value_a,
                "value_b": value_b,
                "delta_value": value_a - value_b,
                "n_permutations": int(n_permutations),
                "seed": int(task_seed),
                "metric_func": metric_func,
            })
            task_seed += 1

    if progress_desc and tasks:
        if worker_count <= 1:
            progress_bar = tqdm(total=len(tasks), desc=progress_desc, unit="cmp")
        else:
            chunk_count = _chunk_count_per_task(len(tasks), worker_count, n_permutations)
            progress_bar = tqdm(total=len(tasks) * chunk_count, desc=progress_desc, unit="chunk")

    if worker_count <= 1:
        for task in tasks:
            rows.append(_compute_paired_metric_task(task))
            if progress_bar is not None:
                progress_bar.update(1)
    else:
        chunk_tasks = _build_permutation_chunk_tasks(tasks, worker_count)
        exceed_counts = [0] * len(tasks)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_compute_paired_metric_chunk_task, chunk_task) for chunk_task in chunk_tasks]
            for future in as_completed(futures):
                task_index, exceed_count = future.result()
                exceed_counts[task_index] += exceed_count
                if progress_bar is not None:
                    progress_bar.update(1)

        for task_index, task in enumerate(tasks):
            rows.append(_finalize_paired_metric_row(task, exceed_counts[task_index]))

    if progress_bar is not None:
        progress_bar.close()

    metric_df = pd.DataFrame(rows)
    if metric_df.empty:
        return metric_df

    metric_df["pvalue_bh"] = (
        metric_df.groupby(list(group_cols), sort=False)["pvalue"]
        .transform(lambda values: benjamini_hochberg(values.to_numpy()))
    )
    metric_df["pvalue_bonferroni"] = (
        metric_df.groupby(list(group_cols), sort=False)["pvalue"]
        .transform(lambda values: bonferroni_adjust(values.to_numpy()))
    )
    return metric_df


def _compute_delong_task(
        task: tuple[dict[str, object], str, str, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    group_payload, method_a, method_b, y_true, score_a, score_b = task
    result = utils.delong_test(y_true, score_a, score_b)
    y_true_bool = y_true.astype(bool)
    return {
        **group_payload,
        "method_a": method_a,
        "method_b": method_b,
        "n_items": int(len(y_true)),
        "n_positive": int(y_true_bool.sum()),
        "n_negative": int((~y_true_bool).sum()),
        "auroc_a": float(result["AUROC_A"]),
        "auroc_b": float(result["AUROC_B"]),
        "delta_auroc": float(result["delta"]),
        "z_statistic": float(result["statistic"]),
        "pvalue": float(result["pvalue"]),
    }


def _compute_paired_metric_task(
        task: dict[str, object],
) -> dict[str, object]:
    y_true = task["y_true"]
    score_a = task["score_a"]
    score_b = task["score_b"]
    n_permutations = int(task["n_permutations"])
    seed = int(task["seed"])
    metric_func = task["metric_func"]
    rng = np.random.default_rng(seed)
    delta_value = float(task["delta_value"])

    exceed_count = 0
    for _ in range(n_permutations):
        swap_mask = rng.random(y_true.size) < 0.5
        perm_a = np.where(swap_mask, score_b, score_a)
        perm_b = np.where(swap_mask, score_a, score_b)
        perm_delta = float(metric_func(y_true, perm_a) - metric_func(y_true, perm_b))
        if abs(perm_delta) >= abs(delta_value):
            exceed_count += 1

    return _finalize_paired_metric_row(task, exceed_count)


def _chunk_count_per_task(task_count: int, worker_count: int, n_permutations: int) -> int:
    if task_count <= 0:
        return 0
    return max(1, min(n_permutations, ceil(worker_count / task_count)))


def _build_permutation_chunk_tasks(tasks: list[dict[str, object]], worker_count: int) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, float, int, int, Callable[[np.ndarray, np.ndarray], float]]]:
    chunk_tasks = []
    chunk_count = _chunk_count_per_task(len(tasks), worker_count, int(tasks[0]["n_permutations"])) if tasks else 0

    for task_index, task in enumerate(tasks):
        n_permutations = int(task["n_permutations"])
        chunk_size = ceil(n_permutations / chunk_count)
        remaining = n_permutations
        chunk_seed = int(task["seed"])

        while remaining > 0:
            current_chunk = min(chunk_size, remaining)
            chunk_tasks.append((
                task_index,
                task["y_true"],
                task["score_a"],
                task["score_b"],
                float(task["delta_value"]),
                current_chunk,
                chunk_seed,
                task["metric_func"],
            ))
            remaining -= current_chunk
            chunk_seed += 1

    return chunk_tasks


def _compute_paired_metric_chunk_task(
        task: tuple[int, np.ndarray, np.ndarray, np.ndarray, float, int, int, Callable[[np.ndarray, np.ndarray], float]],
) -> tuple[int, int]:
    task_index, y_true, score_a, score_b, delta_value, n_permutations, seed, metric_func = task
    rng = np.random.default_rng(seed)
    exceed_count = 0

    for _ in range(n_permutations):
        swap_mask = rng.random(y_true.size) < 0.5
        perm_a = np.where(swap_mask, score_b, score_a)
        perm_b = np.where(swap_mask, score_a, score_b)
        perm_delta = float(metric_func(y_true, perm_a) - metric_func(y_true, perm_b))
        if abs(perm_delta) >= abs(delta_value):
            exceed_count += 1

    return task_index, exceed_count


def _finalize_paired_metric_row(task: dict[str, object], exceed_count: int) -> dict[str, object]:
    y_true = task["y_true"]
    pvalue = (exceed_count + 1.0) / (int(task["n_permutations"]) + 1.0)
    return {
        **task["group_payload"],
        "metric": task["metric"],
        "method_a": task["method_a"],
        "method_b": task["method_b"],
        "n_items": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "n_negative": int((~y_true).sum()),
        "value_a": float(task["value_a"]),
        "value_b": float(task["value_b"]),
        "delta_value": float(task["delta_value"]),
        "pvalue": pvalue,
        "n_permutations": int(task["n_permutations"]),
        "test_method": "paired_permutation",
    }
