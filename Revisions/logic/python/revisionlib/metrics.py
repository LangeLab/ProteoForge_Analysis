import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def _is_grouping_positive(labels: np.ndarray) -> bool:
    unique_labels = np.unique(labels)
    return len(unique_labels) > 1 and not (len(unique_labels) == 1 and unique_labels[0] == -1)


def _per_protein_ari_values(
        data: pd.DataFrame,
        true_col: str,
        pred_col: str,
        protein_col: str,
        include_unperturbed: bool,
) -> dict[str, float]:
    protein_aris: dict[str, float] = {}

    for protein, sub in data.groupby(protein_col):
        true_labels = sub[true_col].to_numpy()
        pred_labels = sub[pred_col].to_numpy()
        is_perturbed = _is_grouping_positive(true_labels)

        if not is_perturbed and not include_unperturbed:
            continue

        if not is_perturbed:
            protein_aris[protein] = 1.0 if len(np.unique(pred_labels)) == 1 else 0.0
        else:
            protein_aris[protein] = float(adjusted_rand_score(true_labels, pred_labels))

    return protein_aris


def _concatenated_labels(
        protein_frame: pd.DataFrame,
        true_col: str,
        pred_col: str,
        protein_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    true_labels: list[str] = []
    pred_labels: list[str] = []

    for protein, sub in protein_frame.groupby(protein_col):
        true_labels.extend([f'{protein}::{value}' for value in sub[true_col].to_numpy()])
        pred_labels.extend([f'{protein}::{value}' for value in sub[pred_col].to_numpy()])

    return np.asarray(true_labels), np.asarray(pred_labels)


def _factorize_labels(labels: np.ndarray) -> np.ndarray:
    codes, _ = pd.factorize(labels, sort=False)
    return codes.astype(np.int64, copy=False)


def _prepare_concatenated_blocks(
        protein_frame: pd.DataFrame,
        true_col: str,
        pred_col: str,
        protein_col: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    true_blocks: list[np.ndarray] = []
    pred_blocks: list[np.ndarray] = []
    true_offset = 0
    pred_offset = 0

    for _, sub in protein_frame.groupby(protein_col, sort=False):
        true_codes = _factorize_labels(sub[true_col].to_numpy())
        pred_codes = _factorize_labels(sub[pred_col].to_numpy())

        true_blocks.append(true_codes + true_offset)
        pred_blocks.append(pred_codes + pred_offset)

        true_offset += int(true_codes.max()) + 1
        pred_offset += int(pred_codes.max()) + 1

    return true_blocks, pred_blocks


def bootstrap_ari_ci(
        data: pd.DataFrame,
        true_col: str = 'pertPFG',
        pred_col: str = 'ClusterID',
        protein_col: str = 'Protein',
        mode: str = 'per_protein',
        include_unperturbed: bool = False,
        n_bootstrap: int = 1000,
        seed: int = 42,
        alpha: float = 0.05,
) -> dict:
    """
    Bootstrap percentile CI for peptide-grouping ARI.

    For concatenated mode the resampling unit is the protein, not individual
    peptides, so the CI matches the biological grouping unit and remains
    coherent with the point estimate.
    """
    df = data[[protein_col, true_col, pred_col]].copy()
    rng = np.random.default_rng(seed)

    if mode not in {'per_protein', 'concatenated'}:
        raise ValueError("mode must be either 'per_protein' or 'concatenated'")

    protein_aris = _per_protein_ari_values(
        data=df,
        true_col=true_col,
        pred_col=pred_col,
        protein_col=protein_col,
        include_unperturbed=include_unperturbed,
    )
    if not protein_aris:
        return {'ARI': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'n_proteins': 0}

    proteins = np.asarray(list(protein_aris.keys()))

    if mode == 'per_protein':
        protein_values = np.asarray(list(protein_aris.values()), dtype=float)
        ari_point = float(np.mean(protein_values))
        boot_aris = np.empty(n_bootstrap, dtype=float)
        for idx in range(n_bootstrap):
            sample_idx = rng.integers(0, len(protein_values), size=len(protein_values))
            boot_aris[idx] = float(np.mean(protein_values[sample_idx]))
    else:
        eligible = df[df[protein_col].isin(proteins)].copy()
        true_blocks, pred_blocks = _prepare_concatenated_blocks(
            protein_frame=eligible,
            true_col=true_col,
            pred_col=pred_col,
            protein_col=protein_col,
        )
        true_labels = np.concatenate(true_blocks)
        pred_labels = np.concatenate(pred_blocks)
        ari_point = float(adjusted_rand_score(true_labels, pred_labels))
        boot_aris = np.empty(n_bootstrap, dtype=float)
        protein_count = len(true_blocks)
        for idx in range(n_bootstrap):
            sampled_idx = rng.integers(0, protein_count, size=protein_count)
            boot_true = np.concatenate([true_blocks[i] for i in sampled_idx])
            boot_pred = np.concatenate([pred_blocks[i] for i in sampled_idx])
            boot_aris[idx] = float(adjusted_rand_score(boot_true, boot_pred))

    ci_lower = float(np.percentile(boot_aris, 100.0 * alpha / 2.0))
    ci_upper = float(np.percentile(boot_aris, 100.0 * (1.0 - alpha / 2.0)))
    return {
        'ARI': ari_point,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'n_proteins': len(proteins),
    }