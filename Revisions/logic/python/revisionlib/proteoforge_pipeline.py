import math
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from ProteoForge import disluster, model, weight

from .truth import apply_biological_absence_truth


DEFAULT_CORRECTION = {
    'strategy': 'two-step',
    'methods': ('bonferroni', 'fdr_bh'),
}


def recommend_revision_n_jobs(total_cores: int | None = None) -> int:
    total_cores = total_cores or mp.cpu_count()
    return max(1, min(total_cores, math.ceil(total_cores * 0.75)))


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.1f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}m'
    return f'{seconds / 3600:.2f}h'


def _stringify_annotation_value(value: object) -> str:
    if isinstance(value, np.ndarray):
        return str(value.tolist())
    return str(value)


def _build_summary(full_data: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        'Protein', 'pertProtein', 'PeptideID', 'pertPeptide', 'pertPFG',
        'basePertPeptide', 'basePertPFG',
        'isCompMiss', 'Reason', 'baseReason',
        'pval', 'adj_pval', 'ClusterID',
    ]
    return full_data[
        [column for column in summary_cols if column in full_data.columns]
    ].drop_duplicates().copy()


def _fit_model_with_optional_diagnostics(
        data: pd.DataFrame,
        formula: str,
        model_type: str,
        weight_col: str | None,
        collect_rlm_weights: bool,
) -> tuple[float, pd.DataFrame | None]:
    try:
        if model_type == 'ols':
            cur_model = smf.ols(formula=formula, data=data).fit()
        elif model_type == 'wls':
            if weight_col is None:
                raise ValueError('Weight column must be provided for WLS model.')
            cur_model = smf.wls(formula=formula, data=data, weights=data[weight_col]).fit()
        elif model_type == 'rlm':
            cur_model = smf.rlm(
                formula=formula,
                data=data,
                M=sm.robust.norms.HuberT(),
            ).fit(scale_est=sm.robust.scale.HuberScale())
        elif model_type == 'glm':
            if weight_col is not None:
                cur_model = smf.glm(
                    formula=formula,
                    data=data,
                    family=sm.families.Gaussian(),
                    var_weights=data[weight_col],
                ).fit()
            else:
                cur_model = smf.glm(
                    formula=formula,
                    data=data,
                    family=sm.families.Gaussian(),
                ).fit()
        elif model_type == 'quantile':
            if weight_col is not None:
                cur_model = smf.quantreg(formula=formula, data=data).fit(q=0.5, weights=data[weight_col])
            else:
                cur_model = smf.quantreg(formula=formula, data=data).fit(q=0.5)
        else:
            raise ValueError(f'Unsupported model type: {model_type}')

        pval = float(cur_model.wald_test_terms(scalar=False).pvalues[-1])
        if model_type != 'rlm' or not collect_rlm_weights:
            return pval, None

        keep_cols = [
            column for column in [
                'Protein', 'Peptide', 'PeptideID', 'Sample', 'Condition',
                'Intensity', 'adjIntensity', 'Weight', 'isReal', 'isCompMiss',
            ]
            if column in data.columns
        ]
        diagnostics = data[keep_cols].copy()
        diagnostics['HuberWeight'] = np.asarray(cur_model.weights, dtype=float)
        return pval, diagnostics
    except Exception:
        return np.nan, None


def prepare_proteoforge_data(
        test_data: pd.DataFrame,
        n_jobs: int = 1,
) -> pd.DataFrame:
    prepared = test_data.copy()

    weights_data = weight.generate_weights_data(
        prepared,
        sample_cols=['Sample'],
        log_intensity_col='log10Intensity',
        adj_intensity_col='adjIntensity',
        control_condition='control',
        condition_col='Condition',
        protein_col='Protein',
        peptide_col='Peptide',
        is_real_col='isReal',
        is_comp_miss_col='isCompMiss',
        sparse_imputed_val=1e-10,
        dense_imputed_val=0.75,
        verbose=False,
    )
    prepared['Weight'] = (
        (weights_data['W_Impute'] * 0.90) +
        (weights_data['W_RevTechVar'] * 0.10)
    )

    for column in ['pertCondition', 'pertShift']:
        if column in prepared.columns:
            prepared[column] = prepared[column].map(_stringify_annotation_value)

    clusters = disluster.distance_and_cluster(
        data=prepared,
        protein_col='Protein',
        peptide_col='PeptideID',
        cond_col='Condition',
        quant_col='adjIntensity',
        clustering_params={
            'min_clusters': 1,
            'distance_transform': 'corr',
            'clustering_method': 'hybrid_outlier_cut',
            'linkage_method': 'ward',
            'distance_metric': 'euclidean',
        },
        n_jobs=n_jobs,
        verbose=False,
    )
    prepared = prepared.merge(
        clusters[['Protein', 'PeptideID', 'cluster_label']],
        on=['Protein', 'PeptideID'],
        how='left',
    ).rename(columns={'cluster_label': 'ClusterID'})

    return apply_biological_absence_truth(prepared)


def _worker_process_protein_models(args: tuple) -> dict[str, list]:
    protein_id, protein_data, formula, model_types, peptide_col, weight_col, collect_rlm_weights = args
    unique_peptides = protein_data[peptide_col].unique()

    if len(unique_peptides) < 2:
        return {'results': [], 'diagnostics': []}

    results: list[dict[str, object]] = []
    diagnostics: list[pd.DataFrame] = []
    for peptide in unique_peptides:
        protein_data_copy = protein_data.copy()
        protein_data_copy.loc[:, 'allothers'] = np.where(
            protein_data_copy[peptide_col] == peptide,
            peptide,
            'allothers',
        )

        for model_type in model_types:
            pval, diagnostic_frame = _fit_model_with_optional_diagnostics(
                data=protein_data_copy,
                formula=formula,
                model_type=model_type,
                weight_col=weight_col,
                collect_rlm_weights=collect_rlm_weights,
            )
            results.append({
                'protein_id': protein_id,
                'peptide_id': peptide,
                'model_type': model_type,
                'pval': pval,
            })

            if diagnostic_frame is not None:
                diagnostic_frame = diagnostic_frame.copy()
                diagnostic_frame['target_peptide'] = peptide
                diagnostic_frame['model_type'] = model_type
                diagnostic_frame['is_target_peptide'] = (
                    protein_data_copy[peptide_col].to_numpy() == peptide
                )
                diagnostics.append(diagnostic_frame)

    return {'results': results, 'diagnostics': diagnostics}


def run_proteoforge_models(
        prepared_data: pd.DataFrame,
        model_types: tuple[str, ...] = ('rlm',),
        correction: dict | None = None,
        n_jobs: int = 1,
    collect_rlm_weights: bool = False,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]] | tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    correction = correction or DEFAULT_CORRECTION
    cur_model = model.LinearModel(
        data=prepared_data,
        protein_col='Protein',
        peptide_col='Peptide',
        cond_col='Condition',
        intensity_col='adjIntensity',
        weight_col='Weight',
    )

    n_jobs = model.select_n_jobs(n_jobs)
    protein_groups = prepared_data.groupby('Protein', sort=False)
    tasks = [
        (
            protein_id,
            group_df,
            cur_model.formula,
            tuple(model_types),
            cur_model.peptide_col,
            cur_model.weight_col,
            collect_rlm_weights,
        )
        for protein_id, group_df in protein_groups
    ]

    with Pool(processes=n_jobs) as pool:
        results_list = list(pool.imap(_worker_process_protein_models, tasks))

    flat_results = [
        item
        for worker_output in results_list
        for item in worker_output['results']
    ]
    if not flat_results:
        return ({}, pd.DataFrame()) if collect_rlm_weights else {}

    result_frame = pd.DataFrame(flat_results).rename(columns={
        'protein_id': cur_model.protein_col,
        'peptide_id': cur_model.peptide_col,
    })

    diagnostic_frame = pd.DataFrame()
    if collect_rlm_weights:
        diagnostic_parts = [
            part
            for worker_output in results_list
            for part in worker_output['diagnostics']
            if part is not None and not part.empty
        ]
        if diagnostic_parts:
            diagnostic_frame = pd.concat(diagnostic_parts, ignore_index=True)

    outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for model_type in model_types:
        model_results = result_frame[result_frame['model_type'] == model_type].drop(
            columns='model_type'
        )
        model_results = cur_model._apply_correction_strategy(
            model_results.copy(),
            correction['strategy'],
            correction['methods'],
        )
        full_data = prepared_data.merge(
            model_results,
            on=[cur_model.protein_col, cur_model.peptide_col],
            how='left',
        ).drop_duplicates().reset_index(drop=True)
        outputs[model_type] = (_build_summary(full_data), full_data)

    if collect_rlm_weights:
        return outputs, diagnostic_frame

    return outputs


def run_proteoforge_pipeline(
        test_data: pd.DataFrame,
        model_type: str = 'rlm',
        correction: dict | None = None,
        n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the revision ProteoForge analysis and return summary plus full data."""
    prepared_data = prepare_proteoforge_data(test_data=test_data, n_jobs=n_jobs)
    model_outputs = run_proteoforge_models(
        prepared_data=prepared_data,
        model_types=(model_type,),
        correction=correction,
        n_jobs=n_jobs,
    )
    return model_outputs[model_type]