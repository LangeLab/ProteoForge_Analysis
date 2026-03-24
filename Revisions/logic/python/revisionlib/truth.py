import numpy as np
import pandas as pd


def apply_biological_absence_truth(
        data: pd.DataFrame,
        pert_peptide_col: str = 'pertPeptide',
        pert_pfg_col: str = 'pertPFG',
        is_comp_miss_col: str = 'isCompMiss',
        reason_col: str = 'Reason',
        preserve_original: bool = True,
) -> pd.DataFrame:
    """
    Apply the revision-specific evaluation rule that complete-condition absence
    is treated as biologically perturbed for downstream benchmarking.

    The original labels are preserved as base* columns so validation can compare
    pre- and post-relabelled truth without losing provenance.
    """
    out = data.copy()

    if preserve_original:
        if pert_peptide_col in out.columns and 'basePertPeptide' not in out.columns:
            out['basePertPeptide'] = out[pert_peptide_col]
        if pert_pfg_col in out.columns and 'basePertPFG' not in out.columns:
            out['basePertPFG'] = out[pert_pfg_col]
        if reason_col in out.columns and 'baseReason' not in out.columns:
            out['baseReason'] = out[reason_col]

    if is_comp_miss_col not in out.columns:
        return out

    missing_mask = out[is_comp_miss_col].fillna(0).astype(int) == 1

    if pert_peptide_col in out.columns:
        out[pert_peptide_col] = np.where(
            missing_mask,
            True,
            out[pert_peptide_col].fillna(False),
        ).astype(bool)

    if reason_col in out.columns:
        out[reason_col] = np.where(
            missing_mask,
            'Biological Absence',
            out[reason_col],
        )

    if pert_pfg_col in out.columns:
        out.loc[missing_mask, pert_pfg_col] = 1

    return out


def build_rmethod_input_frame(test_data: pd.DataFrame) -> pd.DataFrame:
    """Return the R-method input frame with harmonized revision truth labels."""
    out = apply_biological_absence_truth(test_data)
    keep_cols = [
        'Protein', 'Peptide', 'PeptideID', 'Intensity', 'Condition', 'Sample',
        'pertProtein', 'pertPeptide', 'pertPFG',
        'basePertPeptide', 'basePertPFG', 'Reason', 'baseReason',
        'isReal', 'isCompMiss',
    ]
    out = out[[column for column in keep_cols if column in out.columns]].copy()

    if 'pertPFG' in out.columns:
        out['pertPFG'] = out['pertPFG'].astype('int32')
    if 'pertProtein' in out.columns:
        out['pertProtein'] = out['pertProtein'].astype(bool)
    if 'pertPeptide' in out.columns:
        out['pertPeptide'] = out['pertPeptide'].astype(bool)
    if 'basePertPeptide' in out.columns:
        out['basePertPeptide'] = out['basePertPeptide'].astype(bool)
    if 'basePertPFG' in out.columns:
        out['basePertPFG'] = out['basePertPFG'].astype('int32')

    return out.reset_index(drop=True)