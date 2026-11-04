"""Original Raw106 nested-CV score implementation.

Execute in notebook 12's namespace after loading 00_shared_config and setting
nmr106_cols from the validated reference manifest. This file intentionally
uses that shared context; it is not a standalone importable training package.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

REQUIRED_SHARED_GLOBALS = ('AUDIT_DIR', 'CPU_WORKERS', 'NMR_BOUNDARY_POLICY', 'NMR_ELASTICNET_ALPHA_GRID', 'NMR_ELASTICNET_INNER_CV', 'NMR_ELASTICNET_L1_RATIOS', 'NMR_ELASTICNET_MAX_ITER', 'RANDOM_STATE', 'atomic_write_csv', 'canonical_json_digest', 'eid_digest', 'log_progress', 'nmr106_cols')
_score106_context = globals()
_missing_context = [name for name in REQUIRED_SHARED_GLOBALS if name not in _score106_context]
if _missing_context:
    raise RuntimeError(
        "score106.py requires notebook 12's configured namespace. Missing: "
        + ", ".join(_missing_context)
    )
(AUDIT_DIR, CPU_WORKERS, NMR_BOUNDARY_POLICY, NMR_ELASTICNET_ALPHA_GRID, NMR_ELASTICNET_INNER_CV, NMR_ELASTICNET_L1_RATIOS, NMR_ELASTICNET_MAX_ITER, RANDOM_STATE, atomic_write_csv, canonical_json_digest, eid_digest, log_progress, nmr106_cols) = (
    _score106_context[name] for name in REQUIRED_SHARED_GLOBALS
)

from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from joblib import Parallel, delayed


NMR_ALPHA_GRID=np.asarray(NMR_ELASTICNET_ALPHA_GRID,dtype=float)
NMR_L1_RATIO_GRID=np.asarray(NMR_ELASTICNET_L1_RATIOS,dtype=float)
NMR_INNER_FOLDS=int(NMR_ELASTICNET_INNER_CV)
NMR_SELECTION_HORIZON_YEARS=10.0

def survival_y(frame):
    return Surv.from_arrays(
        event=frame["Event"].astype(bool).to_numpy(),
        time=frame["Time"].astype(float).to_numpy(),
    )

def uno_at_horizon(train_reference,eval_frame,risk,horizon=NMR_SELECTION_HORIZON_YEARS):
    tau=min(
        float(horizon),
        float(train_reference.loc[train_reference.Event.eq(0),"Time"].max())-1e-8,
        float(eval_frame.Time.max())-1e-8,
    )
    if tau<=0:
        raise ValueError("Invalid IPCW tau")
    return float(
        concordance_index_ipcw(
            survival_y(train_reference),
            survival_y(eval_frame),
            np.asarray(risk,float),
            tau=tau,
        )[0]
    )






def fit_nmr_score_nested(train_frame,val_frame,test_frame):
    # NMR_score uses only the ordered Raw106 contract. The main cohort itself
    # remains PRS+Raw106 complete; remaining Raw106 missingness is handled
    # inside each training fold.
    raw_train=train_frame[nmr106_cols].replace([np.inf,-np.inf],np.nan).astype(float)
    raw_val=val_frame[nmr106_cols].replace([np.inf,-np.inf],np.nan).astype(float)
    raw_test=test_frame[nmr106_cols].replace([np.inf,-np.inf],np.nan).astype(float)
    all_missing=raw_train.columns[raw_train.notna().sum(axis=0).eq(0)].tolist()
    if all_missing:
        raise RuntimeError(f"Raw106 fields entirely missing in train: {all_missing}")

    event=train_frame.Event.astype(int).to_numpy()
    cv=StratifiedKFold(
        n_splits=NMR_INNER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores=np.full(
        (len(NMR_L1_RATIO_GRID),len(NMR_ALPHA_GRID),NMR_INNER_FOLDS),
        np.nan,
        dtype=float,
    )

    def _run_inner_fold(fold, inner_train, inner_val):

        log_progress(
            f"NMR_score Raw106 inner CV fold={fold}/{NMR_INNER_FOLDS} started"
        )

        fold_imputer=SimpleImputer(strategy="median")
        fold_scaler=StandardScaler()

        Xi_train=fold_scaler.fit_transform(
            fold_imputer.fit_transform(raw_train.iloc[inner_train])
        )
        Xi_val=fold_scaler.transform(
            fold_imputer.transform(raw_train.iloc[inner_val])
        )

        fold_train=train_frame.iloc[inner_train].reset_index(drop=True)
        fold_val=train_frame.iloc[inner_val].reset_index(drop=True)
        yi=survival_y(fold_train)

        local_scores=np.full(
            (len(NMR_L1_RATIO_GRID),len(NMR_ALPHA_GRID)),
            np.nan,
            dtype=float,
        )

        for l1_index,l1_ratio in enumerate(NMR_L1_RATIO_GRID):

            log_progress(
                f"NMR_score Raw106 inner CV fold={fold}/{NMR_INNER_FOLDS} "
                f"l1_ratio={l1_ratio:g}"
            )

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "error",
                    SklearnConvergenceWarning
                )

                model=CoxnetSurvivalAnalysis(
                    l1_ratio=float(l1_ratio),
                    alphas=NMR_ALPHA_GRID,
                    max_iter=NMR_ELASTICNET_MAX_ITER,
                    tol=1e-8,
                )

                model.fit(Xi_train,yi)

            fitted_alpha_count=model.coef_.shape[1]

            if fitted_alpha_count != len(NMR_ALPHA_GRID):
                log_progress(
                    f"Coxnet alpha path truncated: "
                    f"requested={len(NMR_ALPHA_GRID)}, "
                    f"returned={fitted_alpha_count}"
                )

            for alpha_index in range(fitted_alpha_count):

                beta=np.asarray(
                    model.coef_[:,alpha_index],
                    dtype=float
                )

                if np.allclose(beta,0):
                    continue

                local_scores[
                    l1_index,
                    alpha_index
                ]=uno_at_horizon(
                    fold_train,
                    fold_val,
                    Xi_val@beta,
                )

        return fold,local_scores


    fold_results=Parallel(
        n_jobs=CPU_WORKERS,
        backend="loky",
    )(
        delayed(_run_inner_fold)(
            fold,
            inner_train,
            inner_val,
        )
        for fold,(inner_train,inner_val)
        in enumerate(cv.split(raw_train,event),start=1)
    )

    for fold,local_scores in fold_results:
        scores[:,:,fold-1]=local_scores

    mean_score=np.where(np.isfinite(scores).all(axis=2), np.mean(scores,axis=2), np.nan)
    if not np.isfinite(mean_score).any():
        raise RuntimeError("No valid inner-CV Raw106 NMR_score candidate")
    best_index=np.unravel_index(np.nanargmax(mean_score),mean_score.shape)
    best_l1=float(NMR_L1_RATIO_GRID[best_index[0]])
    best_alpha=float(NMR_ALPHA_GRID[best_index[1]])

    boundary_audit={
        "best_l1_ratio":best_l1,
        "best_alpha":best_alpha,
        "l1_first_grid_value":float(NMR_L1_RATIO_GRID[0]),
        "l1_last_grid_value":float(NMR_L1_RATIO_GRID[-1]),
        "alpha_first_grid_value":float(NMR_ALPHA_GRID[0]),
        "alpha_last_grid_value":float(NMR_ALPHA_GRID[-1]),
        "l1_at_first_boundary":bool(best_index[0]==0),
        "l1_at_last_boundary":bool(best_index[0]==len(NMR_L1_RATIO_GRID)-1),
        "alpha_at_first_boundary":bool(best_index[1]==0),
        "alpha_at_last_boundary":bool(best_index[1]==len(NMR_ALPHA_GRID)-1),
        "boundary_policy":NMR_BOUNDARY_POLICY,
    }
    boundary_audit["accepted_parameter_endpoint"]=bool(
        boundary_audit["l1_at_last_boundary"]
        and np.isclose(best_l1,1.0)
    )
    boundary_audit["accepted_parameter_endpoint_note"]=(
        "l1_ratio=1.0 is the legal pure-L1 endpoint; it is audited but does not require upward grid extension."
        if boundary_audit["accepted_parameter_endpoint"]
        else None
    )
    boundary_audit["any_boundary_hit"]=bool(
        boundary_audit["l1_at_first_boundary"]
        or boundary_audit["l1_at_last_boundary"]
        or boundary_audit["alpha_at_first_boundary"]
        or boundary_audit["alpha_at_last_boundary"]
    )
    boundary_audit["search_extension_required"]=bool(
        boundary_audit["l1_at_first_boundary"]
        or boundary_audit["alpha_at_first_boundary"]
        or boundary_audit["alpha_at_last_boundary"]
    )
    boundary_audit["fail_trigger_boundaries"]="l1_first|alpha_first|alpha_last"
    boundary_path=os.path.join(AUDIT_DIR,"nmr_score_boundary_audit.csv")
    atomic_write_csv(pd.DataFrame([boundary_audit]),boundary_path,index=False)
    if boundary_audit["accepted_parameter_endpoint"]:
        log_progress(
            "NMR_score boundary audit | accepted_parameter_endpoint=true | "
            "l1_ratio=1.0 (pure L1)"
        )
    if boundary_audit["search_extension_required"]:
        message=(
            "NMR_score optimum is on a search boundary that can truncate selection; "
            "extend the lower l1_ratio or alpha grid and rerun before final analysis. "
            f"l1_ratio={best_l1:g}, alpha={best_alpha:g}"
        )
        if NMR_BOUNDARY_POLICY=="fail":
            raise RuntimeError(message)
        warnings.warn(message)

    # Refit train-only preprocessing and the selected model after inner-CV selection.
    outer_imputer=SimpleImputer(strategy="median")
    outer_scaler=StandardScaler()
    train_imputed=outer_imputer.fit_transform(raw_train)
    X_train=outer_scaler.fit_transform(train_imputed)
    X_val=outer_scaler.transform(outer_imputer.transform(raw_val))
    X_test=outer_scaler.transform(outer_imputer.transform(raw_test))

    final_model=CoxnetSurvivalAnalysis(
        l1_ratio=best_l1,
        alphas=[best_alpha],
        max_iter=NMR_ELASTICNET_MAX_ITER,
        tol=1e-8,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error",SklearnConvergenceWarning)
        final_model.fit(X_train,survival_y(train_frame))
    beta=np.asarray(final_model.coef_[:,0],float)
    if np.allclose(beta,0):
        raise RuntimeError("Selected Raw106 NMR_score has no non-zero coefficient")

    audit_rows=[]
    for li,l1 in enumerate(NMR_L1_RATIO_GRID):
        for ai,alpha in enumerate(NMR_ALPHA_GRID):
            audit_rows.append({
                "l1_ratio":float(l1),
                "alpha":float(alpha),
                "mean_inner_uno_c_10y":float(mean_score[li,ai]) if np.isfinite(mean_score[li,ai]) else np.nan,
                "valid_folds":int(np.isfinite(scores[li,ai]).sum()),
                "selected":bool((li,ai)==best_index),
                "l1_boundary":bool(li in {0,len(NMR_L1_RATIO_GRID)-1}),
                "alpha_boundary":bool(ai in {0,len(NMR_ALPHA_GRID)-1}),
            })
    audit=pd.DataFrame(audit_rows)
    coefficient=pd.DataFrame({
        "Contract_Order":np.arange(1,len(nmr106_cols)+1),
        "Field_ID":nmr106_cols,
        "Coefficient":beta,
        "Selected":np.abs(beta)>1e-12,
        "In_Raw106":pd.Series(nmr106_cols).isin(set(nmr106_cols)).to_numpy(),
    })
    preprocessing=pd.DataFrame({
        "Contract_Order":np.arange(1,len(nmr106_cols)+1),
        "Field_ID":nmr106_cols,
        "Train_Median_Imputation":outer_imputer.statistics_,
        "Train_Standardization_Mean":outer_scaler.mean_,
        "Train_Standardization_Scale":outer_scaler.scale_,
        "Train_Missing_N":raw_train.isna().sum().to_numpy(),
        "Train_Missing_Rate":raw_train.isna().mean().to_numpy(),
        "Validation_Missing_N":raw_val.isna().sum().to_numpy(),
        "Validation_Missing_Rate":raw_val.isna().mean().to_numpy(),
        "Test_Missing_N":raw_test.isna().sum().to_numpy(),
        "Test_Missing_Rate":raw_test.isna().mean().to_numpy(),
    })
    selected_n=int((np.abs(beta)>1e-12).sum())
    metadata={
        "score_name":"NMR_score",
        "source":"train-derived elastic-net Cox score",
        "predictor_contract_name":"Raw106",
        "predictor_field_count":len(nmr106_cols),
        "predictor_field_ids_sha256":canonical_json_digest(nmr106_cols),
        "source_definition":"Prespecified Raw106 candidate panel",
        "source_definition_version":"Raw106-v1; Category 220/phase3 companion contract verified 2026-08-24",
        "explicitly_excluded_extension_fields":[20280,20281],
        "explicitly_excluded_extension_meanings":[
            "glucose-lactate",
            "spectrometer-corrected alanine",
        ],
        "criterion":"mean inner-fold Uno IPCW C-index at 10 years",
        "inner_cv_preprocessing":"fold-specific median imputation and standardization",
        "final_preprocessing_fit_split":"train",
        "alpha_grid":NMR_ALPHA_GRID.tolist(),
        "l1_ratio_grid":NMR_L1_RATIO_GRID.tolist(),
        "inner_folds":NMR_INNER_FOLDS,
        "best_l1_ratio":best_l1,
        "best_alpha":best_alpha,
        "best_score":float(mean_score[best_index]),
        "selected_n":selected_n,
        "selected_n_fixed":False,
        "selected_n_note":"Selected_n is determined by train-only inner CV; 39 is not imposed and is only a historical legacy-run count.",
        "boundary_audit":boundary_audit,
        "train_eid_sha256":eid_digest(train_frame),
        "validation_used_for_score_selection":False,
        "test_used_for_score_selection":False,
    }
    return (
        X_train@beta,
        X_val@beta,
        X_test@beta,
        audit,
        coefficient,
        preprocessing,
        metadata,
        boundary_audit,
    )
