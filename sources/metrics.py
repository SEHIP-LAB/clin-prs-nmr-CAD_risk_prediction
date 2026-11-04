"""Shared survival evaluation methods used by notebooks 13–15."""
from __future__ import annotations

import hashlib
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index

try:
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
except ImportError as error:
    raise ImportError(
        "scikit-survival is required for Uno IPCW C-index and Coxnet. "
        "Install it in the server environment with: uv pip install scikit-survival"
    ) from error

try:
    from dcurves import dca as dcurves_dca
except ImportError as error:
    raise ImportError(
        "dcurves is required for standard survival DCA. "
        "Install it in the server environment with: uv pip install 'dcurves>=1.1.7'"
    ) from error


CNS = {
    "blue": "#3C5488",
    "red": "#E64B35",
    "green": "#00A087",
    "teal": "#4DBBD5",
    "orange": "#F39B7F",
    "purple": "#8491B4",
    "gray": "#6E6E6E",
    "light_gray": "#D9D9D9",
    "black": "#222222",
}

plt.rcParams.update({
    "figure.dpi": 170,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "font.size": 13.0,
    "axes.titlesize": 15.0,
    "axes.labelsize": 13.5,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 11.0,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

HORIZONS = [5.0, 10.0]
NRI_CUTPOINTS = [
    float(value)
    for value in os.environ.get("CHD_NRI_CUTPOINTS", "0.05,0.10,0.20").split(",")
]
NRI_CUTPOINT_SOURCE = os.environ.get("CHD_NRI_CUTPOINT_SOURCE", "").strip()
if sorted(NRI_CUTPOINTS) != NRI_CUTPOINTS or any(
    value <= 0 or value >= 1 for value in NRI_CUTPOINTS
):
    raise ValueError("CHD_NRI_CUTPOINTS must be strictly increasing probabilities in (0, 1).")
CATEGORICAL_NRI_ENABLED = bool(NRI_CUTPOINT_SOURCE)
if not CATEGORICAL_NRI_ENABLED:
    warnings.warn("CHD_NRI_CUTPOINT_SOURCE is unset: categorical NRI will be omitted; continuous NRI and IDI remain active.")
BOOTSTRAP_N = int(os.environ.get("CHD_BOOTSTRAP_N", "1000"))
BOOTSTRAP_SEED = int(os.environ.get("CHD_BOOTSTRAP_SEED", "2026"))
IBS_GRID_N = int(os.environ.get("CHD_IBS_GRID_N", "51"))
DCA_THRESHOLDS = np.arange(0.01, 0.201, 0.005)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, output_dir, stem):
    ensure_dir(output_dir)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(output_dir, f"{stem}.{extension}"),
            dpi=600,
            bbox_inches="tight",
        )
    print("Saved figure:", os.path.join(output_dir, stem + ".png"))


def quantile_ci(values, lower=0.025, upper=0.975):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.quantile(values, lower)), float(np.quantile(values, upper))


def make_surv(frame):
    return Surv.from_arrays(
        event=frame["Event"].astype(bool).to_numpy(),
        time=frame["Time"].astype(float).to_numpy(),
    )


def eid_digest(frame):
    if "Eid" not in frame.columns:
        return "no_eid"
    values = pd.util.hash_pandas_object(frame["Eid"], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def harrell_c(frame, risk):
    return float(
        concordance_index(
            frame["Time"].astype(float).to_numpy(),
            -np.asarray(risk, dtype=float).reshape(-1),
            frame["Event"].astype(int).to_numpy(),
        )
    )


def uno_c(train_frame, eval_frame, risk, tau):
    if float(tau) <= 0:
        raise ValueError("Uno C tau must be positive.")
    estimate = concordance_index_ipcw(
        make_surv(train_frame),
        make_surv(eval_frame),
        np.asarray(risk, dtype=float).reshape(-1),
        tau=float(tau),
    )[0]
    return float(estimate)


def paired_bootstrap_uno(
    train_frame,
    eval_frame,
    risk_new,
    risk_reference,
    tau,
    bootstrap_n=BOOTSTRAP_N,
    seed=BOOTSTRAP_SEED,
):
    risk_new = np.asarray(risk_new, dtype=float).reshape(-1)
    risk_reference = np.asarray(risk_reference, dtype=float).reshape(-1)
    original_new = uno_c(train_frame, eval_frame, risk_new, tau)
    original_reference = uno_c(train_frame, eval_frame, risk_reference, tau)
    original_delta = original_new - original_reference

    rng = np.random.default_rng(seed)
    n = len(eval_frame)
    values = []
    for _ in range(int(bootstrap_n)):
        idx = rng.integers(0, n, size=n)
        boot_frame = eval_frame.iloc[idx].reset_index(drop=True)
        if boot_frame["Event"].nunique() < 2:
            continue
        try:
            values.append(
                uno_c(train_frame, boot_frame, risk_new[idx], tau)
                - uno_c(train_frame, boot_frame, risk_reference[idx], tau)
            )
        except Exception:
            continue
    lower, upper = quantile_ci(values)
    return {
        "Uno_C_New": original_new,
        "Uno_C_Reference": original_reference,
        "Delta_Uno_C": original_delta,
        "Delta_Uno_C_Lower95": lower,
        "Delta_Uno_C_Upper95": upper,
        "Bootstrap_N_Used": int(len(values)),
    }


def paired_bootstrap_harrell(
    eval_frame,
    risk_new,
    risk_reference,
    bootstrap_n=BOOTSTRAP_N,
    seed=BOOTSTRAP_SEED,
):
    risk_new = np.asarray(risk_new, dtype=float).reshape(-1)
    risk_reference = np.asarray(risk_reference, dtype=float).reshape(-1)
    original_new = harrell_c(eval_frame, risk_new)
    original_reference = harrell_c(eval_frame, risk_reference)
    rng = np.random.default_rng(seed)
    n = len(eval_frame)
    values = []
    for _ in range(int(bootstrap_n)):
        idx = rng.integers(0, n, size=n)
        boot_frame = eval_frame.iloc[idx].reset_index(drop=True)
        if boot_frame["Event"].nunique() < 2:
            continue
        try:
            values.append(
                harrell_c(boot_frame, risk_new[idx])
                - harrell_c(boot_frame, risk_reference[idx])
            )
        except Exception:
            continue
    lower, upper = quantile_ci(values)
    return {
        "Harrell_C_New": original_new,
        "Harrell_C_Reference": original_reference,
        "Delta_Harrell_C": original_new - original_reference,
        "Delta_Harrell_C_Lower95": lower,
        "Delta_Harrell_C_Upper95": upper,
        "Bootstrap_N_Used": int(len(values)),
    }


def fit_absolute_risk_calibrator(train_frame, train_risk, label):
    train_risk=np.asarray(train_risk,dtype=float).reshape(-1)
    center=float(np.mean(train_risk)); scale=float(np.std(train_risk,ddof=0))
    if not np.isfinite(scale) or scale<=0: raise ValueError(f"Degenerate training risk for {label}")
    calibration_frame=pd.DataFrame({"Time":train_frame["Time"].astype(float).to_numpy(),"Event":train_frame["Event"].astype(int).to_numpy(),"Risk_Score":(train_risk-center)/scale})
    calibrator=CoxPHFitter(penalizer=0.0,l1_ratio=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        calibrator.fit(calibration_frame,duration_col="Time",event_col="Event",show_progress=False)
    return {"label":label,"model":calibrator,"center":center,"scale":scale,"method":"train-fitted Cox recalibration of standardized raw model score"}

def predict_absolute_risk(calibrator, raw_risk, horizon):
    score=(np.asarray(raw_risk,dtype=float).reshape(-1)-calibrator["center"])/calibrator["scale"]
    survival=calibrator["model"].predict_survival_function(pd.DataFrame({"Risk_Score":score}),times=[float(horizon)])
    return np.clip(1.0-survival.iloc[0].to_numpy(dtype=float),0.0,1.0)

def prediction_matrix(calibrator, raw_risk, times):
    return np.column_stack([
        predict_absolute_risk(calibrator, raw_risk, float(time))
        for time in times
    ])


def fit_censoring_km(frame):
    km = KaplanMeierFitter()
    km.fit(
        frame["Time"].astype(float).to_numpy(),
        event_observed=1 - frame["Event"].astype(int).to_numpy(),
        label="G_censoring",
    )
    return km


def censor_survival(km, times, left_limit=False, minimum=1e-8):
    times = np.asarray(times, dtype=float)
    if left_limit:
        times = np.nextafter(times, -np.inf)
        times = np.maximum(times, 0.0)
    values = np.asarray(km.predict(times), dtype=float).reshape(-1)
    return np.clip(values, minimum, 1.0)


def horizon_support_audit(train_frame, eval_frame, horizons=HORIZONS):
    train_censoring_km = fit_censoring_km(train_frame)
    eval_censoring_km = fit_censoring_km(eval_frame)
    rows = []
    for horizon in horizons:
        horizon = float(horizon)
        train_time = train_frame["Time"].astype(float).to_numpy()
        train_event = train_frame["Event"].astype(int).to_numpy()
        eval_time = eval_frame["Time"].astype(float).to_numpy()
        eval_event = eval_frame["Event"].astype(int).to_numpy()
        rows.append({
            "Horizon_Years": horizon,
            "Train_Event_By_Horizon_N": int(((train_event == 1) & (train_time <= horizon)).sum()),
            "Train_Observed_Beyond_Horizon_N": int((train_time > horizon).sum()),
            "Test_Event_By_Horizon_N": int(((eval_event == 1) & (eval_time <= horizon)).sum()),
            "Test_Observed_Beyond_Horizon_N": int((eval_time > horizon).sum()),
            "Train_Censoring_Survival_Gt": float(censor_survival(train_censoring_km, [horizon])[0]),
            "Test_Censoring_Survival_Gt": float(censor_survival(eval_censoring_km, [horizon])[0]),
            "Train_Max_Followup": float(np.nanmax(train_time)),
            "Test_Max_Followup": float(np.nanmax(eval_time)),
        })
    audit = pd.DataFrame(rows)
    unsupported = audit.loc[
        (audit["Train_Event_By_Horizon_N"] == 0)
        | (audit["Train_Observed_Beyond_Horizon_N"] == 0)
        | (audit["Test_Event_By_Horizon_N"] == 0)
        | (audit["Test_Observed_Beyond_Horizon_N"] == 0)
        | (audit["Train_Censoring_Survival_Gt"] <= 1e-4)
        | (audit["Test_Censoring_Survival_Gt"] <= 1e-4)
    ]
    if not unsupported.empty:
        raise RuntimeError(
            "Insufficient event/censoring support at one or more requested horizons:\n"
            + unsupported.to_string(index=False)
        )
    return audit


def ipcw_nri_idi_once(
    eval_frame,
    pred_reference,
    pred_new,
    horizon,
    cutpoints=NRI_CUTPOINTS,
    censoring_km=None,
):
    pred_reference = np.asarray(pred_reference, dtype=float).reshape(-1)
    pred_new = np.asarray(pred_new, dtype=float).reshape(-1)
    time = eval_frame["Time"].astype(float).to_numpy()
    event = eval_frame["Event"].astype(int).to_numpy()
    horizon = float(horizon)
    km = fit_censoring_km(eval_frame) if censoring_km is None else censoring_km

    cases = (event == 1) & (time <= horizon)
    controls = time > horizon
    if cases.sum() == 0 or controls.sum() == 0:
        raise RuntimeError(f"No IPCW cases or controls at horizon={horizon}.")

    case_weights = np.zeros(len(eval_frame), dtype=float)
    control_weights = np.zeros(len(eval_frame), dtype=float)
    case_weights[cases] = 1.0 / censor_survival(km, time[cases], left_limit=True)
    g_horizon = float(censor_survival(km, [horizon])[0])
    control_weights[controls] = 1.0 / g_horizon

    case_weights /= case_weights.sum()
    control_weights /= control_weights.sum()
    delta_prediction = pred_new - pred_reference

    case_up = float(np.sum(case_weights * (delta_prediction > 0)))
    case_down = float(np.sum(case_weights * (delta_prediction < 0)))
    control_up = float(np.sum(control_weights * (delta_prediction > 0)))
    control_down = float(np.sum(control_weights * (delta_prediction < 0)))
    continuous_nri = (case_up - case_down) + (control_down - control_up)

    bins = np.asarray([-np.inf] + list(cutpoints) + [np.inf], dtype=float)
    category_reference = np.digitize(pred_reference, bins, right=False)
    category_new = np.digitize(pred_new, bins, right=False)
    category_delta = category_new - category_reference
    categorical_nri = (
        float(np.sum(case_weights * (category_delta > 0)))
        - float(np.sum(case_weights * (category_delta < 0)))
        + float(np.sum(control_weights * (category_delta < 0)))
        - float(np.sum(control_weights * (category_delta > 0)))
    )

    if not NRI_CUTPOINT_SOURCE:
        categorical_nri = np.nan

    discrimination_new = (
        float(np.sum(case_weights * pred_new))
        - float(np.sum(control_weights * pred_new))
    )
    discrimination_reference = (
        float(np.sum(case_weights * pred_reference))
        - float(np.sum(control_weights * pred_reference))
    )
    idi = discrimination_new - discrimination_reference
    return {
        "NRI_Categorical": categorical_nri,
        "NRI_Continuous": continuous_nri,
        "IDI": idi,
        "Censoring_Survival_At_Horizon": g_horizon,
        "Case_N": int(cases.sum()),
        "Control_N": int(controls.sum()),
        "Censored_Before_Horizon_N": int((~cases & ~controls).sum()),
    }


def bootstrap_ipcw_nri_idi(
    eval_frame,
    pred_reference,
    pred_new,
    horizon,
    cutpoints=NRI_CUTPOINTS,
    bootstrap_n=BOOTSTRAP_N,
    seed=BOOTSTRAP_SEED,
):
    original = ipcw_nri_idi_once(
        eval_frame,
        pred_reference,
        pred_new,
        horizon,
        cutpoints=cutpoints,
    )
    pred_reference = np.asarray(pred_reference, dtype=float).reshape(-1)
    pred_new = np.asarray(pred_new, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)
    n = len(eval_frame)
    values = {"NRI_Categorical": [], "NRI_Continuous": [], "IDI": []}
    for _ in range(int(bootstrap_n)):
        idx = rng.integers(0, n, size=n)
        boot_frame = eval_frame.iloc[idx].reset_index(drop=True)
        try:
            estimate = ipcw_nri_idi_once(
                boot_frame,
                pred_reference[idx],
                pred_new[idx],
                horizon,
                cutpoints=cutpoints,
            )
            for metric in values:
                values[metric].append(estimate[metric])
        except Exception:
            continue

    output = dict(original)
    for metric, metric_values in values.items():
        lower, upper = quantile_ci(metric_values)
        output[f"{metric}_Lower95"] = lower
        output[f"{metric}_Upper95"] = upper
    output["Bootstrap_N_Used"] = int(min(len(v) for v in values.values()))
    return output


def subject_level_ipcw_ibs(
    train_frame,
    eval_frame,
    calibrator,
    eval_risk,
    horizon,
    grid_n=IBS_GRID_N,
):
    horizon = float(horizon)
    times = np.linspace(0.0, horizon, int(grid_n))
    predicted_risk = prediction_matrix(calibrator, eval_risk, times)
    time = eval_frame["Time"].astype(float).to_numpy()
    event = eval_frame["Event"].astype(int).to_numpy()
    censor_km = fit_censoring_km(train_frame)
    contributions = np.zeros_like(predicted_risk, dtype=float)

    for j, point in enumerate(times):
        y = ((event == 1) & (time <= point)).astype(float)
        weights = np.zeros(len(eval_frame), dtype=float)
        cases = (event == 1) & (time <= point)
        controls = time > point
        if cases.any():
            weights[cases] = 1.0 / censor_survival(
                censor_km,
                time[cases],
                left_limit=True,
            )
        if controls.any():
            weights[controls] = 1.0 / float(censor_survival(censor_km, [point])[0])
        contributions[:, j] = weights * (y - predicted_risk[:, j]) ** 2

    if hasattr(np, "trapezoid"):
        subject_ibs = np.trapezoid(contributions, times, axis=1) / horizon
    else:
        subject_ibs = np.trapz(contributions, times, axis=1) / horizon
    return {
        "IBS": float(np.mean(subject_ibs)),
        "Subject_IBS": subject_ibs,
        "Times": times,
    }


def paired_bootstrap_delta_ibs(
    new_ibs,
    reference_ibs,
    bootstrap_n=BOOTSTRAP_N,
    seed=BOOTSTRAP_SEED,
):
    paired = (
        np.asarray(new_ibs["Subject_IBS"], dtype=float)
        - np.asarray(reference_ibs["Subject_IBS"], dtype=float)
    )
    rng = np.random.default_rng(seed)
    n = len(paired)
    values = []
    for _ in range(int(bootstrap_n)):
        idx = rng.integers(0, n, size=n)
        values.append(float(np.mean(paired[idx])))
    lower, upper = quantile_ci(values)
    return {
        "IBS_New": float(new_ibs["IBS"]),
        "IBS_Reference": float(reference_ibs["IBS"]),
        "Delta_IBS": float(np.mean(paired)),
        "Delta_IBS_Lower95": lower,
        "Delta_IBS_Upper95": upper,
        "Bootstrap_N_Used": int(len(values)),
    }


def km_observed_risk(frame, horizon):
    if len(frame) == 0:
        return np.nan
    km = KaplanMeierFitter()
    km.fit(
        frame["Time"].astype(float).to_numpy(),
        event_observed=frame["Event"].astype(int).to_numpy(),
    )
    return float(1.0 - km.predict(float(horizon)))


def calibration_by_quantile(eval_frame, predicted_risk, horizon, groups=10):
    temporary = eval_frame[["Time", "Event"]].reset_index(drop=True).copy()
    temporary["Predicted_Risk"] = np.asarray(predicted_risk, dtype=float).reshape(-1)
    temporary["Risk_Group"] = pd.qcut(
        temporary["Predicted_Risk"].rank(method="first"),
        q=int(groups),
        labels=False,
        duplicates="drop",
    )
    rows = []
    for group, data in temporary.groupby("Risk_Group"):
        rows.append({
            "Risk_Group": int(group) + 1,
            "N": int(len(data)),
            "Predicted_Risk": float(data["Predicted_Risk"].mean()),
            "Observed_KM_Risk": km_observed_risk(data, horizon),
        })
    return pd.DataFrame(rows)


def _safe_dca_column(name):
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return "model_" + text.lower()


def survival_dca(eval_frame, prediction_dict, horizon, thresholds=DCA_THRESHOLDS):
    data = eval_frame[["Time", "Event"]].reset_index(drop=True).copy()
    model_columns = []
    label_map = {}
    for label, prediction in prediction_dict.items():
        column = _safe_dca_column(label)
        data[column] = np.asarray(prediction, dtype=float).reshape(-1)
        model_columns.append(column)
        label_map[column] = label

    result = dcurves_dca(
        data=data,
        outcome="Event",
        modelnames=model_columns,
        thresholds=np.asarray(thresholds, dtype=float),
        time_to_outcome_col="Time",
        time=float(horizon),
    ).copy()
    result.columns = [str(column).lower() for column in result.columns]
    rename = {}
    for candidate in ["model", "models", "label"]:
        if candidate in result.columns:
            rename[candidate] = "Strategy"
            break
    for candidate in ["threshold", "thresholds"]:
        if candidate in result.columns:
            rename[candidate] = "Threshold"
            break
    for candidate in ["net_benefit", "net benefit", "netbenefit"]:
        if candidate in result.columns:
            rename[candidate] = "Net_Benefit"
            break
    result = result.rename(columns=rename)
    required = {"Strategy", "Threshold", "Net_Benefit"}
    if not required.issubset(result.columns):
        raise RuntimeError(
            "Unexpected dcurves output columns: " + ", ".join(result.columns)
        )
    result["Strategy"] = result["Strategy"].replace(label_map).replace({
        "all": "Treat all",
        "none": "Treat none",
        "All": "Treat all",
        "None": "Treat none",
    })
    result["Horizon_Years"] = float(horizon)
    result["Split"] = "test"
    result["DCA_Method"] = "dcurves survival DCA; no external IPCW multiplication"
    return result


def draw_forest(
    frame,
    estimate,
    lower,
    upper,
    label,
    title,
    xlabel,
    output_dir,
    stem,
    color="#3C5488",
    null=0.0,
    figsize=(8.2, 4.6),
):
    plot = frame.reset_index(drop=True).copy()
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(plot))
    ax.axvline(float(null), color=CNS["gray"], lw=1.0, zorder=0)
    for i, row in plot.iterrows():
        value = float(row[estimate])
        low = float(row[lower])
        high = float(row[upper])
        row_color = row[color] if isinstance(color, str) and color in plot.columns else color
        ax.errorbar(
            value,
            i,
            xerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
            fmt="o",
            color=row_color,
            ecolor=row_color,
            elinewidth=1.5,
            capsize=3,
            markersize=6.5,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot[label].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color=CNS["light_gray"], lw=0.8)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_dir, stem)
    plt.show()


def draw_dca(dca_frame, title, output_dir, stem, colors, figsize=(8.2, 5.0)):
    fig, ax = plt.subplots(figsize=figsize)
    preferred_order = list(colors)
    available = list(dict.fromkeys(dca_frame["Strategy"].astype(str)))
    order = [item for item in preferred_order if item in available]
    order += [item for item in available if item not in order]
    for strategy in order:
        data = dca_frame.loc[dca_frame["Strategy"].astype(str) == strategy].sort_values("Threshold")
        if data.empty:
            continue
        style = "--" if strategy == "Treat all" else ":" if strategy == "Treat none" else "-"
        width = 1.5 if strategy in {"Treat all", "Treat none"} else 2.1
        ax.plot(
            data["Threshold"],
            data["Net_Benefit"],
            color=colors.get(strategy, CNS["gray"]),
            linestyle=style,
            linewidth=width,
            label=strategy,
        )
    ax.set_xlabel("Predicted-risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(color=CNS["light_gray"], lw=0.8)
    ax.legend(frameon=False, fontsize=11.0, ncol=2)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_dir, stem)
    plt.show()
