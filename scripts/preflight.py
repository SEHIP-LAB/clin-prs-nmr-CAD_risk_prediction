"""Report missing prerequisites together, before starting a long analysis.

This reads input headers only when --data-dir is supplied. It does not train
models, create analysis directories, or validate already generated results.
"""
import argparse
import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
IMPORTS = {
    'cloudpickle': 'cloudpickle', 'adjustText': 'adjustText', 'dcurves': 'dcurves', 'ipykernel': 'ipykernel',
    'ipython': 'IPython', 'joblib': 'joblib', 'jupyterlab': 'jupyterlab',
    'lifelines': 'lifelines', 'matplotlib': 'matplotlib', 'nbformat': 'nbformat',
    'numpy': 'numpy', 'pandas': 'pandas', 'scikit-learn': 'sklearn',
    'scikit-survival': 'sksurv', 'scipy': 'scipy',
    'threadpoolctl': 'threadpoolctl', 'torch': 'torch', 'xgboost': 'xgboost',
}
METRIC_API = (
    'BOOTSTRAP_SEED', 'NRI_CUTPOINTS', 'uno_c', 'quantile_ci',
    'horizon_support_audit', 'fit_absolute_risk_calibrator',
    'predict_absolute_risk', 'subject_level_ipcw_ibs',
    'paired_bootstrap_delta_ibs', 'ipcw_nri_idi_once',
    'km_observed_risk', 'calibration_by_quantile', 'survival_dca',
)


def input_names():
    notebook = json.loads((ROOT / 'notebooks/00_shared_config.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'file_names' for t in node.targets
            ):
                return ast.literal_eval(node.value)
    raise RuntimeError('Input contract file_names was not found.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Authorized input directory; check all seven CSV headers.')
    parser.add_argument('--ethnicity-csv', type=Path, help='Optional separate Field 21000 CSV.')
    parser.add_argument('--require-ethnicity', action='store_true', help='Require baseline Field 21000 for notebooks 14–15.')
    parser.add_argument('--require-cuda', action='store_true', help='Also require a visible PyTorch CUDA device.')
    args = parser.parse_args()
    errors, notes = [], []
    for package, module in IMPORTS.items():
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f'{package}: {type(exc).__name__}: {exc}')
    for name in ('metrics.py', 'score106.py'):
        path = ROOT / 'sources' / name
        if not path.is_file():
            errors.append(f'Missing source: {path}')
        else:
            try:
                compile(path.read_text(), str(path), 'exec')
            except Exception as exc:
                errors.append(f'{name}: {exc}')
    try:
        spec = importlib.util.spec_from_file_location('preflight_metrics', ROOT / 'sources/metrics.py')
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)
        missing = [name for name in METRIC_API if not hasattr(metrics, name)]
        if missing:
            errors.append(f'Missing metrics API: {missing}')
    except Exception as exc:
        errors.append(f'metrics import: {type(exc).__name__}: {exc}')
    if args.require_cuda:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append('PyTorch cannot see CUDA. For CPU fallback use CHD_GPU_REQUIRED=0 and omit --require-cuda.')
        except Exception as exc:
            errors.append(f'CUDA availability: {exc}')
    ethnicity_found = False
    paths = []
    if args.data_dir:
        paths.extend(args.data_dir / name for name in input_names())
    ethnicity_path = args.ethnicity_csv or os.environ.get('CHD_ETHNICITY_CSV')
    if ethnicity_path:
        paths.append(Path(ethnicity_path))
    for path in paths:
        if not path.is_file():
            errors.append(f'Missing input: {path}')
            continue
        try:
            import pandas as pd
            columns = set(pd.read_csv(path, nrows=0).columns)
            ethnicity_found |= bool(columns.intersection(
                {'21000','21000-0.0','21000_0_0','p21000_i0','p21000_i0_a0'}))
        except Exception as exc:
            errors.append(f'Cannot read input header {path}: {exc}')
    if args.require_ethnicity and not ethnicity_found:
        errors.append('Baseline Field 21000 not found. Supply --data-dir and/or --ethnicity-csv.')
    if not args.data_dir:
        notes.append('Input data not checked: pass --data-dir to check the seven required CSVs.')
    if not os.environ.get('CHD_NRI_CUTPOINT_SOURCE', '').strip():
        notes.append('Categorical NRI is disabled until CHD_NRI_CUTPOINT_SOURCE is provided; continuous NRI and IDI remain active.')
    for note in notes:
        print(f'NOTE: {note}')
    for error in errors:
        print(f'MISSING/ERROR: {error}', file=sys.stderr)
    print(f'Preflight: {len(errors)} error(s). No scientific analyses were executed.')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
