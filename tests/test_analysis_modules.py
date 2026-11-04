"""Small synthetic integration checks; no participant data or GPU required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
try:
    from joblib.externals import cloudpickle
except ImportError:
    import cloudpickle

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_test_metrics', ROOT / 'sources/metrics.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
sys.modules[spec.name] = M
cloudpickle.register_pickle_by_value(M)


def synthetic_frame(n, seed, columns=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, columns))
    event_time = rng.exponential(9 * np.exp(-0.7*x[:, 0])) + .01
    censor_time = rng.uniform(8, 22, size=n)
    frame = pd.DataFrame(x, columns=[f'm{i:03}' for i in range(columns)])
    frame['Eid'] = np.arange(seed*10000, seed*10000+n)
    frame['Event'] = (event_time < censor_time).astype(int)
    frame['Time'] = np.minimum(event_time, censor_time)
    return frame


def score_context(folder, workers=1):
    return dict(
        AUDIT_DIR=str(folder), CPU_WORKERS=workers, NMR_BOUNDARY_POLICY='warn',
        NMR_ELASTICNET_ALPHA_GRID=[.1, .05, .02],
        NMR_ELASTICNET_L1_RATIOS=[.5, 1.0], NMR_ELASTICNET_INNER_CV=5,
        NMR_ELASTICNET_MAX_ITER=100000, RANDOM_STATE=2026,
        atomic_write_csv=lambda frame,path,index=False: frame.to_csv(path,index=index),
        canonical_json_digest=lambda value: hashlib.sha256(json.dumps(value).encode()).hexdigest(),
        eid_digest=lambda frame: hashlib.sha256(frame.Eid.to_numpy().tobytes()).hexdigest(),
        log_progress=lambda *args: None,
        nmr106_cols=[f'm{i:03}' for i in range(106)],
    )


def load_score(context):
    path = ROOT / 'sources/score106.py'
    exec(compile(path.read_text(), str(path), 'exec'), context)
    return context


class MetricIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = synthetic_frame(300, 31)
        cls.test = synthetic_frame(150, 32)
        cls.risk = cls.test.m000.to_numpy()
        cls.calibrator = M.fit_absolute_risk_calibrator(cls.train, cls.train.m000, 'synthetic')

    def test_calibration_and_identical_model_differences(self):
        p = M.predict_absolute_risk(self.calibrator, self.risk, 5)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((0 <= p) & (p <= 1)).all())
        result = M.ipcw_nri_idi_once(self.test, p, p, 5)
        self.assertEqual(result['NRI_Continuous'], 0)
        self.assertEqual(result['IDI'], 0)
        uno = M.paired_bootstrap_uno(self.train, self.test, self.risk, self.risk, 5, bootstrap_n=12)
        for key in ['Delta_Uno_C', 'Delta_Uno_C_Lower95', 'Delta_Uno_C_Upper95']:
            self.assertEqual(uno[key], 0)
        self.assertEqual(uno['Bootstrap_N_Used'], 12)

    def test_ibs_support_calibration_and_dca_execute(self):
        ibs = M.subject_level_ipcw_ibs(self.train, self.test, self.calibrator, self.risk, 5, grid_n=11)
        self.assertEqual(len(ibs['Subject_IBS']), len(self.test))
        self.assertTrue(np.isfinite(ibs['Subject_IBS']).all())
        self.assertTrue((ibs['Subject_IBS'] >= 0).all())
        delta = M.paired_bootstrap_delta_ibs(ibs, ibs, bootstrap_n=12)
        for key in ['Delta_IBS', 'Delta_IBS_Lower95', 'Delta_IBS_Upper95']:
            self.assertEqual(delta[key], 0)
        audit = M.horizon_support_audit(self.train, self.test, horizons=[5])
        self.assertEqual(len(audit), 1)
        p = M.predict_absolute_risk(self.calibrator, self.risk, 5)
        calibration = M.calibration_by_quantile(self.test, p, 5, groups=5)
        self.assertEqual(len(calibration), 5)
        dca = M.survival_dca(self.test, {'synthetic': p}, 5, thresholds=np.array([.05,.1,.2]))
        self.assertFalse(dca.empty)

    def test_dynamic_module_works_in_process_workers(self):
        expected = M.uno_c(self.train, self.test, self.risk, 5)
        actual = Parallel(n_jobs=2, backend='loky')(
            delayed(M.uno_c)(self.train, self.test, self.risk, 5) for _ in range(2))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


class Raw106IntegrationTests(unittest.TestCase):
    def test_missing_shared_context_reports_all_required_names(self):
        with self.assertRaisesRegex(RuntimeError, 'Missing: .*AUDIT_DIR.*nmr106_cols'):
            load_score({})

    def test_five_fold_training_does_not_use_held_out_values(self):
        train = synthetic_frame(250, 41, columns=106)
        val = synthetic_frame(80, 42, columns=106)
        test = synthetic_frame(90, 43, columns=106)
        # Exercise fold-specific imputation, including missing held-out entries.
        train.loc[::13, 'm001'] = np.nan
        val.loc[::7, 'm001'] = np.nan
        with tempfile.TemporaryDirectory() as temp:
            ns = load_score(score_context(Path(temp), workers=2))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='NMR_score optimum is on a search boundary')
                first = ns['fit_nmr_score_nested'](train, val, test)
                changed_val, changed_test = val.copy(), test.copy()
                changed_val['m000'] += 10
                changed_test['m000'] -= 10
                changed_test['Event'] = 1-changed_test.Event
                second = ns['fit_nmr_score_nested'](train, changed_val, changed_test)
            self.assertEqual(len(first), 8)
            for index, expected_length in [(0,len(train)),(1,len(val)),(2,len(test))]:
                self.assertEqual(len(first[index]), expected_length)
                self.assertTrue(np.isfinite(first[index]).all())
            pd.testing.assert_frame_equal(first[3], second[3])
            pd.testing.assert_frame_equal(first[4], second[4])
            np.testing.assert_array_equal(first[0], second[0])
            self.assertEqual(first[6], second[6])
            selected = first[3].loc[first[3]['selected']]
            self.assertEqual(selected.valid_folds.tolist(), [5])
            self.assertEqual(first[6]['predictor_field_count'], 106)
            self.assertFalse(first[6]['validation_used_for_score_selection'])
            self.assertFalse(first[6]['test_used_for_score_selection'])


if __name__ == '__main__':
    unittest.main()
