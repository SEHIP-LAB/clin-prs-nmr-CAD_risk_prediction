"""Synthetic checks for serialization, source-driven plotting and dispatch paths."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def source(name, cell):
    notebook = json.loads((ROOT / 'notebooks' / name).read_text())
    return ''.join(notebook['cells'][cell]['source'])


def shared_utilities():
    names = {'sha256_file', 'eid_digest', 'dataframe_digest',
             'atomic_write_csv', 'atomic_write_json'}
    notebook = json.loads((ROOT / 'notebooks' / '00_shared_config.ipynb').read_text())
    definitions = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            definitions.extend(n for n in ast.parse(''.join(cell['source'])).body
                               if isinstance(n, ast.FunctionDef) and n.name in names)
    namespace = dict(np=np, pd=pd, hashlib=hashlib, json=json, Path=Path, os=os, uuid=uuid)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<shared utilities>', 'exec'), namespace)
    return namespace


class SplitPersistenceTests(unittest.TestCase):
    def execute_persistence(self, folder, mutation=None):
        ns = shared_utilities()
        frame = pd.DataFrame({
            'Eid': np.arange(900001, 900009),
            'Time': [1.2345678901234567, 10.123456789012345, 3.25, 4.5,
                     6.314159265358979, 7.1, 12.999999999999998, .01234567890123456],
            'Event': [0, 1, 0, 1, 0, 1, 0, 1],
            'Age': np.arange(50, 58), 'PRS': np.linspace(-1, 1, 8),
            '23400': np.linspace(0.1, 0.8, 8),
        })
        original_writer = ns['atomic_write_csv']
        if mutation:
            def writer(data, path, index=False):
                data = data.copy()
                if 'Time' in data:
                    mutation(data)
                original_writer(data, path, index=index)
            ns['atomic_write_csv'] = writer
        ns.update(df_train=frame.copy(), df_val=frame.copy(), df_test=frame.copy(),
                  final_feature_sets={'Clinical_only': ['Age']},
                  FEATURE_SETS_PATH=folder / 'features.json',
                  PIPELINE_ARTIFACT_DIR=str(folder),
                  sample_summary_df=pd.DataFrame({'N': [8]}),
                  clinical_imputation_cols=['Age'], prs_col='PRS', nmr106_cols=['23400'],
                  split_artifact_path=lambda split: folder / f'{split}.csv')
        code = source('01_cohort_and_nmr_derivation.ipynb', 8)
        code = code[:code.index('if any(item["prs_missing"]')]
        exec(compile(code, '<split persistence>', 'exec'), ns)
        return ns

    def test_manifest_hashes_match_saved_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            ns = self.execute_persistence(Path(temp))
            for split, record in ns['manifest_splits'].items():
                saved = pd.read_csv(Path(temp) / f'{split}.csv', low_memory=False)
                self.assertEqual(record['outcome_sha256'],
                                 ns['dataframe_digest'](saved, ['Eid', 'Time', 'Event']))
                self.assertEqual(record['eid_sha256'], ns['eid_digest'](saved))

    def test_changed_followup_time_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, 'Time changed'):
                self.execute_persistence(Path(temp), lambda d: d.__setitem__('Time', d.Time + 0.01))

    def test_changed_identifier_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, 'population changed'):
                self.execute_persistence(Path(temp), lambda d: d.__setitem__('Eid', d.Eid.to_numpy()[::-1]))


class RepresentationFigureTests(unittest.TestCase):
    def render(self, root, counts):
        features = ['Clinical_plus_PRS_plus_NMR_score',
                    'Clinical_plus_PRS_plus_SpearmanNMR',
                    'Clinical_plus_PRS_plus_Raw106NMR']
        table = root / 'figure3A_representation_scale.csv'
        pd.DataFrame({'Feature_Set': features, 'NMR_Predictor_N': counts}).to_csv(table, index=False)
        (root / '01_cohort').mkdir()
        (root / '01_cohort' / 'pipeline_manifest.json').write_text(
            json.dumps({'nmr_score': {'predictor_contract_name': 'Raw106'}}))
        captured = []
        ns = dict(pd=pd, np=np, plt=plt, json=json, RUN_ROOT=root, TABLE_DIR=root,
                  find_unique=lambda name: table,
                  require_columns=lambda d, cols, path: d[cols],
                  save_show=lambda fig, stem: captured.append(fig),
                  CNS={'green': '#00A087', 'purple': '#8491B4', 'red': '#E64B35',
                       'light_gray': '#D9D9D9'})
        exec(compile(source('16_manuscript_figures.ipynb', 22), '<dimension figure>', 'exec'), ns)
        return ns, captured

    def tearDown(self):
        plt.close('all')

    def test_dimensions_and_panel_name_come_from_active_run(self):
        with tempfile.TemporaryDirectory() as temp:
            ns, figures = self.render(Path(temp), [1, 47, 106])
            self.assertEqual(ns['result3_scale']['Final_Model_NMR_Input_Dimension'].tolist(), [1, 47, 106])
            self.assertEqual(len(figures), 1)
            self.assertEqual([bar.get_height() for bar in figures[0].axes[0].patches], [1, 47, 106])
            self.assertIn('Raw106 candidate', '\n'.join(t.get_text() for t in figures[0].axes[0].texts))

    def test_invalid_dimensions_raise(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'positive integers'):
                self.render(Path(temp), [1, 47.5, 106])


class ResumeDispatchTests(unittest.TestCase):
    def test_raw106_resume_uses_environment_and_renamed_targets(self):
        calls = []
        class Shell:
            def run_line_magic(self, magic, value):
                self.assertion = magic == 'run'
                calls.append(Path(value.strip('"')))
        shell = Shell()
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            'CHD_OUTPUT_ROOT': temp, 'CHD_SCORE106_RUN': 'synthetic_score106',
            'CHD_SCORE106_COHORT': 'coverage90',
        }, clear=False):
            ns = {'get_ipython': lambda: shell, 'print': lambda *a, **k: None}
            with patch.object(Path, 'cwd', return_value=ROOT / 'notebooks'):
                exec(compile(source('12_nmr_score106_derivation.ipynb', 3), '<resume>', 'exec'), ns)
            self.assertEqual(os.environ['CHD_RUN_ID'], 'synthetic_score106')
            self.assertEqual(os.environ['CHD_COHORT_ID'], 'coverage90')
            self.assertEqual(Path(os.environ['CHD_SENSITIVITY_OUTPUT_DIR']), Path(temp) / 'runs' / 'synthetic_score106')
            self.assertEqual(len(calls), 4)
            self.assertTrue(all(path.is_file() for path in calls))
            self.assertEqual([p.name[:2] for p in calls], ['02', '03', '04', '05'])


if __name__ == '__main__':
    unittest.main()
