"""Check notebook packaging without executing scientific analyses."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def check(require_modules=True):
    errors = []
    notebooks = sorted((ROOT / 'notebooks').glob('*.ipynb'))
    available = {p.name for p in notebooks}
    code_cells = 0
    reference_count = 0
    for path in notebooks:
        if not re.fullmatch(r'\d{2}_[a-z0-9_]+\.ipynb', path.name):
            errors.append(f'{path.name}: nonstandard filename')
        notebook = json.loads(path.read_text(encoding='utf-8'))
        try:
            nbformat.validate(nbformat.from_dict(notebook))
        except Exception as exc:
            errors.append(f'{path.name}: invalid notebook: {exc}')
        ids = [c.get('id') for c in notebook['cells']]
        if None in ids or len(ids) != len(set(ids)):
            errors.append(f'{path.name}: missing or duplicate cell IDs')
        if notebook.get('nbformat') != 4:
            errors.append(f'{path.name}: unsupported notebook format')
        for i, cell in enumerate(notebook['cells']):
            source = ''.join(cell['source'])
            if re.search(r'[\u4e00-\u9fff]', source):
                errors.append(f'{path.name}:{i}: Chinese text remains')
            if cell.get('attachments'):
                errors.append(f'{path.name}:{i}: embedded attachments')
            if cell['cell_type'] != 'code':
                continue
            code_cells += 1
            if cell.get('outputs') or cell.get('execution_count') is not None:
                errors.append(f'{path.name}:{i}: execution state remains')
            try:
                tree = ast.parse(source)
                compile(tree, f'{path.name}:{i}', 'exec')
            except SyntaxError as exc:
                errors.append(f'{path.name}:{i}: {exc}')
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for name in re.findall(r'[A-Za-z0-9_().-]+\.ipynb', node.value):
                        reference_count += 1
                        if name not in available:
                            errors.append(f'{path.name}:{i}: unresolved notebook {name}')
    missing = [f'sources/{name}' for name in ['score106.py', 'metrics.py']
               if not (ROOT / 'sources' / name).is_file()]
    print(f'Notebooks: {len(notebooks)}; code cells: {code_cells}; notebook references: {reference_count}')
    for module in missing:
        print(f'MISSING ANALYSIS MODULE: {module}')
    if missing:
        errors.append('Both original analysis modules are required.')
    python_files = sorted(ROOT.glob('sources/*.py')) + sorted(ROOT.glob('scripts/*.py')) + sorted(ROOT.glob('tests/*.py'))
    for path in python_files:
        try:
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        except SyntaxError as exc:
            errors.append(f'{path.relative_to(ROOT)}: {exc}')
    print(f'Python files: {len(python_files)}; required analysis modules: {2-len(missing)}/2')
    for error in errors:
        print(f'ERROR: {error}', file=sys.stderr)
    print('FAIL' if errors else 'PASS: packaging checks only; no scientific execution')
    return bool(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-modules', action='store_true', help='Compatibility option: modules are always required.')
    args = parser.parse_args()
    sys.exit(check(args.require_modules))
