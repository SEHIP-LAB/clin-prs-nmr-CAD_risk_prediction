"""Write actual software/hardware metadata; run with the analysis interpreter."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file outside the public repository.')
    args = parser.parse_args()
    report = {
        'python': sys.version, 'platform': platform.platform(),
        'machine': platform.machine(), 'processor': platform.processor(),
        'packages': dict(sorted((d.metadata['Name'], d.version) for d in metadata.distributions() if d.metadata['Name'])),
    }
    try:
        import torch
        report['torch'] = {
            'version': torch.__version__, 'compiled_cuda': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        report['torch_error'] = str(exc)
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
            report['nvidia_smi'] = result.stdout or result.stderr
        except Exception as exc:
            report['nvidia_smi_error'] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(f'Wrote environment record: {args.output}')


if __name__ == '__main__':
    main()
