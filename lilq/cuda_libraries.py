"""
The NVIDIA libraries a GPU process uses (Package 2, Stage 2, batch 8)
=====================================================================

The cluster environment (``scripts/cluster/env.sh``) loads the PyTorch module
(CUDA 12.6.0's libraries on ``LD_LIBRARY_PATH``), then the venv's PyTorch
2.10.0+cu126 wheel, whose own NVIDIA libraries (``site-packages/nvidia/*/lib``)
it finds through RUNPATH. ``LD_LIBRARY_PATH`` is searched before RUNPATH, so
some of the wheel's companions came from the module's older CUDA instead. On
Grace, with that mix, cuSOLVER's QR (``geqrf``) failed with an internal error
for a band of shapes around 11,036 x 3,675 (item 7's Kovasznay P = 3,675).
With the wheel's folders first on the path, every shape passed (job 20028701).

- ``wheel_library_dirs()``: the venv's ``nvidia/*/lib`` folders. ``env.sh``
  puts them first on ``LD_LIBRARY_PATH``
  (``python -m lilq.cuda_libraries --ld-path``).
- ``loaded_libraries()``: the NVIDIA libraries this process has mapped (from
  ``/proc/self/maps``; empty elsewhere), each with its resolved path.
- ``report()``: the libraries after a small cuBLAS and cuSOLVER call, the
  installed ``nvidia-*`` wheel versions, and, with ``probe``, whether given
  QR shapes factor (``python -m lilq.cuda_libraries --json <file>``).

Nothing here imports torch at module level: ``env.sh`` runs ``--ld-path`` in
every job, CPU-only ones included.
"""

import argparse
import glob
import json
import os
import sys
import sysconfig

_NVIDIA_NAMES = ('libcu', 'libnv', 'libcudnn', 'libnccl')


def wheel_library_dirs(purelib=None):
    """The ``nvidia/*/lib`` folders of the environment's site-packages, sorted."""
    purelib = purelib or sysconfig.get_paths()['purelib']
    return sorted(d for d in glob.glob(os.path.join(purelib, 'nvidia', '*', 'lib')) if os.path.isdir(d))


def loaded_libraries(maps_text=None):
    """``[{'path', 'resolved'}]``: the NVIDIA shared libraries mapped into this
    process (``maps_text`` for tests; ``/proc/self/maps`` otherwise)."""
    if maps_text is None:
        try:
            with open('/proc/self/maps') as f:
                maps_text = f.read()
        except OSError:
            return []
    paths = set()
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and '.so' in parts[5]:
            path = parts[5].strip()
            if os.path.basename(path).startswith(_NVIDIA_NAMES):
                paths.add(path)
    return [{'path': p, 'resolved': os.path.realpath(p)} for p in sorted(paths)]


def wheel_versions():
    """``{distribution: version}`` of the installed ``nvidia-*`` wheels."""
    from importlib import metadata
    return {d.metadata['Name']: d.version for d in metadata.distributions()
            if (d.metadata['Name'] or '').lower().startswith('nvidia-')}


def report(probe=()):
    """The libraries after a small cuBLAS and cuSOLVER call; with ``probe``,
    ``{'MxN': 'ok' | 'FAILED: ...'}`` for a random float64 matrix of each shape."""
    import torch
    out = {'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
           'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', '').split(':'),
           'wheel_library_dirs': wheel_library_dirs(), 'wheel_versions': wheel_versions()}
    if not torch.cuda.is_available():
        out['gpu'] = None
        return out
    out['gpu'] = torch.cuda.get_device_name(0)
    A = torch.randn(64, 32, dtype=torch.float64, device='cuda')
    torch.geqrf(A @ torch.eye(32, dtype=torch.float64, device='cuda'))       # loads cuBLAS and cuSOLVER
    torch.cuda.synchronize()
    out['loaded'] = loaded_libraries()
    probes = {}
    for m, n in probe:
        try:
            torch.geqrf(torch.randn(m, n, dtype=torch.float64, device='cuda'))
            torch.cuda.synchronize()
            probes[f'{m}x{n}'] = 'ok'
        except RuntimeError as e:
            probes[f'{m}x{n}'] = f'FAILED: {str(e)[:120]}'
    if probe:
        out['geqrf_probe'] = probes
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='The NVIDIA libraries of this environment.')
    ap.add_argument('--ld-path', action='store_true', help="print the wheel's nvidia/*/lib folders, ':'-joined")
    ap.add_argument('--json', help='write report() to this file')
    ap.add_argument('--probe', nargs='*', default=[], help='QR shapes to try, e.g. 11036x3675')
    args = ap.parse_args(argv)
    if args.ld_path:
        print(':'.join(wheel_library_dirs()))
        return
    r = report([tuple(int(v) for v in s.split('x')) for s in args.probe])
    text = json.dumps(r, indent=2)
    if args.json:
        with open(args.json, 'w') as f:
            f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    sys.exit(main())
