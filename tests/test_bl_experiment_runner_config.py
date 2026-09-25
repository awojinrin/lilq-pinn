"""Regression guard: run_bl.py must not pass an explicit
max_line_searches, so every method gets the uniform cap of
lilq.solvers.line_search_cap (3x its own iteration budget; DECISIONS.md,
2026-09-24). Checked at the source level: executing run_experiment_for_N
would do real file I/O and training.
"""

import sys

import experiments.run_bl as run_bl_module


def test_paper_setup_leaves_the_line_search_cap_to_the_uniform_rule():
    for gravity in (False, True):
        for N in (8, 32):
            _config, opt = run_bl_module.paper_setup(N, gravity)
            assert opt.max_line_searches is None


def test_gravity_basis_constant_is_cos_fourier():
    """DECISIONS.md, 2026-09-23 "gravity IC needs a better-suited basis"
    entry: 'fourier' (mode_x='both') wastes half the spatial resolution
    on sine modes that don't help represent the gravity IC's monotonic
    steep step. 'cos_fourier' (full cosine resolution in x) took LiL-N
    from not converging at all at N=24/32 to converging in seconds,
    verified against a real run -- this constant is what makes that the
    default for every gravity config built by this script."""
    assert run_bl_module.GRAVITY_BASIS == 'cos_fourier'


def _run_main_capturing_basis(monkeypatch, argv):
    """Run main() with run_experiment_for_N replaced by a stub that just
    records the `basis` it was called with, and every save_*/plot_* side
    effect stubbed out -- verifies main()'s CLI basis-resolution logic
    for real (not by grepping its source) without touching disk or
    triggering real training."""
    captured = {}

    def fake_run_experiment_for_N(N, basis, gravity, methods, verbose, seeds=None):
        captured['basis'] = basis
        return {}

    monkeypatch.setattr(run_bl_module, 'run_experiment_for_N', fake_run_experiment_for_N)
    monkeypatch.setattr(run_bl_module, 'print_summary_table', lambda *a, **k: None)
    monkeypatch.setattr(run_bl_module, 'experiment_base_dir', lambda *a, **k: __import__('pathlib').Path('.'))
    monkeypatch.setattr(run_bl_module, 'save_master_results', lambda *a, **k: None)
    monkeypatch.setattr(run_bl_module, 'save_run_provenance', lambda *a, **k: None)
    monkeypatch.setattr(run_bl_module, 'make_figures_dir', lambda *a, **k: __import__('pathlib').Path('.'))
    monkeypatch.setattr(run_bl_module, 'plot_convergence_by_method', lambda *a, **k: None)
    monkeypatch.setattr(run_bl_module, 'plot_convergence_by_size', lambda *a, **k: None)
    monkeypatch.setattr(sys, 'argv', ['run_bl.py'] + argv)

    run_bl_module.main()
    return captured['basis']


def test_cli_defaults_to_cos_fourier_for_gravity(monkeypatch):
    basis = _run_main_capturing_basis(monkeypatch, ['--gravity', '--N', '8'])
    assert basis == 'cos_fourier'


def test_cli_defaults_to_fourier_for_viscous(monkeypatch):
    basis = _run_main_capturing_basis(monkeypatch, ['--N', '8'])
    assert basis == 'fourier'


def test_cli_explicit_basis_overrides_gravity_default(monkeypatch):
    basis = _run_main_capturing_basis(monkeypatch, ['--gravity', '--N', '8', '--basis', 'chebyshev'])
    assert basis == 'chebyshev'
