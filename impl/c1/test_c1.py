"""Unit tests for C1 (python -m pytest -q c1/test_c1.py from impl/)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from c1_models import System, stack, apply, kinematic
C = json.loads((HERE / 'c1_config.json').read_text())


def test_systems_load_and_dimensions():
    for name, spec in C['systems'].items():
        S = System(name, spec)
        assert S.d == {'planar_2link': 2, 'planar_3link': 3, 'franka_7dof': 7}[name]


def test_truth_differs_from_prior_and_data_deterministic():
    S = System('planar_3link', C['systems']['planar_3link'])
    a, ua = S.episodes(5, 2, 20, C['dt'], C['truth_substeps']); b, ub = S.episodes(5, 2, 20, C['dt'], C['truth_substeps'])
    assert np.array_equal(a, b) and np.array_equal(ua, ub)
    p = S.rollout(a[:, 0], ua, C['dt']); assert np.abs(p - a[:, 1:]).max() > 1e-4


def test_proposition1_linear_constraints_any_matrix():
    """Discrete kinematics shared by P and Z hold for P + R A with any A (d = 7)."""
    S = System('franka_7dof', C['systems']['franka_7dof']); d = S.d
    s, u = S.episodes(3, 3, 15, C['dt'], C['truth_substeps'])
    p = S.rollout(s[:, 0], u, C['dt']); z = S.rollout(s[:, 0], u, C['dt'], lambda x: 0.3 * np.sin(x[:, :d]))
    A = np.random.default_rng(0).normal(size=(d, d))
    assert kinematic(apply(p, z, A, d), s[:, 0], C['dt'], d) < 1e-9


def test_stack_apply_shapes():
    d = 3; x = np.random.default_rng(1).normal(size=(4, 5, 2 * d))
    assert stack(x, d).shape == (4 * 5 * 2, d)
    assert np.allclose(apply(x, x, np.eye(d), d), x) and np.allclose(apply(x, 2 * x, np.zeros((d, d)), d), x)


def test_seeds_match_locked_plan():
    import tomllib
    plan = tomllib.loads((HERE.parent.parent / 'design' / 'C1' / 'plan.toml').read_text())
    assert C['seeds'] == plan['seeds']['values']
