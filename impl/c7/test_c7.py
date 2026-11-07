"""Unit tests for C7 (run from impl/: python3 -c "import sys; sys.path.insert(0,'c7'); import test_c7 as t; ...")."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / 'c1', HERE):
    sys.path.insert(0, str(p))
from c7_rules import betting_ucb, magnitude_gate
from c7_study import directions
from c1_models import System, apply, kinematic


def test_betting_ucb_coverage():
    rng = np.random.default_rng(1); miss = 0
    for _ in range(300):
        x = rng.beta(2, 5, 100); miss += betting_ucb(x, 0.05) < 2 / 7
    assert miss / 300 <= 0.05


def test_gate_rejects_harm_accepts_clear_gain():
    assert not magnitude_gate(np.full(100, 1.2))[0]
    assert magnitude_gate(np.full(100, 0.6))[0]


def test_partial_matrices_are_contractions_and_keep_constraints():
    rng = np.random.default_rng(2); d = 7
    Q, _ = np.linalg.qr(rng.normal(size=(d, d))); A = Q @ np.diag(rng.uniform(0, 1, d)) @ Q.T
    rv = rng.normal(size=(50, d)); ev = rng.normal(size=(50, d))
    pref = directions(A, 'matrix', rv, ev, 1e-9)
    assert len(pref) == d and np.allclose(pref[-1], A, atol=1e-10)
    for a in pref:
        w = np.linalg.eigvalsh((a + a.T) / 2)
        assert np.allclose(a, a.T) and w.min() >= -1e-10 and w.max() <= 1 + 1e-10
    C = json.loads((HERE.parent / 'c1' / 'c1_config.json').read_text())
    S = System('franka_7dof', C['systems']['franka_7dof'])
    s, u = S.episodes(3, 2, 10, C['dt'], C['truth_substeps']); p = S.rollout(s[:, 0], u, C['dt'])
    z = S.rollout(s[:, 0], u, C['dt'], lambda x: 0.2 * np.cos(x[:, :d]))
    for a in pref:
        assert kinematic(apply(p, z, a, d), s[:, 0], C['dt'], d) < 1e-9


def test_direction_order_uses_selection_data_only():
    rng = np.random.default_rng(3); d = 3
    A = np.diag([0.2, 0.9, 0.5]); rv = rng.normal(size=(40, d)); ev = rv @ np.diag([0.2, 1.0, 1.0])
    pref = directions(A, 'diagonal', rv, ev, 1e-9)
    assert np.isclose(pref[0][0, 0], 0.2)       # the direction that fits C_s1 best comes first


def test_seeds_match_locked_plan():
    import tomllib
    plan = tomllib.loads((HERE.parent.parent / 'design' / 'C7' / 'plan.toml').read_text())
    assert json.loads((HERE / 'c7_config.json').read_text())['seeds'] == plan['seeds']['values']
