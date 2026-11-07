"""Unit checks for confirm_r3.py (v22).  python -m unittest test_confirm_r3.py"""
import json, unittest
from pathlib import Path
import numpy as np
import torch
import run
from confirm import generate
from confirm_r3 import conformal_threshold, ensemble_uncertainty, one_step_score, shift_detected

ROOT = Path(__file__).resolve().parent


class TruthModel(torch.nn.Module):
    """Returns the exact true acceleration (heterogeneous regime) -> zero one-step residual on undamped data."""
    def __init__(self, system):
        super().__init__(); self.system = system
    def forward(self, x):
        a = run.truth_np(x[..., :4].numpy(), x[..., 4:].numpy(), self.system, 'heterogeneous')
        return torch.tensor(a, dtype=torch.float32)


class TestR3(unittest.TestCase):
    def test_conformal_coverage(self):
        rng = np.random.default_rng(0); n, alpha, trials = 36, 0.1, 20000
        hits = 0
        for _ in range(trials):
            s = rng.normal(size=n + 1); tau = conformal_threshold(s[:n], alpha); hits += s[n] > tau
        self.assertLessEqual(hits / trials, alpha + 0.01)
        self.assertEqual(conformal_threshold(np.arange(36), 0.05), 35.0)          # ceil(37*0.95)=36 -> max
        self.assertEqual(conformal_threshold(np.arange(10), 0.01), float('inf'))   # k > n

    def test_ensemble_uncertainty(self):
        p = np.zeros((3, 4, 4)); self.assertTrue(np.allclose(ensemble_uncertainty([p, p, p]), 0))
        q = p.copy(); q[1] += 2.0
        u = ensemble_uncertainty([p, q]); self.assertEqual(u.shape, (3,)); self.assertGreater(u[1], 0); self.assertEqual(u[0], 0)

    def test_one_step_score(self):
        cfg = json.loads((ROOT / 'confirm_config.json').read_text()); dt = cfg['dt']
        m = TruthModel('duffing')
        clean = generate(9999, 6, 10, 'duffing', dt, 1.0, 'heterogeneous')
        damped = generate(9999, 6, 10, 'duffing', dt, 1.0, 'heterogeneous', extra_damping=0.3)
        s0 = one_step_score(m, clean, 5, dt, 'cpu'); s1 = one_step_score(m, damped, 5, dt, 'cpu')
        self.assertLess(s0.max(), 1e-6)            # exact model, unchanged dynamics -> ~0 (float32)
        self.assertGreater(s1.min(), 1e-4)         # extra damping shows up in the one-step residual

    def test_shift_rule_false_alarm(self):
        from scipy.stats import binom
        self.assertLess(binom.sf(2, 8, 0.05), 0.01)  # P(X>=3), X~Bin(8, 0.05): ID false alarm bound < 1%
        self.assertEqual(shift_detected([0, 0, 2, 2, 2, 0, 0, 0, 9], 1.0, 8, 3), (True, 3))
        self.assertEqual(shift_detected([0, 0, 2, 2, 0, 0, 0, 0, 9], 1.0, 8, 3), (False, 2))

    def test_lock_covers_files(self):
        lock = ROOT / 'confirm_r3_protocol_lock.json'
        if lock.exists():
            L = json.loads(lock.read_text())
            for f in ('confirm_r3_config.json', 'confirm_r3.py', 'confirm.py', 'run.py', 'calibration.py', 'confirm_config.json'):
                self.assertIn(f, L['sha256'])


if __name__ == '__main__':
    unittest.main()
