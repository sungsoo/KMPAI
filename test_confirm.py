"""Checks for confirm.py (v14). Run: python -m unittest test_confirm.py"""
import unittest
import numpy as np
import confirm


class ConfirmTests(unittest.TestCase):
    def test_clopper_pearson_upper(self):
        # k=0, n=36, eps=.05: 1-0.05**(1/36) ~= 0.0799 <= 0.1 (applies); k=1 exceeds 0.1 (rejects)
        self.assertAlmostEqual(confirm.cp_upper(0, 36, .05), 1 - 0.05 ** (1 / 36), places=6)
        self.assertLess(confirm.cp_upper(0, 36, .05), .1)
        self.assertGreater(confirm.cp_upper(1, 36, .05), .1)
        self.assertEqual(confirm.cp_upper(36, 36, .05), 1.0)
        self.assertTrue(all(confirm.cp_upper(k, 36, .05) < confirm.cp_upper(k + 1, 36, .05) for k in range(35)))

    def test_linear_constraint_preserved_for_any_A(self):
        rng = np.random.default_rng(0); dt = .05; n, h = 3, 20
        def roll(acc):
            s0 = rng.normal(size=(n, 4)); out = np.empty((n, h, 4)); cur = s0.copy()
            for t in range(h):
                v = cur[:, 2:] + dt * acc[:, t]; cur = np.concatenate([cur[:, :2] + dt * v, v], -1); out[:, t] = cur
            return s0, out
        s0, p = roll(rng.normal(size=(n, h, 2)))
        rng2 = np.random.default_rng(0); z = np.empty_like(p); cur = s0.copy()
        acc = rng2.normal(size=(n, h, 2)) * 2
        for t in range(h):
            v = cur[:, 2:] + dt * acc[:, t]; cur = np.concatenate([cur[:, :2] + dt * v, v], -1); z[:, t] = cur
        for a in [np.eye(2), np.zeros((2, 2)), rng.normal(size=(2, 2))]:
            self.assertLess(confirm.kinematic(confirm.apply(p, z, a), s0, dt), 1e-12)
        # time-varying (per-step) weights break the constraint
        w = np.linspace(0, 1, h)[None, :, None]
        self.assertGreater(confirm.kinematic(p + (z - p) * w, s0, dt), 1e-6)

    def test_generate_damping_changes_dynamics_only(self):
        a = confirm.generate(1, 2, 10, 'duffing', .05, 1., 'heterogeneous', 0.0)
        b = confirm.generate(1, 2, 10, 'duffing', .05, 1., 'heterogeneous', 0.3)
        np.testing.assert_array_equal(a[1], b[1]); np.testing.assert_array_equal(a[0][:, 0], b[0][:, 0])
        self.assertGreater(np.abs(a[0] - b[0]).max(), 0)


if __name__ == '__main__':
    unittest.main()
