"""C1 systems (design/C1/plan.toml): planar 2-link, planar 3-link and Franka Panda 7-DOF arms in MuJoCo.

Physics prior P = nominal MuJoCo model. Truth Y = the same tree with link-mass error, extra viscous damping,
smooth Coulomb friction and neighbour coupling (a "missing force" the prior does not know). Every forecaster
(P, the learned world model Z, the truth) receives the same control law: gravity compensation computed from the
NOMINAL model plus a stabilising PD term around q_home, both at the forecaster's own state, plus an exogenous torque
u(t) (smoothly varying sinusoids), so u is the action and the controller is part of each roll-out. P and Z use semi-implicit Euler at dt (shared discrete
kinematics, the premise of Proposition 1); the truth is integrated with `substeps` smaller steps.
Range widening (the only distribution shift, plan decision 2026-11-06) scales the initial-state deviation and the
torque amplitude by `range_scale`.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent


def planar_xml(n_links: int, link_len: float = 0.4, mass: float = 1.0, damping: float = 0.05) -> str:
    """Vertical-plane n-link arm (hinges about y, gravity on); capsule links give the inertias; no contacts."""
    body, close = '', ''
    for i in range(n_links):
        pos = '0 0 0' if i == 0 else f'{link_len} 0 0'
        body += (f'<body name="l{i}" pos="{pos}"><joint name="j{i}" type="hinge" axis="0 1 0" damping="{damping}"/>'
                 f'<geom type="capsule" fromto="0 0 0 {link_len} 0 0" size="0.03" mass="{mass}" contype="0" conaffinity="0"/>')
        close += '</body>'
    return (f'<mujoco model="planar{n_links}"><option timestep="0.002" integrator="Euler"/>'
            f'<worldbody>{body}{close}</worldbody></mujoco>')


def franka_xml() -> str:
    """panda_nohand.xml (Menagerie, pinned commit) with meshes/actuators/keyframes/contacts/ranges stripped.
    Bodies keep their explicit <inertial> elements, so the dynamics are unchanged by removing geoms."""
    root = ET.parse(HERE / 'assets' / 'panda_nohand.xml').getroot()
    for tag in ('asset', 'actuator', 'keyframe', 'contact', 'sensor', 'tendon', 'equality'):
        for el in root.findall(tag):
            root.remove(el)
    for parent in root.iter():
        for g in list(parent.findall('geom')):
            parent.remove(g)
        for site in list(parent.findall('site')):
            parent.remove(site)
    for j in root.iter('joint'):
        j.attrib.pop('range', None); j.set('limited', 'false')
    comp = root.find('compiler')
    if comp is not None:
        comp.attrib.pop('meshdir', None); comp.set('autolimits', 'false')
    for d in root.iter('default'):
        for child in list(d):
            if child.tag in ('geom', 'site', 'mesh', 'material', 'position', 'general', 'motor'):
                d.remove(child)
    return ET.tostring(root, encoding='unicode')


class System:
    """One arm: nominal and true MuJoCo models plus the shared control law."""

    def __init__(self, name: str, spec: dict):
        self.name = name; self.spec = spec
        xml = planar_xml(spec['n_links']) if spec['kind'] == 'planar' else franka_xml()
        self.m_nom = mujoco.MjModel.from_xml_string(xml)
        self.m_true = mujoco.MjModel.from_xml_string(xml)
        self.d = self.m_nom.nv
        scales = list(spec['mass_scale'])
        for b in range(1, self.m_true.nbody):                 # body 0 is the world; extra bodies reuse the cycle
            s = scales[(b - 1) % len(scales)]
            self.m_true.body_mass[b] *= s; self.m_true.body_inertia[b] *= s
        self.dn = mujoco.MjData(self.m_nom); self.dt_ = mujoco.MjData(self.m_true)
        self.q_home = np.array(spec['q_home'], dtype=np.float64)
        self.tau_scale = np.array(spec['torque_scale'], dtype=np.float64)

    # ---- dynamics -------------------------------------------------------------------------------------------
    def _acc(self, model, data, q, v, tau):
        data.qpos[:] = q; data.qvel[:] = v; data.qfrc_applied[:] = tau
        mujoco.mj_forward(model, data)
        return data.qacc.copy()

    def gravity_comp(self, q):
        """Nominal gravity torque at q (what the controller knows)."""
        self.dn.qpos[:] = q; self.dn.qvel[:] = 0.0; self.dn.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.m_nom, self.dn)
        return self.dn.qfrc_bias.copy()

    def missing_force(self, q, v):
        """Forces the prior does not model: extra viscous damping, smooth Coulomb friction, neighbour coupling."""
        sp = self.spec; dq = q - self.q_home
        lap = 2 * dq - np.concatenate([[0.0], dq[:-1]]) - np.concatenate([dq[1:], [0.0]])
        mask = np.array(sp.get('mismatch_mask', [1] * self.d), dtype=np.float64)
        return self.tau_scale * (mask * (-sp['extra_damping'] * v - sp['coulomb'] * np.tanh(v / 0.05)) - sp['coupling'] * lap)

    def control(self, q, v, u):
        """Shared control law: nominal gravity compensation + stabilising PD around q_home + action u."""
        sp = self.spec
        return self.gravity_comp(q) + self.tau_scale * (sp['kp'] * (self.q_home - q) - sp['kd'] * v) + u

    def acc_prior(self, q, v, u):
        return self._acc(self.m_nom, self.dn, q, v, self.control(q, v, u))

    def acc_true(self, q, v, u):
        return self._acc(self.m_true, self.dt_, q, v, self.control(q, v, u) + self.missing_force(q, v))

    # ---- data ------------------------------------------------------------------------------------------------
    def episodes(self, seed: int, n: int, steps: int, dt: float, substeps: int, range_scale: float = 1.0):
        """True trajectories s[n, steps+1, 2d] (q then v) and actions u[n, steps, d]."""
        rng = np.random.default_rng(seed); d = self.d; sp = self.spec
        s = np.empty((n, steps + 1, 2 * d)); u_all = np.empty((n, steps, d))
        amp = range_scale * sp['u_amp'] * self.tau_scale
        for i in range(n):
            q = self.q_home + range_scale * rng.uniform(-sp['q_dev'], sp['q_dev'], d)
            v = range_scale * rng.uniform(-sp['v_dev'], sp['v_dev'], d)
            freq = rng.uniform(0.3, 1.5, (3, d)); phase = rng.uniform(0, 2 * np.pi, (3, d)); w = rng.normal(0, 1, (3, d)) / np.sqrt(3)
            s[i, 0] = np.concatenate([q, v])
            for t in range(steps):
                tt = t * dt
                u = amp * (w * np.sin(2 * np.pi * freq * tt + phase)).sum(axis=0)
                u_all[i, t] = u
                h = dt / substeps
                for _ in range(substeps):
                    v = v + h * self.acc_true(q, v, u); q = q + h * v
                s[i, t + 1] = np.concatenate([q, v])
        return s, u_all

    def observed_residual(self, s, u, dt):
        """Finite-difference acceleration of the true trajectory minus the prior acceleration (labels for Z)."""
        n, steps, d = u.shape
        x = np.empty((n * steps, 3 * d)); y = np.empty((n * steps, d)); k = 0
        for i in range(n):
            for t in range(steps):
                q, v = s[i, t, :d], s[i, t, d:]
                a_obs = (s[i, t + 1, d:] - v) / dt
                x[k] = np.concatenate([q, v, u[i, t]]); y[k] = a_obs - self.acc_prior(q, v, u[i, t]); k += 1
        return x, y

    def rollout(self, s0, u, dt, residual=None):
        """Semi-implicit Euler roll-out from s0[n, 2d] under actions u[n, steps, d]; residual(x[n, 3d]) -> [n, d]."""
        n, steps, d = u.shape
        out = np.empty((n, steps, 2 * d)); q = s0[:, :d].copy(); v = s0[:, d:].copy()
        for t in range(steps):
            a = np.stack([self.acc_prior(q[i], v[i], u[i, t]) for i in range(n)])
            if residual is not None:
                a = a + residual(np.concatenate([q, v, u[:, t]], axis=1))
            v = v + dt * a; q = q + dt * v
            out[:, t] = np.concatenate([q, v], axis=1)
        return out


# ---- d-dimensional versions of the stacking / correction helpers (confirm.py hard-codes d = 2) -----------------
def stack(x, d):
    """Rows (episode, step, position/velocity) x d columns."""
    return np.concatenate([x[..., :d], x[..., d:]], axis=0).reshape(-1, d)


def apply(p, z, a, d):
    return np.concatenate([p[..., :d] + (z[..., :d] - p[..., :d]) @ a, p[..., d:] + (z[..., d:] - p[..., d:]) @ a], axis=-1)


def kinematic(pred, s0, dt, d):
    qprev = np.concatenate([s0[:, None, :d], pred[:, :-1, :d]], axis=1)
    return float(np.max(np.abs(pred[..., :d] - qprev - dt * pred[..., d:])))
