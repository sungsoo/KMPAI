"""C7 pilot (exploratory, NOT confirmatory): recompute C1 judge-episode losses for a few C1 cases to size C7.
Uses C1 seeds/data only to choose C7 thresholds and power; C7 itself will use new seeds."""
import json, sys
from pathlib import Path
import numpy as np
IMPL = Path(__file__).resolve().parents[3] / 'impl'
sys.path.insert(0, str(IMPL)); sys.path.insert(0, str(IMPL / 'c1'))
import c1_study as S
from c1_models import System, stack, apply
from calibration import select

def judge_and_test(C, sys_name, sys_idx, n, noise, seed):
    Sy = System(sys_name, C['systems'][sys_name]); d = Sy.d; dt, H, sub = C['dt'], C['steps'], C['truth_substeps']; E = C['episodes']
    base = seed + 100000 * sys_idx; gen = lambda k, m, rs=1.0: Sy.episodes(base + 1000 * k, m, H, dt, sub, rs)
    tr_s, tr_u = gen(1, max(C['train_episodes'])); tr_s, tr_u = tr_s[:n], tr_u[:n]
    cf, cs1, cs2 = gen(2, E['calibration']), gen(3, E['selection']), gen(4, E['judge'])
    x, y = Sy.observed_residual(tr_s, tr_u, dt); xv, yv = Sy.observed_residual(*cs1, dt)
    rng = np.random.default_rng(base + 7); rms = np.sqrt((y ** 2).mean(0)); y_noisy = y + rng.normal(0, 1, y.shape) * noise * rms
    fZ, _ = S.fit_mlp(x, y_noisy, xv, yv, d, C['mlp'], seed)
    roll = lambda blk, f=None: Sy.rollout(blk[0][:, 0], blk[1], dt, f)
    pf, zf = roll(cf), roll(cf, fZ); p1, z1 = roll(cs1), roll(cs1, fZ); p2, z2 = roll(cs2), roll(cs2, fZ)
    yf, y1, y2 = cf[0][:, 1:], cs1[0][:, 1:], cs2[0][:, 1:]
    r, e = stack(zf - pf, d), stack(yf - pf, d); rv, ev = stack(z1 - p1, d), stack(y1 - p1, d)
    mats = {'prior': np.zeros((d, d)), 'original': np.eye(d)}
    for kind in ('scalar', 'diagonal', 'matrix'):
        mats[kind], _ = select(r, e, rv, ev, kind, C['lambdas'])
    sel = {k: float(np.mean((rv @ a - ev) ** 2)) for k, a in mats.items()}
    best = min(('prior', 'scalar', 'diagonal', 'matrix'), key=lambda k: sel[k])
    lz = S.episode_loss(z2, y2); lc = S.episode_loss(apply(p2, z2, mats[best], d), y2)
    td = gen(6, E['test']); p, z = roll(td), roll(td, fZ); t = td[0][:, 1:]
    lzt = S.episode_loss(z, t); lct = S.episode_loss(apply(p, z, mats[best], d), t)
    return {'key': f'{sys_name}_n{n}_noise{noise}_seed{seed}', 'best': best, 'judge_ratio': (lc / lz).tolist(), 'test_ratio': (lct / lzt).tolist(),
            'test_rel_gain': float(1 - ((apply(p, z, mats[best], d) - t) ** 2).mean() / ((z - t) ** 2).mean())}

if __name__ == '__main__':
    C = S.cfg(); out = []
    for i, s in enumerate(C['systems']):
        for (n, nz, sd) in [(12, 0.3, 3037), (36, 0.0, 3041), (12, 0.0, 3049), (36, 0.3, 3061)]:
            out.append(judge_and_test(C, s, i, n, nz, sd)); print(out[-1]['key'], out[-1]['best'], flush=True)
    Path('pilot_judge_ratios.json').write_text(json.dumps(out))
