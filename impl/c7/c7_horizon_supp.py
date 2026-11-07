"""Supplementary figure H5 (post hoc, descriptive): forecast error over the 1-s horizon on the 7-joint arm.

Selection rule (fixed before any trajectory was recomputed, 2026-11-07): among Franka in-distribution runs of
Study 2 where the magnitude rule with directions applied a correction, take the run with the median per-run gain
(sorted ascending, ties by key, index (n - 1) // 2). The run is recomputed deterministically with the Study 2
code; its test MSE must equal the stored value (relative difference < 1e-9) or no figure is drawn.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; IMPL = HERE.parent
for p in (IMPL, IMPL / 'c1', HERE):
    sys.path.insert(0, str(p))


def main(suffix='_mac', gen=None):
    import c1_study as S1, c7_study as S7
    from c1_models import System, stack, apply
    from calibration import select
    C7, C1 = S7.cfg(); out = IMPL / f'results_c7{suffix}'
    reps = [json.loads(f.read_text()) for f in sorted((out / 'cases').glob('*/complete.json'))]
    rows = [r for x in reps for r in x['rows'] if x['system'] == 'franka_7dof' and r['split'] == 'id' and r['method'] == 'mag_dir_gate' and r['applied']]
    rows = sorted(rows, key=lambda r: (r['rel_gain'], r['key'])); pick = rows[(len(rows) - 1) // 2]
    rep = next(x for x in reps if x['key'] == pick['key'])
    S = System('franka_7dof', C1['systems']['franka_7dof']); d = S.d; dt, H, sub = C1['dt'], C1['steps'], C1['truth_substeps']
    sys_idx = list(C1['systems']).index('franka_7dof'); base = rep['seed'] + 100000 * sys_idx
    gen_ = lambda k, m, rs=1.0: S.episodes(base + 1000 * k, m, H, dt, sub, rs)
    tr_s, tr_u = gen_(1, max(C1['train_episodes'])); tr_s, tr_u = tr_s[:rep['n']], tr_u[:rep['n']]
    cs1 = gen_(3, C1['episodes']['selection'])
    x, y = S.observed_residual(tr_s, tr_u, dt); xv, yv = S.observed_residual(*cs1, dt)
    rng = np.random.default_rng(base + 7); rms = np.sqrt((y ** 2).mean(0)); y_noisy = y + rng.normal(0, 1, y.shape) * rep['noise'] * rms
    fZ, _ = S1.fit_mlp(x, y_noisy, xv, yv, d, C1['mlp'], rep['seed'])
    td = gen_(6, C1['episodes']['test']); truth = td[0][:, 1:]
    p = S.rollout(td[0][:, 0], td[1], dt); z = S.rollout(td[0][:, 0], td[1], dt, fZ); A = np.array(rep['gates']['mag_dir_gate']); c = apply(p, z, A, d)
    m = float(((c - truth) ** 2).mean()); rel = abs(m - pick['mse']) / pick['mse']
    if rel > 1e-9:
        raise SystemExit(f'recomputed MSE differs from the stored run ({rel:.2e}); no figure')
    t = np.arange(1, H + 1) * dt; err = lambda f: np.sqrt(((f[..., :d] - truth[..., :d]) ** 2).mean(axis=(0, 2)))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import figure_ictx as FI
    fig = FI._setup(1.9); ax = fig.subplots(1, 1)
    ax.plot(t, err(p), color='#999999', lw=1.0, label='physics prior P')
    ax.plot(t, err(z), color='#D55E00', lw=1.0, label='learned model Z')
    ax.plot(t, err(c), color='#0072B2', lw=1.2, label='accepted correction')
    ax.set_xlabel('forecast time (s)'); ax.set_ylabel('joint-position RMSE (rad)'); ax.legend(frameon=False, fontsize=FI.FS - 1)
    ax.set_title(f"7-joint arm, median run of the rule ({pick['key'].replace('_', ' ')})", fontsize=FI.FS - 1)
    gdir = Path(gen) if gen else IMPL.parent / 'paper' / 'generated'
    FI._save(fig, gdir / 'supp_horizon'); plt.close(fig)
    print(f"supp_horizon: run {pick['key']}, gain {100 * pick['rel_gain']:.1f}%, recomputed MSE rel diff {rel:.1e}")


if __name__ == '__main__':
    main()
