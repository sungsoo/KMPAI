"""Round-2 reviewer analyses of the pre-registered confirmatory run (v15).

The confirmatory run (confirm.py, protocol locked) is NOT changed. This script
re-derives every case deterministically (same seeds -> identical models, data and
picks; checked against results_confirm/cases/*/complete.json) and adds quantities the
second review asked for. Items marked SUPPLEMENTARY use test data that were already
analysed and are reported as post-hoc analyses, not confirmatory results.

  (M2)  episode-level ID worse rate of every LTT-applied run (direct check of Prop. 3)
  (M5)  unconstrained least-squares matrix vs. symmetric contraction (fixed, no rule);
        rotation error of the diagonal candidate
  (M4)  SUPPLEMENTARY retrain+LTT: retrain on train+calibration (judge set held out),
        apply the retrained model only if LTT on the judge set passes
  (M1)  SUPPLEMENTARY judge-set size n in {36, 60, 100} (fresh judge episodes)

    python confirm_r2.py --results results_confirm --out results_confirm_r2
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
import run
from calibration import select, fit as cfit
from confirm import generate, stack, apply, kinematic, cp_upper, episode_loss

ROOT = Path(__file__).resolve().parent


def roll(m, d, system, dt, dev):
    return run.rollout('prior' if m is None else 'residual', m, None, d, system, dt, dev)[1]


def ltt_pick(loss_c, loss_i, delta, eps):
    k = int((loss_c > loss_i).sum()); n = len(loss_c); u = cp_upper(k, n, eps)
    return u <= delta, k, u


def case(cfg, rep, dev):
    system, regime, n, noise, seed = rep['system'], rep['regime'], rep['n'], rep['noise'], rep['seed']
    dt, steps = cfg['dt'], cfg['steps']; rng = np.random.default_rng(seed + 90000)
    sizes = {'train': n, 'validation': cfg['validation_episodes'], 'calibration': cfg['calibration_episodes'], 'judge': cfg['judge_episodes']}
    data = {k: generate(seed + 1000 * i, sizes[k], steps, system, dt, 1.0, regime) for i, k in enumerate(sizes, 1)}
    x, y = run.targets(data['train'], system, noise, rng, regime); xv, yv = run.targets(data['validation'], system, noise, rng, regime)
    xc, yc = run.targets(data['calibration'], system, noise, rng, regime); xj, yj = run.targets(data['judge'], system, noise, rng, regime)
    z_model, _ = run.fit(x, y, xv, yv, system, 'residual', 0, cfg, seed, dev)
    eb_model, _ = run.fit(np.concatenate([x, xc, xj]), np.concatenate([y, yc, yj]), xv, yv, system, 'residual', 0, cfg, seed, dev)
    t0 = time.perf_counter()
    r2_model, _ = run.fit(np.concatenate([x, xc]), np.concatenate([y, yc]), xv, yv, system, 'residual', 0, cfg, seed, dev)  # judge held out
    r2_seconds = time.perf_counter() - t0
    pc, zc = roll(None, data['calibration'], system, dt, dev), roll(z_model, data['calibration'], system, dt, dev)
    pv, zv = roll(None, data['validation'], system, dt, dev), roll(z_model, data['validation'], system, dt, dev)
    pj, zj = roll(None, data['judge'], system, dt, dev), roll(z_model, data['judge'], system, dt, dev)
    s0 = data['calibration'][0][:, 0]; aobs = yc.reshape(-1, steps, 2)
    vv = s0[:, None, 2:] + dt * np.cumsum(aobs, axis=1); qq = s0[:, None, :2] + dt * np.cumsum(vv, axis=1)
    obs = np.concatenate([qq, vv], axis=-1)
    r, e = stack(zc - pc), stack(obs - pc); rv, ev = stack(zv - pv), stack(data['validation'][0][:, 1:] - pv)
    mats = {'prior': np.zeros((2, 2)), 'original': np.eye(2)}; lam = {}
    for kind in ('scalar', 'diagonal', 'matrix'):
        mats[kind], info = select(r, e, rv, ev, kind, cfg['lambdas']); lam[kind] = info['lambda_']
    mats['unconstrained'] = np.linalg.lstsq(r.reshape(-1, 2).astype('float64'), e.reshape(-1, 2).astype('float64'), rcond=None)[0]
    sel = {c: float(np.mean((rv @ mats[c] - ev) ** 2)) for c in ('prior', 'original', 'scalar', 'diagonal', 'matrix')}
    best = min(('prior', 'scalar', 'diagonal', 'matrix'), key=lambda c: (sel[c], ('prior', 'original', 'scalar', 'diagonal', 'matrix').index(c)))
    truth_j = data['judge'][0][:, 1:]
    ok, k, u = ltt_pick(episode_loss(apply(pj, zj, mats[best]), truth_j), episode_loss(zj, truth_j), cfg['ltt_delta'], cfg['ltt_epsilon'])
    pick = best if ok else 'original'
    assert pick == rep['ltt_pick'] and k == rep['judge_worse'], (rep['key'], pick, rep['ltt_pick'], k, rep['judge_worse'])
    # retrain+LTT decision on the held-out judge set
    r2j = roll(r2_model, data['judge'], system, dt, dev)
    r2_ok, r2_k, r2_u = ltt_pick(episode_loss(r2j, truth_j), episode_loss(zj, truth_j), cfg['ltt_delta'], cfg['ltt_epsilon'])
    # diagonal rotation error (Prop. 2 does not hold for axis-wise weights)
    th = np.pi / 4; rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dr, _ = cfit(r @ rot, e @ rot, lam['diagonal'], 'diagonal'); diag_rot_err = float(np.max(np.abs(dr - rot.T @ mats['diagonal'] @ rot)))
    # fresh judge sets of size 60 and 100 (supplementary)
    size_picks = {36: pick}
    for j, nj in ((11, 60), (12, 100)):
        jd = generate(seed + 1000 * j, nj, steps, system, dt, 1.0, regime)
        pjj, zjj = roll(None, jd, system, dt, dev), roll(z_model, jd, system, dt, dev); tj = jd[0][:, 1:]
        okj, _, _ = ltt_pick(episode_loss(apply(pjj, zjj, mats[best]), tj), episode_loss(zjj, tj), cfg['ltt_delta'], cfg['ltt_epsilon'])
        size_picks[nj] = best if okj else 'original'
    rows = []
    splits = {'id': (1.0, 0.0), 'ood_scale': (cfg['ood_scale'], 0.0), 'ood_damp': (1.0, cfg['ood_extra_damping'])}
    stored = {(rw['split'], rw['method']): rw['mse'] for rw in rep['rows']}
    for i, (split, (scale, damp)) in enumerate(splits.items(), 6):
        td = generate(seed + 1000 * i + cfg['test_seed_offset'], cfg['test_episodes'], steps, system, dt, scale, regime, damp)
        truth = td[0][:, 1:]; p, z = roll(None, td, system, dt, dev), roll(z_model, td, system, dt, dev)
        r2 = roll(r2_model, td, system, dt, dev)
        cand = {c: apply(p, z, a) for c, a in mats.items()}
        z = cand['original']  # same numeric path as confirm.py's keep_original (apply with I)
        lz = episode_loss(z, truth)
        chk = float(((cand[rep['ltt_pick']].astype('float64') - truth) ** 2).mean())
        assert abs(chk - stored[(split, 'ltt')]) <= 1e-9 * max(1, chk), (rep['key'], split, chk, stored[(split, 'ltt')])
        out = {'ltt': (cand[pick], pick != 'original'), 'unconstrained_fixed': (cand['unconstrained'], True),
               'retrain_ltt': (r2 if r2_ok else z, bool(r2_ok)), 'retrain_no_judge': (r2, True)}
        for nj, pk in size_picks.items():
            out[f'ltt_n{nj}'] = (cand[pk], pk != 'original')
        for mname, (pred, was_applied) in out.items():
            le = episode_loss(pred, truth)
            rows.append({'key': rep['key'], 'seed': seed, 'n': n, 'split': split, 'method': mname,
                         'applied': int(was_applied), 'mse': float(((pred.astype('float64') - truth) ** 2).mean()),
                         'mse_original': float(((z.astype('float64') - truth) ** 2).mean()),
                         'episode_worse_rate': float((le > lz).mean()), 'kinematic_max': kinematic(pred, td[0][:, 0], dt)})
    return rows, {'key': rep['key'], 'seed': seed, 'n': n, 'ltt_pick': pick, 'judge_k': k, 'cp_upper': u,
                  'retrain_ltt_applied': int(r2_ok), 'retrain_judge_k': r2_k, 'retrain_cp_upper': r2_u, 'retrain_seconds': r2_seconds,
                  'diag_rotation_error': diag_rot_err, 'matrix_rotation_error': rep['rotation_equivariance_max_error'],
                  'pick_n60': size_picks[60], 'pick_n100': size_picks[100], 'unconstrained_symmetric': bool(np.allclose(mats['unconstrained'], mats['unconstrained'].T)),
                  'unconstrained_eigs': np.linalg.eigvals(mats['unconstrained']).real.tolist()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results_confirm'); ap.add_argument('--out', default='results_confirm_r2')
    a = ap.parse_args(); cfg = json.loads((ROOT / 'confirm_config.json').read_text())
    res = (ROOT / a.results).resolve(); out = (ROOT / a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['cpu_threads']); dev = run.choose_device('cpu')
    (out / 'protocol.json').write_text(json.dumps({'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': 'Post-hoc analyses requested by round-2 review. Confirmatory protocol and results unchanged; test data already analysed.',
        'confirm_protocol_lock': json.loads((ROOT / 'confirm_protocol_lock.json').read_text())}, indent=2))
    import sysinfo
    (out / 'system_env.json').write_text(json.dumps(sysinfo.collect('cpu'), indent=2, ensure_ascii=False))
    rows, info = [], []
    for d in sorted((res / 'cases').iterdir()):
        rep = json.loads((d / 'complete.json').read_text()); rw, inf = case(cfg, rep, dev); rows += rw; info.append(inf)
        print(rep['key'], 'ok', flush=True)
    for name, data in (('rows.csv', rows), ('cases.csv', info)):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print('DONE', len(info))


if __name__ == '__main__':
    main()
