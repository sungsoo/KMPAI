"""Pre-registered confirmatory experiment (v14).

Answers reviewer points M3-M7 on NEW seeds and NEW test data, generated only here:
  * new seeds (101..229) and a new test-seed offset (700000), never used before;
  * two distribution shifts: larger initial-state/input range (ood_scale) and a
    dynamics shift with extra damping (ood_damp);
  * Learn-then-Test (LTT) application rule: candidate chosen on C_s1 (validation),
    applied only if the one-sided Clopper-Pearson upper bound of P(episode worse than
    the original forecast) on C_s2 (judge episodes) is <= delta;
  * baselines: always keep Z, fixed trajectory matrix, mean-loss rule (tau), P/Z
    selection, Bates-Granger per-step weights, equal-budget retraining, oracle;
  * cost: post-hoc calibration seconds vs retraining seconds.
run.py / physics_full.py / calibration.py are imported unchanged.

    python confirm.py --out results_confirm            # resumes by case
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import beta
import run
from calibration import select
import sysinfo

ROOT = Path(__file__).resolve().parent
CAND = ('prior', 'original', 'scalar', 'diagonal', 'matrix')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stack(x):
    return np.concatenate([x[..., :2], x[..., 2:]], axis=0)


def apply(p, z, a):
    return np.concatenate([p[..., :2] + (z[..., :2] - p[..., :2]) @ a, p[..., 2:] + (z[..., 2:] - p[..., 2:]) @ a], axis=-1)


def kinematic(pred, s0, dt):
    qprev = np.concatenate([s0[:, None, :2], pred[:, :-1, :2]], axis=1)
    return float(np.max(np.abs(pred[..., :2] - qprev - dt * pred[..., 2:])))


def generate(seed, n, steps, system, dt, scale, regime, extra_damping=0.0):
    """Same generator as run.generate; optional extra damping changes the true dynamics."""
    rng = np.random.default_rng(seed)
    s = np.empty((n, steps + 1, 4), dtype=np.float32); u_all = np.empty((n, steps, 2), dtype=np.float32)
    s[:, 0] = rng.uniform(-scale, scale, (n, 4)); u = rng.normal(0, 0.65 * scale, (n, 2))
    for t in range(steps):
        if t % 8 == 0: u = rng.normal(0, 0.65 * scale, (n, 2))
        u_all[:, t] = u
        a = run.truth_np(s[:, t], u, system, regime) - extra_damping * s[:, t, 2:]
        v = s[:, t, 2:] + dt * a
        s[:, t + 1] = np.concatenate((s[:, t, :2] + dt * v, v), axis=-1)
    return s, u_all


def cp_upper(k, n, eps):
    return 1.0 if k >= n else float(beta.ppf(1 - eps, k + 1, n - k))


def episode_loss(pred, truth):
    return ((pred.astype('float64') - truth) ** 2).mean(axis=(1, 2))


def run_case(cfg, system, regime, n, noise, seed, dev, out):
    key = f'{system}_{regime}_n{n}_noise{noise}_seed{seed}'
    case = out / 'cases' / key
    if (case / 'complete.json').exists():
        return json.loads((case / 'complete.json').read_text())
    case.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter(); dt, steps = cfg['dt'], cfg['steps']
    rng = np.random.default_rng(seed + 90000)
    sizes = {'train': n, 'validation': cfg['validation_episodes'], 'calibration': cfg['calibration_episodes'], 'judge': cfg['judge_episodes']}
    data = {k: generate(seed + 1000 * i, sizes[k], steps, system, dt, 1.0, regime) for i, k in enumerate(sizes, 1)}
    x, y = run.targets(data['train'], system, noise, rng, regime)
    xv, yv = run.targets(data['validation'], system, noise, rng, regime)
    xc, yc = run.targets(data['calibration'], system, noise, rng, regime)
    xj, yj = run.targets(data['judge'], system, noise, rng, regime)
    tf = time.perf_counter(); z_model, z_info = run.fit(x, y, xv, yv, system, 'residual', 0, cfg, seed, dev); z_seconds = time.perf_counter() - tf
    tf = time.perf_counter()
    eb_model, eb_info = run.fit(np.concatenate([x, xc, xj]), np.concatenate([y, yc, yj]), xv, yv, system, 'residual', 0, cfg, seed, dev)
    retrain_seconds = time.perf_counter() - tf
    roll = lambda m, d: run.rollout('prior' if m is None else 'residual', m, None, d, system, dt, dev)[1]
    # ---- post-hoc calibration (timed): rollouts on held-out sets + fit + selection + LTT
    tc = time.perf_counter()
    pc, zc = roll(None, data['calibration']), roll(z_model, data['calibration'])
    pv, zv = roll(None, data['validation']), roll(z_model, data['validation'])
    pj, zj = roll(None, data['judge']), roll(z_model, data['judge'])
    s0 = data['calibration'][0][:, 0]; aobs = yc.reshape(-1, steps, 2)
    vv = s0[:, None, 2:] + dt * np.cumsum(aobs, axis=1); qq = s0[:, None, :2] + dt * np.cumsum(vv, axis=1)
    obs = np.concatenate([qq, vv], axis=-1)
    r, e = stack(zc - pc), stack(obs - pc)
    rv, ev = stack(zv - pv), stack(data['validation'][0][:, 1:] - pv)
    mats = {'prior': np.zeros((2, 2)), 'original': np.eye(2)}; cal = {}
    for kind in ('scalar', 'diagonal', 'matrix'):
        mats[kind], info = select(r, e, rv, ev, kind, cfg['lambdas'])
        cal[kind] = {k: info[k] for k in ('lambda_', 'selection_mse', 'alpha') if k in info}
    sel_loss = {c: float(np.mean((rv @ a - ev) ** 2)) for c, a in mats.items()}
    best = min(('prior', 'scalar', 'diagonal', 'matrix'), key=lambda c: (sel_loss[c], CAND.index(c)))
    truth_j = data['judge'][0][:, 1:]
    d = episode_loss(apply(pj, zj, mats[best]), truth_j) - episode_loss(zj, truth_j)
    k_worse, n_judge = int((d > 0).sum()), len(d)
    ucb = cp_upper(k_worse, n_judge, cfg['ltt_epsilon'])
    ltt_pick = best if ucb <= cfg['ltt_delta'] else 'original'
    mean_pick = best if sel_loss[best] <= (1 - cfg['mean_rule_tau']) * sel_loss['original'] else 'original'
    pz_pick = 'prior' if sel_loss['prior'] < sel_loss['original'] else 'original'
    # Bates-Granger style per-step scalar weights fitted on the calibration set
    rr = (zc - pc).astype('float64'); ee = (obs - pc).astype('float64')
    num, den = (rr * ee).sum(axis=(0, 2)), (rr * rr).sum(axis=(0, 2))
    w = np.clip(np.divide(num, den, out=np.zeros_like(num), where=den > 0), 0, 1)
    calib_seconds = time.perf_counter() - tc
    # rotation equivariance check (Proposition 2)
    from calibration import fit as cfit
    th = np.pi / 4; rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ar, _ = cfit(r @ rot, e @ rot, cal['matrix']['lambda_'], 'matrix')
    rot_err = float(np.max(np.abs(ar - rot.T @ mats['matrix'] @ rot)))
    # ---- new test data, generated only now
    splits = {'id': (1.0, 0.0), 'ood_scale': (cfg['ood_scale'], 0.0), 'ood_damp': (1.0, cfg['ood_extra_damping'])}
    rows, curves = [], {}
    for i, (split, (scale, damp)) in enumerate(splits.items(), 6):
        td = generate(seed + 1000 * i + cfg['test_seed_offset'], cfg['test_episodes'], steps, system, dt, scale, regime, damp)
        truth = td[0][:, 1:]; p, z = roll(None, td), roll(z_model, td); eb = roll(eb_model, td)
        cand = {c: apply(p, z, a) for c, a in mats.items()}
        cand_mse = {c: float(((v.astype('float64') - truth) ** 2).mean()) for c, v in cand.items()}
        oracle = min(CAND, key=lambda c: (cand_mse[c], CAND.index(c)))
        methods = {'keep_original': ('original', cand['original']), 'fixed_matrix': ('matrix', cand['matrix']),
                   'mean_rule': (mean_pick, cand[mean_pick]), 'ltt': (ltt_pick, cand[ltt_pick]),
                   'pz_select': (pz_pick, cand[pz_pick]), 'bates_granger': ('per_step', p + (z - p) * w[None, :, None]),
                   'equal_budget_retrain': ('retrain', eb), 'oracle': (oracle, cand[oracle]), 'prior': ('prior', cand['prior'])}
        for mname, (pick, pred) in methods.items():
            err = (pred.astype('float64') - truth) ** 2
            curves[f'{split}__{mname}'] = err.mean(axis=(0, 2))
            rows.append({'key': key, 'system': system, 'regime': regime, 'n': n, 'noise': noise, 'seed': seed, 'split': split,
                         'method': mname, 'pick': pick, 'applied': int(pick not in ('original',)), 'mse': float(err.mean()),
                         'kinematic_max': kinematic(pred, td[0][:, 0], dt)})
    np.savez_compressed(case / 'curves.npz', **curves)
    rep = {'key': key, 'system': system, 'regime': regime, 'n': n, 'noise': noise, 'seed': seed, 'rows': rows,
           'selection_loss': sel_loss, 'best_candidate': best, 'judge_worse': k_worse, 'judge_n': n_judge, 'cp_upper': ucb,
           'ltt_pick': ltt_pick, 'mean_rule_pick': mean_pick, 'pz_pick': pz_pick, 'per_step_weights': w.tolist(),
           'matrices': {k: v.tolist() for k, v in mats.items()}, 'calibration': cal, 'rotation_equivariance_max_error': rot_err,
           'seconds': {'z_training': z_seconds, 'retrain': retrain_seconds, 'posthoc_calibration': calib_seconds, 'case_total': time.perf_counter() - t0},
           'training': {'z': z_info, 'retrain': eb_info}}
    run.atomic_json(case / 'complete.json', rep)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results_confirm'); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    cfg = json.loads((ROOT / 'confirm_config.json').read_text()); out = (ROOT / a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = ROOT / 'confirm_protocol_lock.json'
    if not lock.exists():
        raise SystemExit('confirm_protocol_lock.json missing: lock the protocol before running')
    L = json.loads(lock.read_text())
    if L['confirm_config_sha256'] != sha(ROOT / 'confirm_config.json') or L['confirm_py_sha256'] != sha(__file__):
        raise SystemExit('Protocol changed after locking; use a new protocol/version')
    torch.set_num_threads(cfg['cpu_threads']); dev = run.choose_device(a.device)
    if not (out / 'system_env.json').exists():
        (out / 'system_env.json').write_text(json.dumps(sysinfo.collect(str(dev)), indent=2, ensure_ascii=False))
    reps = []; start = time.perf_counter()
    for system in cfg['systems']:
        for regime in cfg['regimes']:
            for n in cfg['train_episodes']:
                for noise in cfg['noise']:
                    for seed in cfg['seeds']:
                        r = run_case(cfg, system, regime, n, noise, seed, dev, out); reps.append(r)
                        print(r['key'], 'ltt', r['ltt_pick'], 'k', r['judge_worse'], flush=True)
    rows = [row for r in reps for row in r['rows']]
    import csv
    with open(out / 'cases.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    run.atomic_json(out / 'run_summary.json', {'cases': len(reps), 'seconds_this_invocation': time.perf_counter() - start,
                                               'protocol_lock': L, 'device': str(dev)})
    print('TOTAL', time.perf_counter() - start)


if __name__ == '__main__':
    main()
