"""Tolerance-margin (eta) LTT study (ICTExpress-v1), pre-registered.

    python eta_study.py --lock        # 1) fix the protocol (refuses if results already exist)
    python eta_study.py --run         # 2) 80 runs, new seeds/test data -> results_h10s_eta/
    python eta_study.py --analyze     # 3) summary.csv, numbers.csv, claims.json (claims E1-E4 of eta_config.json)

Rule: apply a candidate c iff u(k_eta, n, eps) <= delta with k_eta = #{i in C_s2 : L_i(c) > (1+eta) L_i(Z)}.
Targets: the post-hoc correction c* (chosen on C_s1 exactly as in confirm.py) and the retrained world model
(confirm_r2.py definition: refitted on train + calibration, judge set held out).
The locked files run.py, calibration.py, confirm.py, confirm_config.json and horizon_config.json are imported
and read, never modified; their hashes are part of this study's lock (eta_protocol_lock.json).
"""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'eta_config.json'; LOCK = ROOT / 'eta_protocol_lock.json'
LOCKED = ('eta_config.json', 'eta_study.py', 'horizon_config.json', 'confirm.py', 'confirm_config.json', 'run.py',
          'calibration.py')
SPLITS = ('id', 'ood_scale', 'ood_damp')
CAND = ('prior', 'original', 'scalar', 'diagonal', 'matrix')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def configs():
    e = json.loads(CONFIG.read_text()); h = json.loads((ROOT / e['horizon_config']).read_text())
    base = json.loads((ROOT / e['base_config']).read_text())
    steps = int(round(h['horizon_seconds'] / base['dt']))
    cfg = dict(base, steps=steps, seeds=e['seeds'], test_seed_offset=e['test_seed_offset'])
    etas = sorted({e['eta_reference'], e['eta_primary'], *e['eta_sensitivity']})
    return e, cfg, etas, f"h{h['horizon_seconds']:g}s"


def out_dir(tag, suffix=''):
    return ROOT / f'results_{tag}_eta{suffix}'


def check_lock():
    if not LOCK.exists():
        raise SystemExit('eta_protocol_lock.json missing: run  python eta_study.py --lock  first')
    L = json.loads(LOCK.read_text())
    bad = [f for f in LOCKED if L['sha256'].get(f) != sha(ROOT / f)]
    if bad:
        raise SystemExit(f'Protocol changed after locking ({", ".join(bad)}): new seeds + new lock required')
    return L


def do_lock():
    e, cfg, etas, tag = configs()
    if out_dir(tag).exists():
        raise SystemExit(f'{out_dir(tag)} already exists; a lock must precede the data')
    L = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'steps': cfg['steps'], 'seeds': cfg['seeds'],
         'test_seed_offset': cfg['test_seed_offset'], 'etas': etas, 'sha256': {f: sha(ROOT / f) for f in LOCKED}}
    LOCK.write_text(json.dumps(L, indent=2)); print('locked', tag, 'steps', cfg['steps'], 'etas', etas)


def decide(lc, lz, eta, delta, eps, cp_upper):
    k = int((lc > (1.0 + eta) * lz).sum()); u = cp_upper(k, len(lc), eps)
    return bool(u <= delta), k, float(u)


def run_case(cfg, e, etas, system, regime, n, noise, seed, dev, out):
    import run
    from calibration import select
    from confirm import generate, stack, apply, cp_upper, episode_loss, kinematic
    key = f'{system}_{regime}_n{n}_noise{noise}_seed{seed}'; case = out / 'cases' / key
    if (case / 'complete.json').exists():
        return json.loads((case / 'complete.json').read_text())
    case.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter(); dt, steps = cfg['dt'], cfg['steps']
    roll = lambda m, d: run.rollout('prior' if m is None else 'residual', m, None, d, system, dt, dev)[1]
    rng = np.random.default_rng(seed + 90000)
    sizes = {'train': n, 'validation': cfg['validation_episodes'], 'calibration': cfg['calibration_episodes'], 'judge': cfg['judge_episodes']}
    data = {k: generate(seed + 1000 * i, sizes[k], steps, system, dt, 1.0, regime) for i, k in enumerate(sizes, 1)}
    x, y = run.targets(data['train'], system, noise, rng, regime); xv, yv = run.targets(data['validation'], system, noise, rng, regime)
    xc, yc = run.targets(data['calibration'], system, noise, rng, regime); xj, yj = run.targets(data['judge'], system, noise, rng, regime)
    z_model, _ = run.fit(x, y, xv, yv, system, 'residual', 0, cfg, seed, dev)
    tr = time.perf_counter()
    r_model, _ = run.fit(np.concatenate([x, xc]), np.concatenate([y, yc]), xv, yv, system, 'residual', 0, cfg, seed, dev)
    retrain_seconds = time.perf_counter() - tr
    tc = time.perf_counter()
    pc, zc = roll(None, data['calibration']), roll(z_model, data['calibration'])
    pv, zv = roll(None, data['validation']), roll(z_model, data['validation'])
    pj, zj = roll(None, data['judge']), roll(z_model, data['judge'])
    s0 = data['calibration'][0][:, 0]; aobs = yc.reshape(-1, steps, 2)
    vv = s0[:, None, 2:] + dt * np.cumsum(aobs, axis=1); qq = s0[:, None, :2] + dt * np.cumsum(vv, axis=1)
    obs = np.concatenate([qq, vv], axis=-1)
    r, ee = stack(zc - pc), stack(obs - pc); rv, ev = stack(zv - pv), stack(data['validation'][0][:, 1:] - pv)
    mats = {'prior': np.zeros((2, 2)), 'original': np.eye(2)}
    for kind in ('scalar', 'diagonal', 'matrix'):
        mats[kind], _ = select(r, ee, rv, ev, kind, cfg['lambdas'])
    sel = {c: float(np.mean((rv @ a - ev) ** 2)) for c, a in mats.items()}
    best = min(('prior', 'scalar', 'diagonal', 'matrix'), key=lambda c: (sel[c], CAND.index(c)))
    truth_j = data['judge'][0][:, 1:]; lzj = episode_loss(zj, truth_j)
    judge_loss = {'posthoc': episode_loss(apply(pj, zj, mats[best]), truth_j)}
    calib_seconds = time.perf_counter() - tc
    judge_loss['retrain'] = episode_loss(roll(r_model, data['judge']), truth_j)
    dec = {}
    for tgt in e['targets']:
        for eta in etas:
            ok, k, u = decide(judge_loss[tgt], lzj, eta, e['delta'], e['epsilon'], cp_upper)
            dec[(tgt, eta)] = {'applied': ok, 'k': k, 'u': u}
    ratio = {t: (judge_loss[t] / lzj).tolist() for t in e['targets']}
    splits = {'id': (1.0, 0.0), 'ood_scale': (cfg['ood_scale'], 0.0), 'ood_damp': (1.0, cfg['ood_extra_damping'])}
    rows = []
    for i, (split, (scale, damp)) in enumerate(splits.items(), 6):
        td = generate(seed + 1000 * i + cfg['test_seed_offset'], cfg['test_episodes'], steps, system, dt, scale, regime, damp)
        truth = td[0][:, 1:]; p, z0 = roll(None, td), roll(z_model, td)
        z = apply(p, z0, mats['original'])            # same numeric path as confirm.py keep_original
        pred = {'posthoc': apply(p, z0, mats[best]), 'retrain': roll(r_model, td)}
        lz = episode_loss(z, truth); mz = float(((z.astype('float64') - truth) ** 2).mean())
        for tgt in e['targets']:
            lc = episode_loss(pred[tgt], truth)
            for eta in etas:
                ap = dec[(tgt, eta)]['applied']; out_pred = pred[tgt] if ap else z; lo = lc if ap else lz
                rows.append({'key': key, 'system': system, 'regime': regime, 'n': n, 'noise': noise, 'seed': seed, 'split': split,
                             'target': tgt, 'eta': eta, 'applied': int(ap), 'judge_k': dec[(tgt, eta)]['k'], 'judge_u': dec[(tgt, eta)]['u'],
                             'mse': float(((out_pred.astype('float64') - truth) ** 2).mean()), 'mse_original': mz,
                             'ep_n': len(lo), 'ep_worse_strict': int((lo > lz).sum()), 'ep_exceed_eta': int((lo > (1 + eta) * lz).sum()),
                             'cand_ep_worse_strict': int((lc > lz).sum()), 'cand_ep_exceed_eta': int((lc > (1 + eta) * lz).sum()),
                             'kinematic_max': kinematic(out_pred, td[0][:, 0], dt)})
    rep = {'key': key, 'system': system, 'regime': regime, 'n': n, 'noise': noise, 'seed': seed, 'best_candidate': best,
           'selection_loss': sel, 'decisions': {f'{t}|{eta}': v for (t, eta), v in dec.items()}, 'judge_ratio': ratio,
           'matrices': {k: v.tolist() for k, v in mats.items()}, 'rows': rows,
           'seconds': {'retrain': retrain_seconds, 'posthoc_calibration': calib_seconds, 'case_total': time.perf_counter() - t0}}
    run.atomic_json(case / 'complete.json', rep)
    return rep


def do_run(device, suffix):
    import torch, run, sysinfo
    e, cfg, etas, tag = configs(); L = check_lock(); out = out_dir(tag, suffix); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['cpu_threads']); dev = run.choose_device(device)
    if not (out / 'system_env.json').exists():
        (out / 'system_env.json').write_text(json.dumps(sysinfo.collect(str(dev)), indent=2, ensure_ascii=False))
    reps, t0 = [], time.perf_counter()
    for system in cfg['systems']:
        for regime in cfg['regimes']:
            for n in cfg['train_episodes']:
                for noise in cfg['noise']:
                    for seed in cfg['seeds']:
                        rp = run_case(cfg, e, etas, system, regime, n, noise, seed, dev, out); reps.append(rp)
                        d = rp['decisions']
                        print(rp['key'], 'post', [int(d[f'posthoc|{x}']['applied']) for x in etas],
                              'retrain', [int(d[f'retrain|{x}']['applied']) for x in etas], flush=True)
    rows = [r for rp in reps for r in rp['rows']]
    with open(out / 'rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    run.atomic_json(out / 'run_summary.json', {'cases': len(reps), 'seconds_this_invocation': time.perf_counter() - t0,
                                               'protocol_lock': L, 'device': str(dev), 'steps': cfg['steps']})


def cp_hi(k, n, eps=0.05):
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(1 - eps, k + 1, n - k))


def do_analyze(suffix):
    e, cfg, etas, tag = configs(); out = out_dir(tag, suffix); A = out / 'analysis'; A.mkdir(exist_ok=True)
    reps = [json.loads((d / 'complete.json').read_text()) for d in sorted((out / 'cases').iterdir())]
    rows = [r for rp in reps for r in rp['rows']]
    B = e['bootstrap']['B']; rng = np.random.default_rng(e['bootstrap']['rng_seed'])
    summ = []
    for split in SPLITS:
        for tgt in e['targets']:
            for eta in etas:
                sel = [r for r in rows if r['split'] == split and r['target'] == tgt and r['eta'] == eta]
                m = np.array([r['mse'] for r in sel]); z = np.array([r['mse_original'] for r in sel])
                seeds = np.array([r['seed'] for r in sel]); us = np.unique(seeds); groups = {s: np.where(seeds == s)[0] for s in us}
                appl = np.array([r['applied'] for r in sel], bool)
                gain = lambda idx: 1 - m[idx].mean() / z[idx].mean()
                boots = []
                for _ in range(B):
                    idx = np.concatenate([groups[s] for s in rng.choice(us, size=len(us), replace=True)]); boots.append(gain(idx))
                ep_n = sum(r['ep_n'] for r, a in zip(sel, appl) if a)
                ep_s = sum(r['ep_worse_strict'] for r, a in zip(sel, appl) if a); ep_e = sum(r['ep_exceed_eta'] for r, a in zip(sel, appl) if a)
                rel = (z - m) / z
                summ.append({'split': split, 'target': tgt, 'eta': eta, 'runs': len(sel), 'applied': int(appl.sum()),
                             'net_gain': float(gain(np.arange(len(m)))), 'gain_lo': float(np.percentile(boots, 2.5)),
                             'gain_hi': float(np.percentile(boots, 97.5)), 'median_rel_gain_applied': float(np.median(rel[appl])) if appl.any() else float('nan'),
                             'runs_worse': int((m > z * (1 + 1e-12)).sum()), 'applied_runs_worse': int(((m > z * (1 + 1e-12)) & appl).sum()),
                             'ep_applied': ep_n, 'ep_worse_strict': ep_s, 'ep_exceed_eta': ep_e,
                             'ep_worse_strict_rate': ep_s / ep_n if ep_n else float('nan'), 'ep_exceed_eta_rate': ep_e / ep_n if ep_n else float('nan'),
                             'ep_exceed_eta_hi': cp_hi(ep_e, ep_n) if ep_n else float('nan'),
                             'ep_worse_strict_hi': cp_hi(ep_s, ep_n) if ep_n else float('nan')})
    with open(A / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    S = {(r['split'], r['target'], r['eta']): r for r in summ}
    ep, d = e['eta_primary'], e['delta']
    e1 = {}
    for tgt in e['targets']:
        r = S[('id', tgt, ep)]
        e1[tgt] = 'inconclusive' if r['applied'] == 0 else ('supported' if r['ep_exceed_eta_hi'] <= d else 'not_supported')
    e1_all = 'not_supported' if 'not_supported' in e1.values() else ('supported' if all(v == 'supported' for v in e1.values()) else 'inconclusive')
    inc = S[('id', 'retrain', ep)]['applied'] - S[('id', 'retrain', e['eta_reference'])]['applied']
    claims = {'E1_validity': {'verdict': e1_all, 'per_target': e1,
                              'ep_exceed_hi': {t: S[('id', t, ep)]['ep_exceed_eta_hi'] for t in e['targets']}},
              'E2_usefulness': {'verdict': 'supported' if inc >= e['min_effect_runs'] else 'not_supported', 'increase_runs': inc,
                                'min_effect_runs': e['min_effect_runs']},
              'E3_cost': 'descriptive (summary.csv)', 'E4_shift': 'descriptive (summary.csv, applied_runs_worse at ood_damp)'}
    (A / 'claims.json').write_text(json.dumps(claims, indent=2))
    judge = [{'key': rp['key'], 'seed': rp['seed'], 'best_candidate': rp['best_candidate'],
              **{f'{t}_{eta}_k': rp['decisions'][f'{t}|{eta}']['k'] for t in e['targets'] for eta in etas},
              'retrain_ratio_median': float(np.median(rp['judge_ratio']['retrain'])),
              'posthoc_ratio_median': float(np.median(rp['judge_ratio']['posthoc']))} for rp in reps]
    with open(A / 'judge.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(judge[0])); w.writeheader(); w.writerows(judge)
    for r in summ:
        if r['split'] in ('id', 'ood_damp'):
            print(f"{r['split']:9s} {r['target']:8s} eta={r['eta']:.2f} applied={r['applied']:2d} gain={100*r['net_gain']:6.1f}% "
                  f"[{100*r['gain_lo']:.1f},{100*r['gain_hi']:.1f}] runs_worse_applied={r['applied_runs_worse']} "
                  f"ep_strict={r['ep_worse_strict']}/{r['ep_applied']} ep_eta={r['ep_exceed_eta']}/{r['ep_applied']}")
    print(json.dumps(claims, indent=1))


def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--lock', action='store_true'); g.add_argument('--run', action='store_true'); g.add_argument('--analyze', action='store_true')
    ap.add_argument('--device', default='cpu'); ap.add_argument('--suffix', default='', help="'_mac' for a reproduction on this Mac")
    a = ap.parse_args()
    if a.lock:
        return do_lock()
    if a.run:
        return do_run(a.device, a.suffix)
    return do_analyze(a.suffix)


if __name__ == '__main__':
    sys.exit(main())
