"""C1 study: constraint-preserving correction and LTT acceptance on multi-joint arms (design/C1/plan.toml).

    python c1_study.py --lock                                  # fix code + config hashes before main data
    python c1_study.py --run [--workers 8] [--suffix _mac]     # all cases (resumable); --smoke for a reduced subset
    python c1_study.py --analyze [--suffix _mac]               # claims M1-M5 -> results_c1{suffix}/analysis

Algorithm 1 is the manuscript's procedure, unchanged: candidates {0, alpha I, D_lambda, A_lambda} fitted with
calibration.select (imported), lambda chosen on C_s1, the C_s1-best candidate tested on the first 36 judge episodes
with the one-sided Clopper-Pearson bound (confirm.cp_upper, imported). Judge sizes 60 / 100 are post hoc (M5).
A failed case is recorded in failed.json and counted as missing, never as 0.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; IMPL = HERE.parent
sys.path.insert(0, str(IMPL)); sys.path.insert(0, str(HERE))
CONFIG, LOCK = HERE / 'c1_config.json', HERE / 'c1_protocol_lock.json'
LOCKED = [HERE / 'c1_config.json', HERE / 'c1_models.py', HERE / 'c1_study.py', HERE / 'assets' / 'panda_nohand.xml',
          IMPL / 'calibration.py', IMPL / 'confirm.py', IMPL.parent / 'design' / 'C1' / 'plan.toml']
SPLITS = ('id', 'range')
METHODS = ('keep_Z', 'physics_prior', 'ltt_correction', 'ltt_diagonal_only', 'mean_loss_rule', 'per_step_combination',
           'equal_budget_retrain', 'no_contraction', 'fixed_matrix_no_gate')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cfg():
    return json.loads(CONFIG.read_text())


def out_dir(suffix=''):
    return IMPL / f'results_c1{suffix}'


def do_lock():
    if out_dir().exists():
        raise SystemExit('results_c1 already exists; the lock must precede the main data')
    L = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sha256': {str(p.relative_to(IMPL.parent)): sha(p) for p in LOCKED}}
    LOCK.write_text(json.dumps(L, indent=2)); print('locked', len(L['sha256']), 'files')


def check_lock():
    if not LOCK.exists():
        raise SystemExit('c1_protocol_lock.json missing: run  python c1_study.py --lock  first')
    L = json.loads(LOCK.read_text())
    bad = [k for k, h in L['sha256'].items() if sha(IMPL.parent / k) != h]
    if bad:
        raise SystemExit(f'Protocol changed after locking: {bad}')
    return L


# ---- learned world model (residual MLP) ------------------------------------------------------------------------
def fit_mlp(x, y, xv, yv, d, c, seed):
    import torch
    from torch import nn
    torch.manual_seed(seed); torch.set_num_threads(1)
    mu, sd = x.mean(0), x.std(0) + 1e-8; ymu, ysd = y.mean(0), y.std(0) + 1e-8
    h = c['hidden_per_joint'] * d + c['hidden_base']
    net = nn.Sequential(nn.Linear(3 * d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, d)).double()
    opt = torch.optim.Adam(net.parameters(), lr=c['lr'])
    X = torch.tensor((x - mu) / sd); Y = torch.tensor((y - ymu) / ysd)
    Xv = torch.tensor((xv - mu) / sd); Yv = torch.tensor((yv - ymu) / ysd)
    g = torch.Generator().manual_seed(seed); best, best_state, wait = float('inf'), None, 0
    for ep in range(c['max_epochs']):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), c['batch']):
            idx = perm[i:i + c['batch']]; opt.zero_grad()
            loss = ((net(X[idx]) - Y[idx]) ** 2).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            vl = float(((net(Xv) - Yv) ** 2).mean())
        if vl < best - 1e-9:
            best, wait = vl, 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= c['patience']:
                break
    net.load_state_dict(best_state)

    def residual(xq):
        with torch.no_grad():
            return net(torch.tensor((xq - mu) / sd)).numpy() * ysd + ymu
    return residual, {'epochs': ep + 1, 'val_mse': best}


def episode_loss(pred, truth):
    return ((pred - truth) ** 2).mean(axis=(1, 2))


# ---- one case -------------------------------------------------------------------------------------------------------
def run_case(C, sys_name, sys_idx, n, noise, seed, out):
    from c1_models import System, stack, apply, kinematic
    from calibration import select
    from confirm import cp_upper
    key = f'{sys_name}_n{n}_noise{noise}_seed{seed}'; case = out / 'cases' / key
    if (case / 'complete.json').exists():
        return json.loads((case / 'complete.json').read_text())
    case.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()
    S = System(sys_name, C['systems'][sys_name]); d = S.d; dt, H, sub = C['dt'], C['steps'], C['truth_substeps']; E = C['episodes']
    base = seed + 100000 * sys_idx
    gen = lambda k, m, rs=1.0: S.episodes(base + 1000 * k, m, H, dt, sub, rs)
    tr_s, tr_u = gen(1, max(C['train_episodes'])); tr_s, tr_u = tr_s[:n], tr_u[:n]
    cf, cs1, cs2 = gen(2, E['calibration']), gen(3, E['selection']), gen(4, E['judge'])
    # labels with noise
    x, y = S.observed_residual(tr_s, tr_u, dt); xv, yv = S.observed_residual(*cs1, dt)
    rng = np.random.default_rng(base + 7); rms = np.sqrt((y ** 2).mean(0))
    y_noisy = y + rng.normal(0, 1, y.shape) * noise * rms
    fZ, infoZ = fit_mlp(x, y_noisy, xv, yv, d, C['mlp'], seed)
    xr = [x]; yr = [y_noisy]
    for blk in (cf, cs2):
        xb, yb = S.observed_residual(*blk, dt); xr.append(xb); yr.append(yb + rng.normal(0, 1, yb.shape) * noise * rms)
    fR, infoR = fit_mlp(np.concatenate(xr), np.concatenate(yr), xv, yv, d, C['mlp'], seed + 1)
    roll = lambda blk, f=None: S.rollout(blk[0][:, 0], blk[1], dt, f)
    pf, zf = roll(cf), roll(cf, fZ); p1, z1 = roll(cs1), roll(cs1, fZ); p2, z2 = roll(cs2), roll(cs2, fZ)
    yf, y1, y2 = cf[0][:, 1:], cs1[0][:, 1:], cs2[0][:, 1:]
    r, e = stack(zf - pf, d), stack(yf - pf, d); rv, ev = stack(z1 - p1, d), stack(y1 - p1, d)
    mats = {'prior': np.zeros((d, d)), 'original': np.eye(d)}; lam = {}
    for kind in ('scalar', 'diagonal', 'matrix'):
        mats[kind], info = select(r, e, rv, ev, kind, C['lambdas']); lam[kind] = info.get('lambda_')
    mats['unconstrained'] = np.linalg.lstsq(r, e, rcond=None)[0]
    sel = {k: float(np.mean((rv @ a - ev) ** 2)) for k, a in mats.items()}
    order = ('prior', 'scalar', 'diagonal', 'matrix')
    best_full = min(order, key=lambda k: (sel[k], order.index(k)))
    best_diag = min(order[:3], key=lambda k: (sel[k], order.index(k)))
    lz2 = episode_loss(z2, y2)
    def gate(cand, nj):
        lc = episode_loss(apply(p2[:nj], z2[:nj], mats[cand], d), y2[:nj]); k = int((lc > lz2[:nj]).sum())
        u = cp_upper(k, nj, C['ltt_epsilon']); return (cand if u <= C['ltt_delta'] else 'original'), k, u
    nprim = E['judge_primary']
    ltt_pick, k_full, u_full = gate(best_full, nprim); diag_pick, k_diag, u_diag = gate(best_diag, nprim)
    size_picks = {nj: gate(best_full, nj)[0] for nj in [nprim] + E['judge_post_hoc']}
    mean_pick = best_full if sel[best_full] <= (1 - C['mean_rule_tau']) * sel['original'] else 'original'
    rr, ee = (zf - pf), (yf - pf); num, den = (rr * ee).sum(axis=(0, 2)), (rr * rr).sum(axis=(0, 2))
    w = np.clip(np.divide(num, den, out=np.zeros_like(num), where=den > 0), 0, 1)
    rows = []
    for si, split in enumerate(SPLITS):
        rs = 1.0 if split == 'id' else C['range_scale']
        td = gen(6 + si, E['test'], rs); truth = td[0][:, 1:]; s0 = td[0][:, 0]
        p, z = roll(td), roll(td, fZ); rt = roll(td, fR)
        cand = lambda k: apply(p, z, mats[k], d)
        preds = {'keep_Z': ('original', z), 'physics_prior': ('prior', p), 'ltt_correction': (ltt_pick, cand(ltt_pick)),
                 'ltt_diagonal_only': (diag_pick, cand(diag_pick)), 'mean_loss_rule': (mean_pick, cand(mean_pick)),
                 'per_step_combination': ('per_step', p + (z - p) * w[None, :, None]), 'equal_budget_retrain': ('retrain', rt),
                 'no_contraction': ('unconstrained', cand('unconstrained')), 'fixed_matrix_no_gate': ('matrix', cand('matrix'))}
        lz = episode_loss(z, truth); mz = float(((z - truth) ** 2).mean())
        for mth, (pick, pred) in preds.items():
            lp = episode_loss(pred, truth); m = float(((pred - truth) ** 2).mean())
            rows.append({'key': key, 'system': sys_name, 'd': d, 'n': n, 'noise': noise, 'seed': seed, 'split': split, 'method': mth,
                         'pick': pick, 'applied': int(pick not in ('original',)), 'mse': m, 'mse_Z': mz, 'rel_gain': (mz - m) / mz,
                         'worse': int(m > mz * (1 + 1e-12)), 'ep_worse': int((lp > lz).sum()), 'ep_n': len(lp),
                         'kinematic_max': kinematic(pred, s0, dt, d)})
    rep = {'key': key, 'system': sys_name, 'd': d, 'n': n, 'noise': noise, 'seed': seed, 'best_full': best_full, 'best_diag': best_diag,
           'ltt_pick': ltt_pick, 'diag_pick': diag_pick, 'mean_pick': mean_pick, 'judge': {'k_full': k_full, 'u_full': u_full, 'k_diag': k_diag, 'u_diag': u_diag},
           'size_picks': {str(k): v for k, v in size_picks.items()}, 'lambda': lam, 'selection_loss': sel,
           'matrices': {k: v.tolist() for k, v in mats.items()}, 'per_step_weights': w.tolist(), 'mlp_Z': infoZ, 'mlp_retrain': infoR,
           'rows': rows, 'seconds': time.perf_counter() - t0}
    tmp = case / 'complete.json.tmp'; tmp.write_text(json.dumps(rep)); tmp.replace(case / 'complete.json')
    return rep


def _job(args):
    C, sys_name, sys_idx, n, noise, seed, out = args
    try:
        return run_case(C, sys_name, sys_idx, n, noise, seed, Path(out))
    except Exception:
        return {'failed': True, 'key': f'{sys_name}_n{n}_noise{noise}_seed{seed}', 'error': traceback.format_exc()[-2000:]}


def do_run(workers, suffix, smoke):
    import sysinfo
    L = check_lock(); C = cfg(); out = out_dir(suffix); out.mkdir(parents=True, exist_ok=True)
    if not (out / 'system_env.json').exists():
        (out / 'system_env.json').write_text(json.dumps(sysinfo.collect('cpu'), indent=2, ensure_ascii=False))
    seeds = C['seeds'][:2] if smoke else C['seeds']
    jobs = [(C, s, i, n, nz, sd, str(out)) for i, s in enumerate(C['systems']) for n in C['train_episodes'] for nz in C['label_noise'] for sd in seeds]
    if smoke:
        jobs = [j for j in jobs if j[3] == 12 and j[4] == 0.3]
    t0 = time.perf_counter()
    if workers > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(workers) as pool:
            reps = list(pool.imap_unordered(_job, jobs))
    else:
        reps = [_job(j) for j in jobs]
    failed = [r for r in reps if r.get('failed')]
    (out / 'failed.json').write_text(json.dumps(failed, indent=2))
    ok = [r for r in reps if not r.get('failed')]
    rows = [row for r in ok for row in r['rows']]
    if rows:
        with open(out / 'rows.csv', 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    (out / 'run_summary.json').write_text(json.dumps({'cases': len(jobs), 'ok': len(ok), 'failed': len(failed), 'smoke': smoke,
                                                       'seconds_this_invocation': time.perf_counter() - t0, 'protocol_lock': L}, indent=2))
    print(f'cases {len(jobs)}: ok {len(ok)}, failed {len(failed)}')


# ---- analysis (claims M1-M5 exactly as locked in plan.toml) ------------------------------------------------------
def cp(k, n, side, eps=0.05):
    from scipy.stats import beta
    if n == 0:
        return float('nan')
    if side == 'upper':
        return 1.0 if k >= n else float(beta.ppf(1 - eps, k + 1, n - k))
    return 0.0 if k == 0 else float(beta.ppf(eps, k, n - k + 1))


def do_analyze(suffix):
    from scipy.stats import wilcoxon
    C = cfg(); out = out_dir(suffix); A = out / 'analysis'; A.mkdir(exist_ok=True)
    reps = [json.loads((d / 'complete.json').read_text()) for d in sorted((out / 'cases').iterdir()) if (d / 'complete.json').exists()]
    rows = [r for rp in reps for r in rp['rows']]
    def sel(system, split, method):
        return [r for r in rows if r['system'] == system and r['split'] == split and r['method'] == method]
    rng = np.random.default_rng(C['bootstrap']['rng_seed']); B = C['bootstrap']['B']
    def boot_median(rs, key='rel_gain'):
        seeds = sorted({r['seed'] for r in rs}); by = {s: [r[key] for r in rs if r['seed'] == s] for s in seeds}
        meds = [np.median(np.concatenate([by[s] for s in rng.choice(seeds, len(seeds))])) for _ in range(B)]
        return float(np.median([r[key] for r in rs])), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))
    systems = list(C['systems']); res = {'cases': len(reps), 'expected_cases': len(systems) * len(C['train_episodes']) * len(C['label_noise']) * len(C['seeds'])}
    m1 = {}
    for s in systems:
        rs = sel(s, 'range', 'fixed_matrix_no_gate'); k = sum(r['worse'] for r in rs); n = len(rs); up = cp(k, n, 'upper')
        lo = cp(k, n, 'lower')   # supported only with evidence of >= 10% harm; absence of harm is not support
        v = 'supported' if lo >= 0.10 else ('not_supported' if up < 0.10 else 'inconclusive')
        m1[s] = {'worse': k, 'runs': n, 'share': k / n if n else float('nan'), 'lower95': lo, 'upper95': up, 'verdict': v}
    m2 = {}
    for s in systems:
        rs = [r for r in sel(s, 'id', 'ltt_correction') if r['applied']]; k = sum(r['worse'] for r in rs); n = len(rs); lo = cp(k, n, 'lower')
        up = cp(k, n, 'upper')   # supported only if the bound shows the share is at most delta
        v = 'inconclusive' if n < 10 else ('not_supported' if lo > 0.10 else ('supported' if up <= 0.10 else 'inconclusive'))
        m2[s] = {'accepted': n, 'worse': k, 'lower95': lo, 'upper95': up, 'verdict': v}
    f = 'franka_7dof'
    ltt = sel(f, 'id', 'ltt_correction'); med, lo3, hi3 = boot_median(ltt)
    p3 = float(wilcoxon(np.array([r['rel_gain'] for r in ltt]) - 0.05, alternative='greater').pvalue) if ltt else float('nan')
    diag = {r['key']: r['rel_gain'] for r in sel(f, 'id', 'ltt_diagonal_only')}
    diff = np.array([r['rel_gain'] - diag[r['key']] for r in ltt if r['key'] in diag])
    nz = diff[np.abs(diff) > 1e-12]
    p4 = float(wilcoxon(nz, alternative='greater').pvalue) if len(nz) >= 5 else 1.0
    ps = sorted([('M3', p3), ('M4', p4)], key=lambda t: t[1]); holm = {}
    for i, (name, p) in enumerate(ps):
        holm[name] = min(1.0, max(p * (len(ps) - i), max([holm[x] for x in holm] or [0])))
    m3 = {'median': med, 'boot95': [lo3, hi3], 'wilcoxon_p_vs_0.05': p3, 'holm_p': holm['M3'], 'verdict': 'not_supported' if lo3 < 0.05 else 'supported'}
    m4 = {'median_diff': float(np.median(diff)) if len(diff) else float('nan'), 'wilcoxon_p': p4, 'holm_p': holm['M4'],
          'verdict': 'not_supported' if (holm['M4'] >= 0.05 or (np.median(diff) if len(diff) else 0) < 0.05) else 'supported'}
    m5 = {}
    for s in systems:
        rp = [x for x in reps if x['system'] == s]; acc = [x for x in rp if x['ltt_pick'] != 'original']
        m5[s] = {'accepted': len(acc), 'full_matrix_share_of_accepted': (sum(x['ltt_pick'] == 'matrix' for x in acc) / len(acc)) if acc else float('nan'),
                 'acceptance_by_judge_n': {nj: sum(x['size_picks'][nj] != 'original' for x in rp) / len(rp) if rp else float('nan') for nj in ('36', '60', '100')}}
    res.update({'M1': m1, 'M2': m2, 'M3': m3, 'M4': m4, 'M5_descriptive': m5,
                'kinematic_max_correction': max([r['kinematic_max'] for r in rows if r['method'] in ('ltt_correction', 'fixed_matrix_no_gate')] or [float('nan')])})
    (A / 'claims.json').write_text(json.dumps(res, indent=2))
    summ = []
    for s in systems:
        for sp in SPLITS:
            for mth in METHODS:
                rs = sel(s, sp, mth)
                if not rs:
                    continue
                mz = np.mean([r['mse_Z'] for r in rs]); mm = np.mean([r['mse'] for r in rs])
                summ.append({'system': s, 'split': sp, 'method': mth, 'runs': len(rs), 'applied': sum(r['applied'] for r in rs),
                             'net_gain': 1 - mm / mz, 'median_rel_gain': float(np.median([r['rel_gain'] for r in rs])),
                             'worse': sum(r['worse'] for r in rs), 'max_harm': max([-r['rel_gain'] for r in rs] + [0.0])})
    with open(A / 'summary.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summ[0])); wr.writeheader(); wr.writerows(summ)
    # runs.csv in the experiment-scaffold / research-paper-loop column format
    with open(A / 'runs.csv', 'w', newline='') as fh:
        wr = csv.writer(fh); wr.writerow(['run_id', 'claim_id', 'config_hash', 'dataset', 'split', 'method', 'seed', 'metric', 'value', 'unit', 'elapsed_sec', 'status'])
        lock = json.loads(LOCK.read_text())['sha256'] if LOCK.exists() else {}
        chash = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()[:12]
        secs = {x['key']: x['seconds'] for x in reps}
        for r in rows:
            for metric in ('rel_gain', 'worse', 'applied'):
                wr.writerow([f"{r['key']}|{r['split']}|{r['method']}", '', chash, r['system'], r['split'], r['method'], r['seed'], metric, r[metric], '', round(secs[r['key']], 2), 'ok'])
    print(json.dumps({k: res[k] for k in ('cases', 'expected_cases')}), json.dumps({s: m1[s]['verdict'] for s in systems}),
          json.dumps({s: m2[s]['verdict'] for s in systems}), m3['verdict'], m4['verdict'])


def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--lock', action='store_true'); g.add_argument('--run', action='store_true'); g.add_argument('--analyze', action='store_true')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2)); ap.add_argument('--suffix', default='')
    ap.add_argument('--smoke', action='store_true', help='2 seeds x 3 systems x (n = 12, noise 0.3) into results_c1{suffix}_smoke')
    a = ap.parse_args()
    if a.lock:
        return do_lock()
    suffix = a.suffix + ('_smoke' if a.smoke else '')
    return do_run(a.workers, suffix, a.smoke) if a.run else do_analyze(suffix)


if __name__ == '__main__':
    sys.exit(main())
