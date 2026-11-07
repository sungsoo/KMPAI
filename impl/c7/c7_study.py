"""C7 study: magnitude-based and direction-wise acceptance (design/C7/plan.toml, locked 39289bba1c10).

    python c7_study.py --lock                                  # fix code + config hashes before main data
    python c7_study.py --run [--workers 8] [--suffix _mac]     # all cases (resumable); --smoke for a reduced subset
    python c7_study.py --analyze [--suffix _mac]               # claims N1-N5 -> results_c7{suffix}/analysis

Data, systems, learned model, retraining and candidate fitting are C1's (imported unchanged from impl/c1 and
impl/calibration.py). New here: the C7 seeds, n = 100 judge episodes (first 36 = small variant) and the gates:
  strict_ltt_c1      C1 rule (whole candidate, Clopper-Pearson on P(episode worse) <= 0.1, n = 36)
  mag_whole_gate     rule (1) on the whole candidate, n = 100
  mag_dir_gate       rule (1) + rule (2) fixed-sequence over eigen-directions, n = 100   (proposed)
  strict_dir_gate    rule (2) with the count rule (Clopper-Pearson <= 0.1), n = 100      (ablation)
  mag_dir_n36        proposed with n = 36 (descriptive variant N4)
  mag_dir_tail       proposed + tail condition P(ratio > 1 + kappa) upper bound <= beta (descriptive variant N4)
Failed cases are recorded in failed.json and counted as missing, never as 0.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; IMPL = HERE.parent; ROOT = IMPL.parent
for p in (IMPL, IMPL / 'c1', HERE):
    sys.path.insert(0, str(p))
CONFIG, LOCK = HERE / 'c7_config.json', HERE / 'c7_protocol_lock.json'
LOCKED = [HERE / 'c7_config.json', HERE / 'c7_rules.py', HERE / 'c7_study.py', IMPL / 'c1' / 'c1_models.py', IMPL / 'c1' / 'c1_study.py',
          IMPL / 'c1' / 'c1_config.json', IMPL / 'c1' / 'assets' / 'panda_nohand.xml', IMPL / 'calibration.py', ROOT / 'design' / 'C7' / 'plan.toml']
SPLITS = ('id', 'range')
GATES = ('strict_ltt_c1', 'mag_whole_gate', 'mag_dir_gate', 'strict_dir_gate', 'mag_dir_n36', 'mag_dir_tail')
METHODS = ('keep_Z', 'always_on', 'mean_loss_rule', 'equal_budget_retrain') + GATES


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cfg():
    c7 = json.loads(CONFIG.read_text()); c1 = json.loads((HERE / c7['c1_config']).read_text())
    return c7, c1


def out_dir(suffix=''):
    return IMPL / f'results_c7{suffix}'


def do_lock():
    if out_dir().exists():
        raise SystemExit('results_c7 already exists; the lock must precede the main data')
    L = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sha256': {str(p.relative_to(ROOT)): sha(p) for p in LOCKED}}
    LOCK.write_text(json.dumps(L, indent=2)); print('locked', len(L['sha256']), 'files')


def check_lock():
    if not LOCK.exists():
        raise SystemExit('c7_protocol_lock.json missing: run  python c7_study.py --lock  first')
    L = json.loads(LOCK.read_text()); bad = [k for k, h in L['sha256'].items() if sha(ROOT / k) != h]
    if bad:
        raise SystemExit(f'Protocol changed after locking: {bad}')
    return L


# ---- rule (2): ordered eigen-directions and nested partial matrices ------------------------------------------
def directions(a, kind, rv, ev, tol):
    """Ordered prefixes A_1, ..., A_m of I + sum_j mu_j v_j v_j^T (mu_j = lambda_j - 1), order fixed on C_s1 only."""
    d = a.shape[0]
    if kind in ('scalar', 'prior'):
        return [a] if np.max(np.abs(a - np.eye(d))) > tol else []
    if kind == 'diagonal':
        mu, V = np.diag(a) - 1.0, np.eye(d)
    else:
        lam, V = np.linalg.eigh((a + a.T) / 2); mu = lam - 1.0
    idx = [j for j in range(d) if abs(mu[j]) > tol]
    single = {j: float(np.mean((rv @ (np.eye(d) + mu[j] * np.outer(V[:, j], V[:, j])) - ev) ** 2)) for j in idx}
    order = sorted(idx, key=lambda j: (single[j], j))
    pref, acc = [], np.eye(d)
    for j in order:
        acc = acc + mu[j] * np.outer(V[:, j], V[:, j]); pref.append(acc.copy())
    return pref


def run_case(C7, C1, sys_name, sys_idx, n, noise, seed, out):
    import c1_study as S1
    from c1_models import System, stack, apply, kinematic
    from calibration import select
    from c7_rules import magnitude_gate, cp_upper
    key = f'{sys_name}_n{n}_noise{noise}_seed{seed}'; case = out / 'cases' / key
    if (case / 'complete.json').exists():
        return json.loads((case / 'complete.json').read_text())
    case.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()
    S = System(sys_name, C1['systems'][sys_name]); d = S.d; dt, H, sub = C1['dt'], C1['steps'], C1['truth_substeps']; E = C1['episodes']
    base = seed + 100000 * sys_idx
    gen = lambda k, m, rs=1.0: S.episodes(base + 1000 * k, m, H, dt, sub, rs)
    tr_s, tr_u = gen(1, max(C1['train_episodes'])); tr_s, tr_u = tr_s[:n], tr_u[:n]
    cf, cs1, cs2 = gen(2, E['calibration']), gen(3, E['selection']), gen(4, C7['judge_primary'])
    x, y = S.observed_residual(tr_s, tr_u, dt); xv, yv = S.observed_residual(*cs1, dt)
    rng = np.random.default_rng(base + 7); rms = np.sqrt((y ** 2).mean(0)); y_noisy = y + rng.normal(0, 1, y.shape) * noise * rms
    fZ, infoZ = S1.fit_mlp(x, y_noisy, xv, yv, d, C1['mlp'], seed)
    xr, yr = [x], [y_noisy]
    for blk in (cf, cs2):
        xb, yb = S.observed_residual(*blk, dt); xr.append(xb); yr.append(yb + rng.normal(0, 1, yb.shape) * noise * rms)
    fR, infoR = S1.fit_mlp(np.concatenate(xr), np.concatenate(yr), xv, yv, d, C1['mlp'], seed + 1)
    roll = lambda blk, f=None: S.rollout(blk[0][:, 0], blk[1], dt, f)
    pf, zf = roll(cf), roll(cf, fZ); p1, z1 = roll(cs1), roll(cs1, fZ); p2, z2 = roll(cs2), roll(cs2, fZ)
    yf, y1, y2 = cf[0][:, 1:], cs1[0][:, 1:], cs2[0][:, 1:]
    r, e = stack(zf - pf, d), stack(yf - pf, d); rv, ev = stack(z1 - p1, d), stack(y1 - p1, d)
    mats = {'prior': np.zeros((d, d)), 'original': np.eye(d)}
    for kind in ('scalar', 'diagonal', 'matrix'):
        mats[kind], _ = select(r, e, rv, ev, kind, C1['lambdas'])
    sel = {k: float(np.mean((rv @ a - ev) ** 2)) for k, a in mats.items()}
    order = ('prior', 'scalar', 'diagonal', 'matrix'); best = min(order, key=lambda k: (sel[k], order.index(k)))
    lz2 = S1.episode_loss(z2, y2)
    ratios = lambda a, nj: S1.episode_loss(apply(p2[:nj], z2[:nj], a, d), y2[:nj]) / lz2[:nj]
    tol = C7['tie_tol']; I = np.eye(d); n_p, n_s = C7['judge_primary'], C7['judge_small']
    def is_tie(a):
        return np.max(np.abs(a - I)) <= tol
    def strict_ok(a, nj, delta):
        k = int((ratios(a, nj) > 1).sum()); return cp_upper(k, nj, C7['eps']) <= delta
    def mag_ok(a, nj, tail=None):
        b = 1.0 if tail is None else tail['beta']; kap = 0.10 if tail is None else tail['kappa']
        return magnitude_gate(ratios(a, nj), C7['tau'], kap, b, C7['eps'], C7['clip'])[0]
    whole = mats[best]
    pref = directions(whole, best, rv, ev, tol)
    def seq(test):
        acc = I
        for a in pref:
            if not test(a):
                break
            acc = a
        return acc
    gates = {'strict_ltt_c1': whole if (not is_tie(whole) and strict_ok(whole, C7['strict_c1']['judge'], C7['strict_c1']['delta'])) else I,
             'mag_whole_gate': whole if (not is_tie(whole) and mag_ok(whole, n_p)) else I,
             'mag_dir_gate': seq(lambda a: mag_ok(a, n_p)),
             'strict_dir_gate': seq(lambda a: strict_ok(a, C7['strict_dir_ablation']['judge'], C7['strict_dir_ablation']['delta'])),
             'mag_dir_n36': seq(lambda a: mag_ok(a, n_s)),
             'mag_dir_tail': seq(lambda a: mag_ok(a, n_p, C7['tail_variant']))}
    n_dir = {g: (0 if is_tie(a) else next((k + 1 for k, pa in enumerate(pref) if np.allclose(pa, a, rtol=0.0, atol=1e-12)), -1)) for g, a in gates.items()}
    mean_pick = whole if sel[best] <= 0.99 * sel['original'] else I
    rows = []
    for si, split in enumerate(SPLITS):
        rs = 1.0 if split == 'id' else C1['range_scale']
        td = gen(6 + si, E['test'], rs); truth = td[0][:, 1:]; s0 = td[0][:, 0]
        p, z = roll(td), roll(td, fZ); rt = roll(td, fR)
        preds = {'keep_Z': z, 'always_on': apply(p, z, whole, d), 'mean_loss_rule': apply(p, z, mean_pick, d), 'equal_budget_retrain': rt}
        preds.update({g: apply(p, z, a, d) for g, a in gates.items()})
        lz = S1.episode_loss(z, truth); mz = float(((z - truth) ** 2).mean())
        for mth, pred in preds.items():
            lp = S1.episode_loss(pred, truth); m = float(((pred - truth) ** 2).mean())
            applied = int(mth not in ('keep_Z',) and (mth in ('always_on', 'equal_budget_retrain') or (mth == 'mean_loss_rule' and not is_tie(mean_pick)) or (mth in gates and not is_tie(gates[mth]))))
            if mth == 'always_on':
                applied = int(not is_tie(whole))
            rows.append({'key': key, 'system': sys_name, 'd': d, 'n': n, 'noise': noise, 'seed': seed, 'split': split, 'method': mth,
                         'applied': applied, 'mse': m, 'mse_Z': mz, 'rel_gain': (mz - m) / mz, 'worse': int(m > mz * (1 + 1e-12)),
                         'test_mean_ratio': float(np.mean(np.minimum(lp / lz, C7['clip']))), 'ep_worse': int((lp > lz).sum()), 'ep_n': len(lp),
                         'kinematic_max': kinematic(pred, s0, dt, d)})
    rep = {'key': key, 'system': sys_name, 'd': d, 'n': n, 'noise': noise, 'seed': seed, 'best': best, 'n_directions': len(pref),
           'accepted_directions': n_dir, 'selection_loss': sel, 'judge_ratio_whole': ratios(whole, n_p).tolist(),
           'judge_ratio_prefixes': [ratios(a, n_p).tolist() for a in pref], 'matrices': {k: v.tolist() for k, v in mats.items()},
           'gates': {g: a.tolist() for g, a in gates.items()}, 'mlp_Z': infoZ, 'mlp_retrain': infoR, 'rows': rows, 'seconds': time.perf_counter() - t0}
    tmp = case / 'complete.json.tmp'; tmp.write_text(json.dumps(rep)); tmp.replace(case / 'complete.json')
    return rep


def _job(args):
    C7, C1, sys_name, sys_idx, n, noise, seed, out = args
    try:
        return run_case(C7, C1, sys_name, sys_idx, n, noise, seed, Path(out))
    except Exception:
        return {'failed': True, 'key': f'{sys_name}_n{n}_noise{noise}_seed{seed}', 'error': traceback.format_exc()[-2000:]}


def do_run(workers, suffix, smoke):
    import sysinfo
    L = check_lock(); C7, C1 = cfg(); out = out_dir(suffix); out.mkdir(parents=True, exist_ok=True)
    if not (out / 'system_env.json').exists():
        (out / 'system_env.json').write_text(json.dumps(sysinfo.collect('cpu'), indent=2, ensure_ascii=False))
    seeds = C7['seeds'][:2] if smoke else C7['seeds']
    jobs = [(C7, C1, s, i, n, nz, sd, str(out)) for i, s in enumerate(C1['systems']) for n in C1['train_episodes'] for nz in C1['label_noise'] for sd in seeds]
    if smoke:
        jobs = [j for j in jobs if j[4] == 12 and j[5] == 0.3]
    t0 = time.perf_counter()
    if workers > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(workers) as pool:
            reps = list(pool.imap_unordered(_job, jobs))
    else:
        reps = [_job(j) for j in jobs]
    failed = [r for r in reps if r.get('failed')]; ok = [r for r in reps if not r.get('failed')]
    (out / 'failed.json').write_text(json.dumps(failed, indent=2))
    rows = [row for r in ok for row in r['rows']]
    if rows:
        with open(out / 'rows.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / 'run_summary.json').write_text(json.dumps({'cases': len(jobs), 'ok': len(ok), 'failed': len(failed), 'smoke': smoke,
                                                       'seconds_this_invocation': time.perf_counter() - t0, 'protocol_lock': L}, indent=2))
    print(f'cases {len(jobs)}: ok {len(ok)}, failed {len(failed)}')


# ---- analysis (N1-N5 exactly as locked) ---------------------------------------------------------------------
def cpb(k, n, side, eps=0.05):
    from scipy.stats import beta
    if n == 0:
        return float('nan')
    if side == 'upper':
        return 1.0 if k >= n else float(beta.ppf(1 - eps, k + 1, n - k))
    return 0.0 if k == 0 else float(beta.ppf(eps, k, n - k + 1))


def do_analyze(suffix):
    from scipy.stats import wilcoxon, binomtest
    C7, C1 = cfg(); out = out_dir(suffix); A = out / 'analysis'; A.mkdir(exist_ok=True)
    reps = [json.loads((d / 'complete.json').read_text()) for d in sorted((out / 'cases').iterdir()) if (d / 'complete.json').exists()]
    rows = [r for rp in reps for r in rp['rows']]
    sel = lambda s, sp, m: [r for r in rows if r['system'] == s and r['split'] == sp and r['method'] == m]
    rng = np.random.default_rng(C7['bootstrap']['rng_seed']); B = C7['bootstrap']['B']
    systems = list(C1['systems']); f = 'franka_7dof'
    res = {'cases': len(reps), 'expected_cases': len(systems) * len(C1['train_episodes']) * len(C1['label_noise']) * len(C7['seeds'])}
    rs = sel(f, 'id', 'mag_dir_gate'); g = np.array([r['rel_gain'] for r in rs])
    seeds = sorted({r['seed'] for r in rs}); by = {s: [r['rel_gain'] for r in rs if r['seed'] == s] for s in seeds}
    meds = [np.median(np.concatenate([by[s] for s in rng.choice(seeds, len(seeds))])) for _ in range(B)] if seeds else [np.nan]
    lo, hi = float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))
    p1 = float(wilcoxon(g - 0.05, alternative='greater').pvalue) if len(g) and np.any(np.abs(g - 0.05) > 0) else 1.0
    dirA = {r['key']: r['applied'] for r in rs}; whA = {r['key']: r['applied'] for r in sel(f, 'id', 'mag_whole_gate')}
    b_ = sum(1 for k in dirA if dirA[k] and not whA.get(k, 0)); c_ = sum(1 for k in dirA if whA.get(k, 0) and not dirA[k])
    p3 = float(binomtest(b_, b_ + c_, 0.5, alternative='greater').pvalue) if b_ + c_ > 0 else 1.0
    ps = sorted([('N1', p1), ('N3', p3)], key=lambda t: t[1]); holm = {}
    for i, (nm, p) in enumerate(ps):
        holm[nm] = min(1.0, max(p * (len(ps) - i), max(list(holm.values()) or [0])))
    nruns = len(rs); diff = sum(dirA.values()) - sum(whA.values())
    res['N1'] = {'median': float(np.median(g)) if len(g) else float('nan'), 'boot95': [lo, hi], 'wilcoxon_p_vs_0.05': p1, 'holm_p': holm['N1'],
                 'applied': int(sum(dirA.values())), 'runs': nruns, 'verdict': 'not_supported' if lo < 0.05 else 'supported'}
    res['N3'] = {'applied_dir': int(sum(dirA.values())), 'applied_whole': int(sum(whA.values())), 'difference': int(diff), 'discordant_b_c': [b_, c_],
                 'mcnemar_p': p3, 'holm_p': holm['N3'],
                 'verdict': 'supported' if (diff >= C7['n3_min_runs_share'] * nruns and holm['N3'] < 0.05) else 'not_supported'}
    n2 = {}
    for s in systems:
        ap = [r for r in sel(s, 'id', 'mag_dir_gate') if r['applied']]; k = sum(r['test_mean_ratio'] > 1 for r in ap); n = len(ap)
        lo2, up2 = cpb(k, n, 'lower'), cpb(k, n, 'upper')
        v = 'inconclusive' if n < 10 else ('not_supported' if lo2 > 0.10 else ('supported' if up2 <= 0.10 else 'inconclusive'))
        n2[s] = {'applied': n, 'mean_ratio_worse': int(k), 'lower95': lo2, 'upper95': up2, 'verdict': v}
    res['N2'] = n2
    n4, n5 = {}, {}
    for s in systems:
        n4[s] = {}; n5[s] = {}
        for gname in GATES + ('mean_loss_rule', 'always_on'):
            idr = sel(s, 'id', gname); rr = sel(s, 'range', gname)
            n4[s][gname] = {'applied': sum(r['applied'] for r in idr), 'runs': len(idr), 'median_rel_gain_id': float(np.median([r['rel_gain'] for r in idr])) if idr else float('nan'),
                            'net_gain_id': float(1 - np.mean([r['mse'] for r in idr]) / np.mean([r['mse_Z'] for r in idr])) if idr else float('nan')}
            apr = [r for r in rr if r['applied']]
            n5[s][gname] = {'applied': len(apr), 'worse': sum(r['worse'] for r in apr), 'max_harm': max([-r['rel_gain'] for r in apr] + [0.0])}
    res['N4_descriptive'] = n4; res['N5_descriptive'] = n5
    res['kinematic_max_gates'] = max([r['kinematic_max'] for r in rows if r['method'] in GATES] or [float('nan')])
    (A / 'claims.json').write_text(json.dumps(res, indent=2))
    summ = []
    for s in systems:
        for sp in SPLITS:
            for m in METHODS:
                rr = sel(s, sp, m)
                if rr:
                    summ.append({'system': s, 'split': sp, 'method': m, 'runs': len(rr), 'applied': sum(r['applied'] for r in rr),
                                 'net_gain': 1 - np.mean([r['mse'] for r in rr]) / np.mean([r['mse_Z'] for r in rr]),
                                 'median_rel_gain': float(np.median([r['rel_gain'] for r in rr])), 'worse': sum(r['worse'] for r in rr),
                                 'max_harm': max([-r['rel_gain'] for r in rr] + [0.0])})
    with open(A / 'summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    with open(A / 'runs.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['run_id', 'claim_id', 'config_hash', 'dataset', 'split', 'method', 'seed', 'metric', 'value', 'unit', 'elapsed_sec', 'status'])
        lock = json.loads(LOCK.read_text())['sha256'] if LOCK.exists() else {}; ch = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()[:12]
        secs = {x['key']: x['seconds'] for x in reps}
        for r in rows:
            for metric in ('rel_gain', 'worse', 'applied', 'test_mean_ratio'):
                w.writerow([f"{r['key']}|{r['split']}|{r['method']}", '', ch, r['system'], r['split'], r['method'], r['seed'], metric, r[metric], '', round(secs[r['key']], 2), 'ok'])
    print(json.dumps({k: res[k] for k in ('cases', 'expected_cases')}), 'N1', res['N1']['verdict'], 'N2', json.dumps({s: n2[s]['verdict'] for s in systems}), 'N3', res['N3']['verdict'])


def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--lock', action='store_true'); g.add_argument('--run', action='store_true'); g.add_argument('--analyze', action='store_true')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2)); ap.add_argument('--suffix', default='')
    ap.add_argument('--smoke', action='store_true', help='2 seeds x 3 systems x (n = 12, noise 0.3) into results_c7{suffix}_smoke')
    a = ap.parse_args()
    if a.lock:
        return do_lock()
    suffix = a.suffix + ('_smoke' if a.smoke else '')
    return do_run(a.workers, suffix, a.smoke) if a.run else do_analyze(suffix)


if __name__ == '__main__':
    sys.exit(main())
