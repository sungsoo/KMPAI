"""Round-2 summaries (v15): exact Clopper-Pearson intervals, conditional (applied) worse
rates, episode-level check of Proposition 3, candidate frequencies, contraction vs.
unconstrained matrix, retrain+LTT and judge-size analyses, timing with 3 significant digits.

    python confirm_r2_analysis.py --results results_confirm --r2 results_confirm_r2

Writes <r2>/analysis/table2.csv, extra.csv, numbers_r2.csv. Rates are counted over runs
(80 per split); runs share seeds, so intervals are approximate (stated in the paper).
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent
SPLITS = ['id', 'ood_scale', 'ood_damp']


def cp2(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def cp1(k, n, eps):
    return 1.0 if k >= n else float(beta.ppf(1 - eps, k + 1, n - k))


def read(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def boot_gain(m, z, seeds, rng, B=2000):
    us = np.unique(seeds); g = {s: np.where(seeds == s)[0] for s in us}; out = []
    for _ in range(B):
        idx = np.concatenate([g[s] for s in rng.choice(us, len(us), replace=True)])
        out.append(1 - m[idx].mean() / z[idx].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results_confirm'); ap.add_argument('--r2', default='results_confirm_r2')
    a = ap.parse_args(); res = (ROOT / a.results).resolve(); r2 = (ROOT / a.r2).resolve(); out = r2 / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'confirm_config.json').read_text()); rng = np.random.default_rng(20261020)
    base = read(res / 'cases.csv'); extra = read(r2 / 'rows.csv'); cases = read(r2 / 'cases.csv')
    rows = [dict(r, mse=float(r['mse']), applied=int(r['applied']), seed=int(r['seed']), n=int(r['n'])) for r in base]
    z = {(r['key'], r['split']): r['mse'] for r in rows if r['method'] == 'keep_original'}
    zr = {}
    for r in extra:  # deterministic re-derivation; agrees with stored results to ~1e-8 relative (float32 rollouts)
        rows.append({'key': r['key'], 'seed': int(r['seed']), 'n': int(r['n']), 'split': r['split'], 'method': 'ltt_r2' if r['method'] == 'ltt' else r['method'],
                     'mse': float(r['mse']), 'applied': int(r['applied']), 'episode_worse_rate': float(r['episode_worse_rate']), 'src': 'r2'})
        zr[(r['key'], r['split'])] = float(r['mse_original'])
        assert abs(float(r['mse_original']) - z[(r['key'], r['split'])]) <= 1e-6 * z[(r['key'], r['split'])]
    zof = lambda r: zr[(r['key'], r['split'])] if r.get('src') == 'r2' else z[(r['key'], r['split'])]
    def block(method, split):
        sel = [r for r in rows if r['method'] == method and r['split'] == split]
        m = np.array([r['mse'] for r in sel]); zz = np.array([zof(r) for r in sel]); s = np.array([r['seed'] for r in sel])
        w = m > zz * (1 + 1e-12); ap_ = np.array([r['applied'] for r in sel]) == 1
        k, N = int(w.sum()), len(w); lo, hi = cp2(k, N); glo, ghi = boot_gain(m, zz, s, rng)
        ka, na = int((w & ap_).sum()), int(ap_.sum())
        nn = np.array([r['n'] for r in sel])
        g12 = 1 - m[nn == 12].mean() / zz[nn == 12].mean(); g36 = 1 - m[nn == 36].mean() / zz[nn == 36].mean()
        return {'method': method, 'split': split, 'runs': N, 'worse_k': k, 'worse_rate': k / N, 'worse_lo': lo, 'worse_hi': hi,
                'net_gain': float(1 - m.mean() / zz.mean()), 'gain_lo': glo, 'gain_hi': ghi, 'applied': na, 'apply_rate': na / N,
                'worse_applied_k': ka, 'worse_given_applied': (ka / na) if na else float('nan'), 'gain_n12': float(g12), 'gain_n36': float(g36)}
    methods = ['keep_original', 'fixed_matrix', 'bates_granger', 'mean_rule', 'ltt', 'equal_budget_retrain', 'retrain_ltt', 'oracle',
               'unconstrained_fixed', 'ltt_n36', 'ltt_n60', 'ltt_n100', 'pz_select', 'retrain_no_judge']
    T = [block(mm, s) for mm in methods for s in SPLITS]
    with open(out / 'table2.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(T[0])); wr.writeheader(); wr.writerows(T)
    S = {(t['method'], t['split']): t for t in T}
    # Proposition 3 check: applied LTT runs, ID episode-level worse rate <= delta
    ltt_id = [r for r in rows if r['method'] == 'ltt_r2' and r['split'] == 'id' and r['applied'] == 1]
    ep = np.array([r['episode_worse_rate'] for r in ltt_id]); ok = (ep <= cfg['ltt_delta'])
    nt = cfg['test_episodes']; kk = np.rint(ep * nt).astype(int)
    viol = int(sum(cp2(int(k_), nt)[0] > cfg['ltt_delta'] for k_ in kk))   # 95% CP lower bound above delta = significant violation
    pooled = float(kk.sum() / (nt * len(kk))) if len(kk) else float('nan')
    pooled_lo, pooled_hi = cp2(int(kk.sum()), int(nt * len(kk))) if len(kk) else (float('nan'), float('nan'))
    picks = {}
    for c in cases:
        if c['ltt_pick'] != 'original': picks[c['ltt_pick']] = picks.get(c['ltt_pick'], 0) + 1
    n = cfg['judge_episodes']; eps = cfg['ltt_epsilon']; d = cfg['ltt_delta']
    nmin1 = next(m for m in range(2, 500) if cp1(1, m, eps) <= d)
    rep = json.loads(next((res / 'cases').iterdir()).joinpath('complete.json').read_text())
    lock = json.loads((ROOT / 'confirm_protocol_lock.json').read_text())
    sec_post = np.mean([json.loads((p / 'complete.json').read_text())['seconds']['posthoc_calibration'] for p in (res / 'cases').iterdir()])
    sec_re = np.mean([json.loads((p / 'complete.json').read_text())['seconds']['retrain'] for p in (res / 'cases').iterdir()])
    sp, sr = float(f'{sec_post:.3g}'), float(f'{sec_re:.3g}')
    ex = {'u0': cp1(0, n, eps), 'u1': cp1(1, n, eps), 'u2': cp1(2, n, eps), 'n_min_k1': nmin1, 'ltt_applied': len(ltt_id),
          'prop3_ok': int(ok.sum()), 'prop3_rate': float(ok.mean()), 'episode_worse_mean': float(ep.mean()), 'episode_worse_max': float(ep.max()),
          'prop3_violations': viol, 'pooled_episode_worse': pooled, 'pooled_lo': pooled_lo, 'pooled_hi': pooled_hi,
          'best_counts': {c: sum(1 for q in (res / 'cases').iterdir() if json.loads((q / 'complete.json').read_text())['best_candidate'] == c) for c in ('prior', 'scalar', 'diagonal', 'matrix')},
          'picks': picks, 'diag_rot_max': max(float(c['diag_rotation_error']) for c in cases),
          'matrix_rot_max': max(float(c['matrix_rotation_error']) for c in cases),
          'retrain_ltt_applied': sum(int(c['retrain_ltt_applied']) for c in cases), 'sec_posthoc_3sf': sp, 'sec_retrain_3sf': sr,
          'cost_ratio_3sf': sr / sp, 'lock_config8': lock['confirm_config_sha256'][:8], 'lock_code8': lock['confirm_py_sha256'][:8]}
    (out / 'extra.json').write_text(json.dumps(ex, indent=2))
    num = [('UZero', ex['u0'], '.3f'), ('UOne', ex['u1'], '.3f'), ('UTwo', ex['u2'], '.3f'), ('NMinOne', nmin1, 'd'),
           ('PropOk', ex['prop3_ok'], 'd'), ('PropApplied', ex['ltt_applied'], 'd'), ('PropRate', 100 * ex['prop3_rate'], '.0f'),
           ('EpWorseMean', 100 * ex['episode_worse_mean'], '.1f'), ('PropViol', viol, 'd'), ('EpPooled', 100 * pooled, '.1f'), ('EpPooledLo', 100 * pooled_lo, '.1f'), ('EpPooledHi', 100 * pooled_hi, '.1f'), ('EpWorseMax', 100 * ex['episode_worse_max'], '.1f'),
           ('PickMatrix', picks.get('matrix', 0), 'd'), ('PickScalar', picks.get('scalar', 0), 'd'), ('PickDiag', picks.get('diagonal', 0), 'd'),
           ('PickPrior', picks.get('prior', 0), 'd'), ('BestMatrix', ex['best_counts']['matrix'], 'd'), ('BestDiag', ex['best_counts']['diagonal'], 'd'), ('BestScalar', ex['best_counts']['scalar'], 'd'), ('BestPrior', ex['best_counts']['prior'], 'd'), ('DiagRotMax', ex['diag_rot_max'], 'sci1'), ('MatRotMax', ex['matrix_rot_max'], 'sci1'),
           ('RetrainLttApplied', ex['retrain_ltt_applied'], 'd'), ('SecPost', sp, '.3g'), ('SecRe', sr, '.3g'), ('CostRatioR', sr / sp, '.1f')]
    tag = {'keep_original': 'Keep', 'fixed_matrix': 'Fixed', 'bates_granger': 'Bg', 'mean_rule': 'Mean', 'ltt': 'Ltt',
           'equal_budget_retrain': 'Retrain', 'retrain_ltt': 'RetrainLtt', 'oracle': 'Oracle', 'unconstrained_fixed': 'Uncon',
           'ltt_n36': 'LttNa', 'ltt_n60': 'LttNb', 'ltt_n100': 'LttNc', 'pz_select': 'Pz', 'retrain_no_judge': 'RetrainNoJudge'}
    stag = {'id': 'Id', 'ood_scale': 'Scale', 'ood_damp': 'Damp'}
    for (mth, s), t in S.items():
        p = 'V' + stag[s] + tag[mth]
        num += [(p + 'K', t['worse_k'], 'd'), (p + 'W', 100 * t['worse_rate'], '.0f'), (p + 'Lo', 100 * t['worse_lo'], '.1f'), (p + 'Hi', 100 * t['worse_hi'], '.1f'),
                (p + 'G', 100 * t['net_gain'], '.1f'), (p + 'GLo', 100 * t['gain_lo'], '.1f'), (p + 'GHi', 100 * t['gain_hi'], '.1f'),
                (p + 'App', t['applied'], 'd'), (p + 'AppK', t['worse_applied_k'], 'd'), (p + 'GTw', 100 * t['gain_n12'], '.1f'), (p + 'GTh', 100 * t['gain_n36'], '.1f')]
        if t['applied']: num.append((p + 'Cond', 100 * t['worse_given_applied'], '.1f'))
    with open(out / 'numbers_r2.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['name', 'value', 'fmt']); w.writerows([(a_, repr(float(v)), fm) for a_, v, fm in num])
    print(json.dumps(ex, indent=1))
    for t in T:
        if t['method'] in ('ltt', 'retrain_ltt', 'equal_budget_retrain', 'unconstrained_fixed', 'fixed_matrix', 'ltt_n60', 'ltt_n100'):
            print(f"{t['method']:20s} {t['split']:9s} worse {t['worse_k']}/80 [{100*t['worse_lo']:.1f},{100*t['worse_hi']:.1f}] gain {100*t['net_gain']:.2f} [{100*t['gain_lo']:.1f},{100*t['gain_hi']:.1f}] app {t['applied']} condW {t['worse_applied_k']} g12 {100*t['gain_n12']:.2f} g36 {100*t['gain_n36']:.2f}")


if __name__ == '__main__':
    main()
