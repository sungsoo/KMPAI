"""Summaries of the pre-registered confirmatory run (v14) -> tidy CSV files.

    python confirm_analysis.py --results results_confirm

Writes into <results>/analysis/:
  summary.csv   one row per (split, method): MSE, worse rate with 95% seed-cluster bootstrap CI,
                net gain (relative mean-MSE reduction vs. always keeping Z) with CI, apply rate,
                gain when applied, kinematic max error, seconds
  curves.csv    per (split, method, step): share of runs worse than Z, RMSE ratio to Z
  numbers.csv   name,value,fmt rows for csv2macros.py (every number quoted in the paper)
  judge.csv     LTT statistics per case (k worse on C_s2, Clopper-Pearson upper bound, pick)
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

METHODS = ['keep_original', 'fixed_matrix', 'mean_rule', 'ltt', 'pz_select', 'bates_granger', 'equal_budget_retrain', 'oracle', 'prior']
SPLITS = ['id', 'ood_scale', 'ood_damp']
B = 2000


def load(res):
    reps = [json.loads((d / 'complete.json').read_text()) for d in sorted((res / 'cases').iterdir())]
    rows = [r for rep in reps for r in rep['rows']]
    return reps, rows


def stat_block(rows, reps, split, method, rng):
    sel = [r for r in rows if r['split'] == split and r['method'] == method]
    base = {r['key']: r['mse'] for r in rows if r['split'] == split and r['method'] == 'keep_original'}
    keys = [r['key'] for r in sel]; m = np.array([r['mse'] for r in sel]); z = np.array([base[k] for k in keys])
    seeds = np.array([r['seed'] for r in sel]); useeds = np.unique(seeds)
    worse = (m > z * (1 + 1e-12)).astype(float); gain_rel = (z - m) / z
    applied = np.array([r['applied'] for r in sel], dtype=float)
    def worse_rate(idx): return worse[idx].mean()
    def net_gain(idx): return 1 - m[idx].mean() / z[idx].mean()
    boots_w, boots_g = [], []
    groups = {s: np.where(seeds == s)[0] for s in useeds}
    for _ in range(B):
        pick = rng.choice(useeds, size=len(useeds), replace=True)
        idx = np.concatenate([groups[s] for s in pick])
        boots_w.append(worse_rate(idx)); boots_g.append(net_gain(idx))
    allidx = np.arange(len(m))
    rep_by = {rp['key']: rp for rp in reps}
    if method == 'keep_original':
        sec = 0.0
    elif method == 'equal_budget_retrain':
        sec = float(np.mean([rep_by[k]['seconds']['retrain'] for k in keys]))
    elif method in ('oracle',):
        sec = float('nan')
    else:
        sec = float(np.mean([rep_by[k]['seconds']['posthoc_calibration'] for k in keys]))
    ga = gain_rel[applied == 1]
    return {'split': split, 'method': method, 'runs': len(m), 'mse': float(m.mean()),
            'worse_rate': worse_rate(allidx), 'worse_lo': float(np.percentile(boots_w, 2.5)), 'worse_hi': float(np.percentile(boots_w, 97.5)),
            'net_gain': net_gain(allidx), 'gain_lo': float(np.percentile(boots_g, 2.5)), 'gain_hi': float(np.percentile(boots_g, 97.5)),
            'apply_rate': float(applied.mean()) if method not in ('equal_budget_retrain', 'prior') else float('nan'),
            'gain_when_applied': float(ga.mean()) if len(ga) and method not in ('equal_budget_retrain',) else float('nan'),
            'kinematic_max': float(max(r['kinematic_max'] for r in sel)), 'seconds': sec}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results_confirm'); a = ap.parse_args()
    res = Path(a.results).resolve(); out = res / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    reps, rows = load(res); rng = np.random.default_rng(20261019)
    summ = [stat_block(rows, reps, s, mth, rng) for s in SPLITS for mth in METHODS]
    with open(out / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    # per-step curves
    cur = []
    for s in SPLITS:
        zc = np.array([np.load(res / 'cases' / rp['key'] / 'curves.npz')[f'{s}__keep_original'] for rp in reps])
        for mth in METHODS:
            mc = np.array([np.load(res / 'cases' / rp['key'] / 'curves.npz')[f'{s}__{mth}'] for rp in reps])
            share = (mc > zc * (1 + 1e-12)).mean(0); ratio = np.sqrt(mc.mean(0) / zc.mean(0))
            for h in range(mc.shape[1]):
                cur.append({'split': s, 'method': mth, 'step': h + 1, 'time': round((h + 1) * 0.05, 4), 'worse_share': float(share[h]), 'rmse_ratio': float(ratio[h])})
    with open(out / 'curves.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cur[0])); w.writeheader(); w.writerows(cur)
    with open(out / 'judge.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['key', 'best_candidate', 'judge_worse', 'judge_n', 'cp_upper', 'ltt_pick', 'mean_rule_pick', 'rotation_err'])
        w.writeheader()
        for rp in reps:
            w.writerow({'key': rp['key'], 'best_candidate': rp['best_candidate'], 'judge_worse': rp['judge_worse'], 'judge_n': rp['judge_n'],
                        'cp_upper': rp['cp_upper'], 'ltt_pick': rp['ltt_pick'], 'mean_rule_pick': rp['mean_rule_pick'], 'rotation_err': rp['rotation_equivariance_max_error']})
    # numbers for the paper
    S = {(r['split'], r['method']): r for r in summ}
    cfg = json.loads((Path(__file__).resolve().parent / 'confirm_config.json').read_text())
    num = [('NRuns', len(reps), 'd'), ('NSeeds', len(cfg['seeds']), 'd'), ('NJudge', cfg['judge_episodes'], 'd'),
           ('NTest', cfg['test_episodes'], 'd'), ('LttDelta', cfg['ltt_delta'], '.1f'), ('LttEps', cfg['ltt_epsilon'], '.2f'),
           ('OodScale', cfg['ood_scale'], '.1f'), ('OodDamp', cfg['ood_extra_damping'], '.1f'),
           ('RotErrMax', max(rp['rotation_equivariance_max_error'] for rp in reps), 'sci1'),
           ('KinFixedMax', max(S[(s, 'fixed_matrix')]['kinematic_max'] for s in SPLITS), 'sci1'),
           ('KinBGMax', max(S[(s, 'bates_granger')]['kinematic_max'] for s in SPLITS), 'sci1')]
    tag = {'keep_original': 'Keep', 'fixed_matrix': 'Fixed', 'mean_rule': 'Mean', 'ltt': 'Ltt', 'pz_select': 'Pz',
           'bates_granger': 'Bg', 'equal_budget_retrain': 'Retrain', 'oracle': 'Oracle', 'prior': 'Prior'}
    stag = {'id': 'Id', 'ood_scale': 'Scale', 'ood_damp': 'Damp'}
    for (s, mth), r in S.items():
        t = stag[s] + tag[mth]
        num += [(t + 'Mse', r['mse'], '.4f'), (t + 'Worse', 100 * r['worse_rate'], '.1f'), (t + 'WorseLo', 100 * r['worse_lo'], '.1f'),
                (t + 'WorseHi', 100 * r['worse_hi'], '.1f'), (t + 'Gain', 100 * r['net_gain'], '.1f'),
                (t + 'GainLo', 100 * r['gain_lo'], '.1f'), (t + 'GainHi', 100 * r['gain_hi'], '.1f')]
        if r['apply_rate'] == r['apply_rate']: num.append((t + 'Apply', 100 * r['apply_rate'], '.0f'))
        if r['gain_when_applied'] == r['gain_when_applied']: num.append((t + 'GainApplied', 100 * r['gain_when_applied'], '.1f'))
    sec_post = np.mean([rp['seconds']['posthoc_calibration'] for rp in reps]); sec_re = np.mean([rp['seconds']['retrain'] for rp in reps])
    num += [('SecPosthoc', sec_post, '.2f'), ('SecRetrain', sec_re, '.2f'), ('CostRatio', sec_re / sec_post, '.1f')]
    lt = [rp for rp in reps]; num.append(('LttApplyCount', sum(rp['ltt_pick'] != 'original' for rp in lt), 'd'))
    with open(out / 'numbers.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['name', 'value', 'fmt']); w.writerows([(n, repr(float(v)), fm) for n, v, fm in num])
    for r in summ:
        if r['split'] != 'id':
            print(f"{r['split']:9s} {r['method']:22s} mse={r['mse']:.4f} worse={100*r['worse_rate']:5.1f}% [{100*r['worse_lo']:.1f},{100*r['worse_hi']:.1f}] gain={100*r['net_gain']:6.2f}% apply={r['apply_rate']:.2f} gA={r['gain_when_applied']:.3f} kin={r['kinematic_max']:.1e} sec={r['seconds']:.2f}")


if __name__ == '__main__':
    main()
