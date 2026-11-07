"""Summaries of confirm_r3.py runs -> <results>/analysis/{summary.csv, detector.csv, numbers.csv, REPORT.md}.

    python confirm_r3_analysis.py --results results_confirm_r3            # confirmatory (new seeds)
    python confirm_r3_analysis.py --results results_confirm_r3_posthoc    # supplementary (v14 seeds)

Worse: run-level evaluation MSE > keep-original MSE (episodes 9..48).  Intervals: worse rate with the exact
two-sided Clopper-Pearson 95% interval (runs share seeds -> approximate), net gain with a 95% seed-cluster
bootstrap (B=2000).
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.stats import beta

SPLITS = ('id', 'ood_scale', 'ood_damp'); METHODS = ('keep_original', 'ltt', 'ltt_detect', 'conformal_gate'); B = 2000


def cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)); hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', required=True); a = ap.parse_args()
    res = Path(a.results).resolve(); out = res / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(res / 'rows.csv')))
    reps = {d.name: json.loads((d / 'complete.json').read_text()) for d in (res / 'cases').iterdir()}
    rng = np.random.default_rng(20261022); summ = []
    for s in SPLITS:
        for mth in METHODS:
            sel = [r for r in rows if r['split'] == s and r['method'] == mth]
            m = np.array([float(r['mse']) for r in sel]); z = np.array([float(r['mse_original']) for r in sel])
            seeds = np.array([int(r['seed']) for r in sel]); us = np.unique(seeds); groups = {k: np.where(seeds == k)[0] for k in us}
            worse = m > z * (1 + 1e-12); applied = np.array([int(r['applied']) for r in sel]) == 1
            k, n = int(worse.sum()), len(m); lo, hi = cp(k, n)
            boots = []
            for _ in range(B):
                idx = np.concatenate([groups[g] for g in rng.choice(us, size=len(us), replace=True)])
                boots.append(1 - m[idx].mean() / z[idx].mean())
            summ.append({'split': s, 'method': mth, 'runs': n, 'worse': k, 'worse_rate': k / n, 'worse_lo': lo, 'worse_hi': hi,
                         'net_gain': float(1 - m.mean() / z.mean()), 'gain_lo': float(np.percentile(boots, 2.5)), 'gain_hi': float(np.percentile(boots, 97.5)),
                         'applied_runs': int(applied.sum()), 'applied_worse': int((worse & applied).sum()),
                         'episode_applied_rate': float(np.mean([float(r['episode_applied_rate']) for r in sel])),
                         'episode_worse_rate': float(np.mean([float(r['episode_worse_rate']) for r in sel]))})
    with open(out / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    det = []
    for s in SPLITS:
        flags = np.array([rp['splits'][s]['shift_detected'] for rp in reps.values()])
        ltt_on = np.array([rp['ltt_pick'] != 'original' for rp in reps.values()])
        det.append({'split': s, 'runs': len(flags), 'detected': int(flags.sum()), 'detected_rate': float(flags.mean()),
                    'ltt_applied_runs': int(ltt_on.sum()), 'detected_among_ltt_applied': int((flags & ltt_on).sum())})
    with open(out / 'detector.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(det[0])); w.writeheader(); w.writerows(det)
    S = {(r['split'], r['method']): r for r in summ}; D = {d['split']: d for d in det}
    nums = []
    def add(name, value, fmt): nums.append({'name': name, 'value': value, 'fmt': fmt})
    for s in SPLITS:
        for mth in METHODS:
            r = S[(s, mth)]; tag = s.replace('_', '') + mth.replace('_', '')
            add(f'{tag}Worse', r['worse'], 'd'); add(f'{tag}Gain', 100 * r['net_gain'], '.1f')
            add(f'{tag}App', r['applied_runs'], 'd'); add(f'{tag}AppWorse', r['applied_worse'], 'd')
            add(f'{tag}EpApp', 100 * r['episode_applied_rate'], '.0f')
        add(f'{s.replace("_", "")}Detected', D[s]['detected'], 'd'); add(f'{s.replace("_", "")}DetectedAmongLtt', D[s]['detected_among_ltt_applied'], 'd')
    with open(out / 'numbers.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'value', 'fmt']); w.writeheader(); w.writerows(nums)
    L = ['# r3 results: ' + res.name, '', f'runs: {S[("id", "ltt")]["runs"]} (evaluation episodes 9..48)', '',
         '| split | method | worse [CP 95%] | net gain [95% boot] | applied runs | applied & worse | episode applied |', '|---|---|---|---|---|---|---|']
    for r in summ:
        L.append(f"| {r['split']} | {r['method']} | {r['worse']}/{r['runs']} [{100*r['worse_lo']:.1f}, {100*r['worse_hi']:.1f}] | "
                 f"{100*r['net_gain']:.1f}% [{100*r['gain_lo']:.1f}, {100*r['gain_hi']:.1f}] | {r['applied_runs']} | {r['applied_worse']} | {100*r['episode_applied_rate']:.0f}% |")
    L += ['', '| split | shift detected | LTT-applied runs | detected among LTT-applied |', '|---|---|---|---|']
    for d in det:
        L.append(f"| {d['split']} | {d['detected']}/{d['runs']} | {d['ltt_applied_runs']} | {d['detected_among_ltt_applied']} |")
    (out / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
