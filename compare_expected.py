"""Compare regenerated results (results_*/) with the reported ones (expected/, Linux run of the paper).

Decision level (must be identical): LTT picks of every run (4-s and 10-s protocols), acceptance and
eta-level exceedance counts of the tolerance-margin protocol, detector decisions of the r3 protocol.
Numeric level (informative): net gains and worse-run counts at the precision reported in the paper.
Training is not bitwise reproducible across processors, so single test errors can differ slightly on
other machines (Apple M5 Max vs Linux: up to 1.2 % in single runs, reported gains within 0.04 points,
all decisions identical). Writes reproduce_report.md; exit code 1 if any decision differs.
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent; EXP = ROOT / 'expected'


def rd(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def main():
    L, bad, num = ['# Reproduction report', ''], [], []
    for res in ('results_confirm', 'results_h10s_confirm'):
        new = ROOT / res / 'analysis' / 'judge.csv'
        if not new.exists():
            L.append(f'- {res}: not run'); continue
        a = {r['key']: r['ltt_pick'] for r in rd(EXP / res / 'judge.csv')}; b = {r['key']: r['ltt_pick'] for r in rd(new)}
        diff = [k for k in a if a[k] != b.get(k)]; bad += [f'{res}: LTT pick {k}' for k in diff]
        L.append(f'- {res}: LTT decisions identical {len(a) - len(diff)}/{len(a)}')
        A = {(r['split'], r['method']): r for r in rd(EXP / res / 'summary.csv')}; B = {(r['split'], r['method']): r for r in rd(ROOT / res / 'analysis' / 'summary.csv')}
        for k, x in A.items():
            y = B.get(k)
            if y and round(100 * float(x['net_gain']), 1) != round(100 * float(y['net_gain']), 1):
                num.append(f'{res} {k} net gain {100 * float(x["net_gain"]):.2f} vs {100 * float(y["net_gain"]):.2f}')
            if y and round(float(x['worse_rate']) * int(x['runs'])) != round(float(y['worse_rate']) * int(y['runs'])):
                num.append(f'{res} {k} worse runs {x["worse_rate"]} vs {y["worse_rate"]}')
    new = ROOT / 'results_h10s_r3' / 'analysis' / 'detector.csv'
    if new.exists():
        a = {r['split']: r['detected'] for r in rd(EXP / 'results_h10s_r3' / 'detector.csv')}; b = {r['split']: r['detected'] for r in rd(new)}
        diff = [s for s in a if a[s] != b.get(s)]; bad += [f'r3 detector {s}' for s in diff]
        L.append(f'- results_h10s_r3: detector decisions identical {len(a) - len(diff)}/{len(a)} splits')
    else:
        L.append('- results_h10s_r3: not run')
    new = ROOT / 'results_h10s_eta' / 'rows.csv'
    if new.exists():
        key = lambda r: (r['key'], r['split'], r['target'], r['eta'])
        a = {key(r): r for r in rd(EXP / 'results_h10s_eta' / 'rows.csv')}; b = {key(r): r for r in rd(new)}
        diff = [k for k in a if k not in b or a[k]['applied'] != b[k]['applied'] or a[k]['ep_exceed_eta'] != b[k]['ep_exceed_eta']]
        bad += [f'eta {k}' for k in diff]
        L.append(f'- results_h10s_eta: acceptance and eta-level exceedances identical {len(a) - len(diff)}/{len(a)} rows')
        ca = json.loads((EXP / 'results_h10s_eta' / 'claims.json').read_text()); cb = json.loads((ROOT / 'results_h10s_eta' / 'analysis' / 'claims.json').read_text())
        L.append(f"- pre-registered claims: E1 {cb['E1_validity']['verdict']} (expected {ca['E1_validity']['verdict']}), E2 {cb['E2_usefulness']['verdict']} (expected {ca['E2_usefulness']['verdict']})")
        if (ca['E1_validity']['verdict'], ca['E2_usefulness']['verdict']) != (cb['E1_validity']['verdict'], cb['E2_usefulness']['verdict']):
            bad.append('eta claim verdicts')
    else:
        L.append('- results_h10s_eta: not run')
    L += ['', f'## Decision level: {"IDENTICAL" if not bad else "DIFFERENT"}'] + [f'- {x}' for x in bad[:50]]
    L += ['', f'## Numeric level at reported precision (informative): {len(num)} differences'] + [f'- {x}' for x in num[:50]]
    (ROOT / 'reproduce_report.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
