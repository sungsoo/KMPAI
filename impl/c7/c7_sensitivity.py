"""A37 (round 16, R2-C2 / R2-C3): post hoc sensitivity of the magnitude rule and gains of applied runs.
Uses only stored Study 2 judge ratios and test rows; changes no locked decision. Whole-matrix rule (the stored test
rows cover the whole candidate through the always-applied forecast)."""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; IMPL = HERE.parent
sys.path.insert(0, str(HERE))
from c7_rules import magnitude_gate
SYS = ('planar_2link', 'planar_3link', 'franka_7dof')


def main(suffix='_mac'):
    d = IMPL / f'results_c7{suffix}'
    reps = {json.loads(f.read_text())['key']: json.loads(f.read_text()) for f in (d / 'cases').glob('*/complete.json')}
    rows = list(csv.DictReader(open(d / 'rows.csv')))
    ao = {r['key']: r for r in rows if r['split'] == 'id' and r['method'] == 'always_on'}
    out = []
    for tau in (0.0, 0.02, 0.05):
        for n in (36, 60, 100):
            for clip in (1.5, 2.0, 3.0):
                rec = {'tau': tau, 'n': n, 'clip': clip}
                for s in SYS:
                    keys = [k for k, r in reps.items() if r['system'] == s]
                    acc = [k for k in keys if magnitude_gate(np.array(reps[k]['judge_ratio_whole'][:n]), tau, 0.1, 1.0, 0.05, clip)[0]]
                    rec[f'{s}_applied'] = len(acc); rec[f'{s}_worse_avg'] = sum(float(ao[k]['test_mean_ratio']) > 1 for k in acc)
                    mse = sum(float(ao[k]['mse']) if k in acc else float(ao[k]['mse_Z']) for k in keys); mz = sum(float(ao[k]['mse_Z']) for k in keys)
                    rec[f'{s}_net_gain'] = 1 - mse / mz
                out.append(rec)
    with open(d / 'analysis' / 'sensitivity.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    gates = ('strict_ltt_c1', 'mag_whole_gate', 'mag_dir_gate', 'mean_loss_rule', 'always_on')
    med = {g: {s: (lambda v: float(np.median(v)) if v else float('nan'))([float(r['rel_gain']) for r in rows if r['system'] == s and r['split'] == 'id' and r['method'] == g and r['applied'] == '1']) for s in SYS} for g in gates}
    (d / 'analysis' / 'applied_only_gain.json').write_text(json.dumps(med, indent=2))
    tot_worse = sum(r[f'{s}_worse_avg'] for r in out for s in SYS)
    print('settings', len(out), 'total applied-and-worse-on-average across all settings and arms:', tot_worse)
    base = [r for r in out if r['tau'] == 0 and r['n'] == 100 and r['clip'] == 2.0][0]
    print('baseline', {k: v for k, v in base.items() if 'applied' in k})


if __name__ == '__main__':
    main()
