"""Reproduce both studies of the paper with one command.

    python reproduce.py [--workers N] [--smoke]

Steps (resumable; state in outputs/state.json, logs in outputs/logs/):
  1 preflight  Python and package versions
  2 locks      Study 1 and Study 2 protocol locks (they hash the plans, configs and the code that the studies use)
  3 tests      unit tests of both studies
  4 study1     fresh re-run of Study 1 into impl/results_c1_rerun   (--smoke: 6 smoke cases only)
  5 study2     fresh re-run of Study 2 into impl/results_c7_rerun
  6 compare    every decision and test error of the re-run against the reference results (impl/results_c*_mac)
  7 analyses   post hoc analyses reported in the paper / supplement: bound-definition check, sensitivity,
               applied-only gains (written next to the reference results)
  8 figures    figures, tables and every reported number from the reference results (outputs/reference/) and,
               after a full re-run, from the re-run (outputs/rerun/); the two number files must agree
               (machine-dependent timing excluded)
  9 cost       acceptance cost on this machine (outputs/ict_cost_<os>.json)
 10 report     outputs/REPORT.md
A failed step stops the run and points to its log; nothing here edits locked files or the reference results' raw data.
"""
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent; IMPL = ROOT / 'impl'
OUT = ROOT / 'outputs'; LOGS = OUT / 'logs'; STATE = OUT / 'state.json'
TIMING = ('\\CxGateMs',)


def sh(cmd, name, cwd=IMPL):
    LOGS.mkdir(parents=True, exist_ok=True); t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    (LOGS / f'{name}.log').write_text(f'$ {" ".join(map(str, cmd))}\n[exit {r.returncode}, {time.time() - t0:.1f}s]\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}')
    if r.returncode != 0:
        raise RuntimeError(f'{name} failed (exit {r.returncode}); see {LOGS / (name + ".log")}\n{(r.stdout + r.stderr)[-1500:]}')
    return r


def compare(tag, fresh, ref='_mac'):
    import numpy as np
    fd, rd = IMPL / f'results_{tag}{fresh}' / 'cases', IMPL / f'results_{tag}{ref}' / 'cases'
    names = sorted(x.name for x in fd.iterdir() if (x / 'complete.json').exists())
    diff, maxrel, missing = [], 0.0, []
    for n in names:
        if not (rd / n / 'complete.json').exists():
            missing.append(n); continue
        a = json.loads((fd / n / 'complete.json').read_text()); b = json.loads((rd / n / 'complete.json').read_text())
        for x, y in zip(a['rows'], b['rows']):
            if (x['method'], x['split'], x['applied'], x['worse']) != (y['method'], y['split'], y['applied'], y['worse']):
                diff.append(f"{n}|{x['split']}|{x['method']}")
            maxrel = max(maxrel, abs(x['mse'] - y['mse']) / max(abs(y['mse']), 1e-30))
        if 'gates' in a and any(not np.allclose(a['gates'][g], b['gates'][g], rtol=0, atol=1e-9) for g in a['gates']):
            diff.append(f'{n}|gates')
    return {'cases_compared': len(names), 'missing_in_reference': missing, 'decision_differences': diff, 'max_rel_mse_diff': maxrel}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); py = sys.executable
    OUT.mkdir(exist_ok=True); st = json.loads(STATE.read_text()) if STATE.exists() else {'done': {}}
    suf = '_rerun_smoke' if a.smoke else '_rerun'

    def step(i, name, fn):
        if name in st['done']:
            print(f'[{i}] {name}: done earlier, skipped'); return
        t0 = time.time(); print(f'[{i}] {name} ...', flush=True); res = fn()
        st['done'][name] = {'seconds': round(time.time() - t0, 1), 'result': res}; STATE.write_text(json.dumps(st, indent=2))
        print(f'[{i}] {name}: ok ({time.time() - t0:.0f}s)', flush=True)

    def preflight():
        info = {'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version.split()[0]}
        for m in ('numpy', 'scipy', 'torch', 'mujoco', 'matplotlib'):
            try:
                info[m] = __import__(m).__version__
            except Exception as e:
                raise RuntimeError(f'package {m} missing: pip install -r requirements.txt ({e})')
        return info
    step(1, 'preflight', preflight)
    step(2, 'locks', lambda: {t: sh([py, '-c', f"import sys; sys.path.insert(0,'{t}'); import {t}_study as S; S.check_lock(); print('ok')"], f'lock_{t}').stdout.strip() for t in ('c1', 'c7')})
    step(3, 'tests', lambda: {t: sh([py, '-c', f"import sys; sys.path.insert(0,'{t}'); import test_{t} as m; n=[x for x in dir(m) if x.startswith('test_')]; [getattr(m,x)() for x in n]; print(len(n),'passed')"], f'tests_{t}').stdout.strip() for t in ('c1', 'c7')})
    for i, (t, name) in enumerate((('c1', 'study1'), ('c7', 'study2')), 4):
        extra = ['--smoke'] if a.smoke else []
        step(i, name, lambda t=t, extra=extra: (sh([py, f'{t}/{t}_study.py', '--run', '--suffix', '_rerun', '--workers', str(a.workers)] + extra, f'{t}_run').stdout.strip().splitlines() or [''])[-1])
        step(i, name + '_analyze', lambda t=t, extra=extra: (sh([py, f'{t}/{t}_study.py', '--analyze', '--suffix', '_rerun'] + extra, f'{t}_analyze').stdout.strip().splitlines() or [''])[-1])

    def do_compare():
        res = {t: compare(t, suf) for t in ('c1', 'c7')}
        (OUT / 'rerun_comparison.json').write_text(json.dumps(res, indent=2))
        bad = [t for t in res if res[t]['decision_differences'] or res[t]['missing_in_reference']]
        if bad:
            raise RuntimeError(f'the re-run differs from the reference results in {bad}; see outputs/rerun_comparison.json')
        return {t: {k: v for k, v in res[t].items() if k != 'decision_differences'} for t in res}
    step(6, 'compare', do_compare)

    def analyses():
        out = {}
        for mod in ('c7_ucb_check', 'c7_sensitivity'):
            out[mod] = (sh([py, '-c', f"import sys; sys.path.insert(0,'c7'); import {mod} as m; m.main('_mac')"], mod).stdout.strip().splitlines() or [''])[-1][:200]
        return out
    step(7, 'analyses', analyses)

    def figures():
        ref = OUT / 'reference'; sh([py, 'make_figures_tables.py', '--suffix', '_mac', '--gen', str(ref)], 'figures_reference')
        sh([py, '-c', f"import sys; sys.path.insert(0,'c7'); import c7_horizon_supp as m; m.main('_mac', r'{ref}')"], 'figure_horizon')
        out = {'reference': str(ref.relative_to(ROOT))}
        if not a.smoke:
            rr = OUT / 'rerun'; sh([py, 'make_figures_tables.py', '--suffix', '_rerun', '--gen', str(rr)], 'figures_rerun')
            strip = lambda p: '\n'.join(l for l in p.read_text().splitlines() if not any(m in l for m in TIMING))
            same = strip(rr / 'numbers_c.tex') == strip(ref / 'numbers_c.tex'); out['numbers_identical_from_rerun'] = same
            if not same:
                raise RuntimeError('numbers from the re-run differ from the reference; compare outputs/rerun and outputs/reference')
        return out
    step(8, 'figures', figures)
    step(9, 'cost', lambda: (sh([py, 'c7/c7_ict_cost.py', '--out', str(OUT / f'ict_cost_{platform.system().lower()}.json')], 'ict_cost').stdout.strip().splitlines() or [''])[:3])

    def report():
        d = st['done']; c = json.loads((OUT / 'rerun_comparison.json').read_text())
        lines = ['# Reproduction report', '', f"- environment: {d['preflight']['result']}", f"- mode: {'smoke (6 cases per study)' if a.smoke else 'full re-run (408 cases per study)'}", '',
                 '| step | seconds | result |', '|---|---|---|'] + [f"| {k} | {v['seconds']} | {json.dumps(v['result'])[:150]} |" for k, v in d.items()]
        lines += ['', '## Re-run against the reference results'] + [f"- {t}: {r['cases_compared']} cases, {len(r['decision_differences'])} decision differences, max relative test-error difference {r['max_rel_mse_diff']:.2e}" for t, r in c.items()]
        (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n'); return 'outputs/REPORT.md'
    step(10, 'report', report)
    print('done:', OUT / 'REPORT.md')


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print('STOPPED:', e); sys.exit(1)
