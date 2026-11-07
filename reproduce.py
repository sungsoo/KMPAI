"""Re-run every pre-registered protocol of the paper and compare with the reported results.

    python reproduce.py            # all steps (about 25 min on one CPU core; resumable: rerun the same command)
    python reproduce.py --step eta # one step: tests | confirm4s | confirm10s | r2 | r3 | eta | compare

Steps (each refuses to run if a locked file was changed, see *_protocol_lock.json):
  tests       unit tests of the locked code
  confirm4s   earlier 4-s protocol (seeds 101-229)          -> results_confirm/
  confirm10s  main 10-s protocol (seeds 503-661)            -> results_h10s_confirm/
  r2          round-2 analyses of the 10-s protocol         -> results_h10s_r2/
  r3          detector / conformal protocol (seeds 701-857) -> results_h10s_r3/
  eta         tolerance-margin protocol (seeds 1301-1423)   -> results_h10s_eta/
  compare     compare with expected/ -> reproduce_report.md (exit 1 if a decision differs)
"""
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
STEPS = {
    'tests': [[PY, '-m', 'unittest', 'test_confirm', 'test_confirm_r3']],
    'confirm4s': [[PY, 'confirm.py', '--out', 'results_confirm'], [PY, 'confirm_analysis.py', '--results', 'results_confirm']],
    'confirm10s': [[PY, 'horizon_study.py', '--stage', 'confirm'], [PY, 'confirm_analysis.py', '--results', 'results_h10s_confirm']],
    'r2': [[PY, 'horizon_study.py', '--stage', 'r2'], [PY, 'confirm_r2_analysis.py', '--results', 'results_h10s_confirm', '--r2', 'results_h10s_r2']],
    'r3': [[PY, 'horizon_study.py', '--stage', 'r3'], [PY, 'confirm_r3_analysis.py', '--results', 'results_h10s_r3']],
    'eta': [[PY, 'eta_study.py', '--run'], [PY, 'eta_study.py', '--analyze']],
    'compare': [[PY, 'compare_expected.py']],
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--step', choices=list(STEPS)); a = ap.parse_args()
    for name in ([a.step] if a.step else list(STEPS)):
        t = time.perf_counter(); print(f'== {name}', flush=True)
        for cmd in STEPS[name]:
            r = subprocess.run(cmd, cwd=ROOT)
            if r.returncode:
                sys.exit(f'step {name} failed: {" ".join(map(str, cmd))} (exit {r.returncode})')
        print(f'== {name} done in {time.perf_counter() - t:.0f} s', flush=True)


if __name__ == '__main__':
    main()
