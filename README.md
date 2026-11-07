# Reproducible code for "Physical consistency is not accuracy: magnitude-based acceptance of post-hoc world-model corrections"

Sungsoo Kim, Electronics and Telecommunications Research Institute (ETRI) — sungsoo@etri.re.kr

This repository holds everything needed to re-run the two pre-registered studies of the paper and to regenerate every
figure, table and number it reports. It contains no manuscript sources.

## What the studies do
A learned state-space world model (a residual MLP on top of nominal dynamics) forecasts 1 s of joint motion for three
MuJoCo arms (planar 2-link, planar 3-link and the 7-joint Franka Emika Panda from MuJoCo Menagerie). A post-hoc correction
contracts the learned forecast toward a physics prior, which preserves every linear time-axis constraint, and an
acceptance rule decides on independent judge episodes whether to use it.

| Study | Acceptance rule | Protocol lock | Reference results |
|---|---|---|---|
| Study 1 | count rule (exact test on the number of worse judge episodes, 36 episodes) | `impl/c1/c1_protocol_lock.json` | `impl/results_c1_mac/` |
| Study 2 | magnitude rule (betting confidence bound on the clipped mean loss ratio, 100 episodes), direction-wise variant | `impl/c7/c7_protocol_lock.json` | `impl/results_c7_mac/` |

Each study has 136 runs per arm (34 seeds x 2 training sizes x 2 label-noise levels), 408 runs in total.
Claims, thresholds and seeds were fixed before any data were generated (`design/C1/plan.toml`, `design/C7/plan.toml`);
the locks store SHA-256 hashes of the plans, configurations and code, and every run refuses to start if any of them changed.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py --smoke        # about 5 min: locks, tests, 6 cases per study, figures from the reference results
python reproduce.py --workers 12   # full re-run of both studies (about 10 min on an Apple M5 Max with 12 workers)
```
`reproduce.py` is resumable; results go to `outputs/` (report in `outputs/REPORT.md`).

## What `reproduce.py` checks
1. both protocol locks and the unit tests;
2. a fresh re-run of Study 1 and Study 2 (`impl/results_c*_rerun/`);
3. every decision and test error of the re-run against the reference results — on the author's machine the re-run gave
   0 decision differences and 0 test-error difference (`evidence/mac_rerun/`), and a Linux re-run of the smoke cases gave
   identical decisions (`impl/results_c*_smoke/`);
4. the post hoc analyses of the paper and supplement (bound-definition check, sensitivity of the magnitude rule,
   gains of applied runs);
5. all figures, tables and reported numbers (`outputs/reference/`; after a full re-run also `outputs/rerun/`, which must
   give the same numbers apart from machine-dependent timing);
6. the cost of one acceptance decision on your machine (`outputs/ict_cost_<os>.json`).

## Layout
```
reproduce.py                 one-command reproduction
requirements.txt
design/C1, design/C7         pre-registered plans (plan.toml) with their locks, run matrices and claim maps
impl/c1/                     Study 1: arms, learned model and correction (c1_models.py), study runner (c1_study.py),
                             configuration, protocol lock, Franka model (assets/), unit tests
impl/c7/                     Study 2: acceptance rules (c7_rules.py), study runner (c7_study.py), configuration, lock,
                             unit tests, post hoc analyses (c7_ucb_check.py, c7_sensitivity.py, c7_horizon_supp.py)
                             and the cost measurement (c7_ict_cost.py)
impl/calibration.py          fitting of the correction matrix (shared)
impl/confirm.py              exact count bound used by Study 1 (cp_upper); kept unchanged because the Study 1 lock hashes it
impl/run.py                  helper module imported by confirm.py (unchanged)
impl/sysinfo.py              records CPU / memory / OS / package versions of each run
impl/make_figures_tables.py  figures, tables and every reported number from a results folder
impl/figure_style.py, figure_ictx.py, fonts/   figure style (Gentium Book Basic, SIL OFL; NanumMyeongjo fallback, SIL OFL)
impl/results_c1_mac, results_c7_mac            reference results (Apple M5 Max): per-case files, runs.csv, claims.json
impl/results_c1_smoke, results_c7_smoke        Linux re-run of the smoke cases (cross-platform check)
reports/ict_cost_*.json      acceptance cost measured on Linux (1 core) and on the Apple M5 Max
evidence/mac_rerun/          report of the author's full re-run against the reference results (that run also built the
                             manuscript; only the study, comparison and figure steps concern this repository)
```

## Pre-specified outcomes (reference results)
- Study 1 (plan lock `c209be719897`): always applying the correction harmed 81.6% and 71.3% of planar-arm runs at a wider operating range; the count rule accepted 6 of 408 runs.
- Study 2 (plan lock `39289bba1c10`): the magnitude rule applied a correction in 120 of 136 Franka runs with a median gain of 25.9%, and no applied run on any arm was worse on average over its test episodes.

## Environment of the reference runs
macOS on an Apple M5 Max (arm64), Python 3.12, PyTorch 2.14.1, MuJoCo 4.0.0, NumPy 2.4.4, SciPy 1.17.1.

## License
To be added before publication.

## Citation
S. Kim, Physical consistency is not accuracy: magnitude-based acceptance of post-hoc world-model corrections (submitted).
