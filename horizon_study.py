"""Long-horizon study (v16): run the locked v14/r2/r3 code with a user-set prediction horizon, pre-registered.

    python horizon_study.py --lock                 # 1) fix the protocol (refuses if results already exist)
    python horizon_study.py --stage confirm        # 2) v14 confirmatory protocol      -> results_h{H}s_confirm/
    python horizon_study.py --stage r2             #    round-2 analyses (on 2)         -> results_h{H}s_r2/
    python horizon_study.py --stage r3             #    round-3 experiments, new seeds  -> results_h{H}s_r3/
    python horizon_study.py --stage all

User setting: `horizon_seconds` in horizon_config.json (steps = horizon_seconds / dt). Changing it means a NEW
protocol: edit the config (and the seeds), delete nothing, and lock again; results of an earlier lock stay where
they are (the folder name contains the horizon). The locked files confirm.py, confirm_config.json, confirm_r2.py,
confirm_r3.py, confirm_r3_config.json, run.py and calibration.py are imported and called, never modified;
their hashes are part of this study's lock.
"""
import argparse, csv, hashlib, json, time
from pathlib import Path
import torch

import run, sysinfo
import confirm, confirm_r2, confirm_r3

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'horizon_config.json'; LOCK = ROOT / 'horizon_protocol_lock.json'
LOCKED = ('horizon_config.json', 'horizon_study.py', 'confirm.py', 'confirm_config.json', 'confirm_r2.py',
          'confirm_r3.py', 'confirm_r3_config.json', 'run.py', 'calibration.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def configs():
    h = json.loads(CONFIG.read_text())
    base = json.loads((ROOT / h['base_config']).read_text()); r3 = json.loads((ROOT / h['base_r3_config']).read_text())
    steps = int(round(h['horizon_seconds'] / base['dt']))
    cfg = dict(base, steps=steps, seeds=h['confirm']['seeds'], test_seed_offset=h['confirm']['test_seed_offset'],
               purpose=h['purpose'])
    r3 = dict(r3, modes={'confirm': {'seeds': h['r3']['seeds'], 'test_seed_offset': h['r3']['test_seed_offset'],
                                      'note': 'long-horizon study: new seeds, new test data'}})
    tag = f"h{h['horizon_seconds']:g}s"
    return h, cfg, r3, tag


def out_dir(tag, stage):
    return ROOT / f'results_{tag}_{stage}'


def check_lock():
    if not LOCK.exists():
        raise SystemExit('horizon_protocol_lock.json missing: run  python horizon_study.py --lock  first')
    L = json.loads(LOCK.read_text())
    bad = [f for f in LOCKED if L['sha256'].get(f) != sha(ROOT / f)]
    if bad:
        raise SystemExit(f'Protocol changed after locking ({", ".join(bad)}): use a new horizon/seeds and lock again')
    return L


def do_lock():
    h, cfg, r3, tag = configs()
    existing = [str(out_dir(tag, s)) for s in h['stages'] if out_dir(tag, s).exists()]
    if existing:
        raise SystemExit(f'results for {tag} already exist ({existing}); a lock must precede the data')
    L = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'horizon_seconds': h['horizon_seconds'],
         'steps': cfg['steps'], 'confirm_seeds': cfg['seeds'], 'r3_seeds': r3['modes']['confirm']['seeds'],
         'sha256': {f: sha(ROOT / f) for f in LOCKED}}
    LOCK.write_text(json.dumps(L, indent=2)); print('locked', tag, 'steps', cfg['steps'])


def env(out, dev):
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'system_env.json').exists():
        (out / 'system_env.json').write_text(json.dumps(sysinfo.collect(str(dev)), indent=2, ensure_ascii=False))


def stage_confirm(cfg, tag, L, dev):
    out = out_dir(tag, 'confirm'); env(out, dev); reps = []; t0 = time.perf_counter()
    for system in cfg['systems']:
        for regime in cfg['regimes']:
            for n in cfg['train_episodes']:
                for noise in cfg['noise']:
                    for seed in cfg['seeds']:
                        r = confirm.run_case(cfg, system, regime, n, noise, seed, dev, out); reps.append(r)
                        print(r['key'], 'ltt', r['ltt_pick'], 'k', r['judge_worse'], flush=True)
    rows = [row for r in reps for row in r['rows']]
    with open(out / 'cases.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    run.atomic_json(out / 'run_summary.json', {'cases': len(reps), 'seconds_this_invocation': time.perf_counter() - t0,
                                               'protocol_lock': L, 'device': str(dev), 'steps': cfg['steps']})


def stage_r2(cfg, tag, L, dev):
    res = out_dir(tag, 'confirm'); out = out_dir(tag, 'r2'); env(out, dev)
    (out / 'protocol.json').write_text(json.dumps({'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': 'Round-2 analyses of the long-horizon confirmatory run (post-hoc items stay post-hoc).', 'horizon_lock': L}, indent=2))
    rows, info = [], []
    for d in sorted((res / 'cases').iterdir()):
        rep = json.loads((d / 'complete.json').read_text()); rw, inf = confirm_r2.case(cfg, rep, dev); rows += rw; info.append(inf)
        print(rep['key'], 'r2 ok', flush=True)
    for name, data in (('rows.csv', rows), ('cases.csv', info)):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def stage_r3(cfg, r3, tag, L, dev):
    out = out_dir(tag, 'r3'); env(out, dev); spec = r3['modes']['confirm']; reps = []; t0 = time.perf_counter()
    for system in cfg['systems']:
        for regime in cfg['regimes']:
            for n in cfg['train_episodes']:
                for noise in cfg['noise']:
                    for seed in spec['seeds']:
                        rep = confirm_r3.run_case(cfg, r3, system, regime, n, noise, seed, spec['test_seed_offset'], dev, out, None)
                        reps.append(rep); print(rep['key'], 'ltt', rep['ltt_pick'], flush=True)
    rows = [row for rp in reps for row in rp['rows']]
    with open(out / 'rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    run.atomic_json(out / 'run_summary.json', {'mode': 'confirm', 'cases': len(reps), 'seconds_this_invocation': time.perf_counter() - t0,
                                               'protocol_lock': L, 'device': str(dev), 'spec': spec, 'steps': cfg['steps']})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lock', action='store_true')
    ap.add_argument('--stage', choices=('confirm', 'r2', 'r3', 'all')); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    if a.lock:
        return do_lock()
    h, cfg, r3, tag = configs(); L = check_lock(); torch.set_num_threads(cfg['cpu_threads']); dev = run.choose_device(a.device)
    for s in (h['stages'] if a.stage == 'all' else [a.stage]):
        print('== stage', s, tag, flush=True)
        {'confirm': lambda: stage_confirm(cfg, tag, L, dev), 'r2': lambda: stage_r2(cfg, tag, L, dev),
         'r3': lambda: stage_r3(cfg, r3, tag, L, dev)}[s]()


if __name__ == '__main__':
    main()
