"""Round-3 reviewer experiments (v22): comparison with a published-method baseline and a shift detector.

(B-1) Conformal uncertainty gating (idea of uncertainty-aware latent safety filters, ref. Seo et al. 2025):
      an ensemble of M residual MLPs (member 0 is the original world model Z) gives a per-episode
      uncertainty u_i = mean over steps/dims of the across-member variance of the rolled-out trajectory.
      The threshold is the split-conformal (1-alpha) quantile of u on the judge set C_s2.  At test time
      the selected candidate c* is applied to an episode iff u_i <= threshold, otherwise Z is kept.
(B-2) Shift detector on top of LTT: the first `detector_window_steps` steps of every test episode are
      observed; the score is the mean squared one-step acceleration residual of Z on those steps.  The
      per-episode threshold is the split-conformal (1-alpha_d) quantile of the same score on C_s2.  If at
      least `detector_min_exceed` of the first `detector_monitor_episodes` test episodes exceed it, a
      distribution shift is declared and the LTT decision is switched off (Z is kept).
Every method is evaluated on test episodes 9..48 (the first 8 are the monitoring window).

The base pipeline (data, Z, candidates, selection, LTT) is the one of confirm.py with the same seeds and
streams; in --mode posthoc the LTT pick and judge count are asserted equal to results_confirm.

    python confirm_r3.py --mode confirm --out results_confirm_r3          # NEW seeds (confirmatory)
    python confirm_r3.py --mode posthoc --out results_confirm_r3_posthoc  # v14 seeds (supplementary)
"""
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import run
from calibration import select
from confirm import generate, stack, apply, cp_upper, episode_loss, kinematic

ROOT = Path(__file__).resolve().parent
CAND = ('prior', 'original', 'scalar', 'diagonal', 'matrix')
SPLITS = ('id', 'ood_scale', 'ood_damp')
METHODS = ('keep_original', 'ltt', 'ltt_detect', 'conformal_gate')
LOCKED = ('confirm_r3_config.json', 'confirm_r3.py', 'confirm.py', 'confirm_config.json', 'run.py', 'calibration.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def conformal_threshold(scores, alpha):
    """Split-conformal upper threshold: the ceil((n+1)(1-alpha))-th smallest score (inf if that exceeds n)."""
    s = np.sort(np.asarray(scores, dtype='float64')); n = len(s); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float('inf') if k > n else float(s[k - 1])


def ensemble_uncertainty(preds):
    """preds: list of (E, H, 4) rollouts -> per-episode mean across-member variance."""
    p = np.stack(preds, 0).astype('float64')
    return p.var(axis=0).mean(axis=(1, 2))


def one_step_score(model, data, m, dt, dev):
    """Mean squared one-step acceleration residual of `model` on the first m observed steps of each episode."""
    s, u = data
    x = np.concatenate((s[:, :m], u[:, :m]), axis=-1).reshape(-1, 6).astype('float32')
    with torch.no_grad():
        a_hat = model(torch.tensor(x, device=dev)).cpu().numpy().reshape(s.shape[0], m, 2)
    a_true = (s[:, 1:m + 1, 2:].astype('float64') - s[:, :m, 2:]) / dt
    return ((a_hat.astype('float64') - a_true) ** 2).mean(axis=(1, 2))


def shift_detected(scores, tau, monitor, min_exceed):
    x = int((np.asarray(scores[:monitor]) > tau).sum())
    return x >= min_exceed, x


def roll(m, d, system, dt, dev):
    return run.rollout('prior' if m is None else 'residual', m, None, d, system, dt, dev)[1]


def run_case(cfg, r3, system, regime, n, noise, seed, test_offset, dev, out, reference=None):
    key = f'{system}_{regime}_n{n}_noise{noise}_seed{seed}'
    case = out / 'cases' / key
    if (case / 'complete.json').exists():
        return json.loads((case / 'complete.json').read_text())
    case.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter(); dt, steps = cfg['dt'], cfg['steps']
    rng = np.random.default_rng(seed + 90000)
    sizes = {'train': n, 'validation': cfg['validation_episodes'], 'calibration': cfg['calibration_episodes'], 'judge': cfg['judge_episodes']}
    data = {k: generate(seed + 1000 * i, sizes[k], steps, system, dt, 1.0, regime) for i, k in enumerate(sizes, 1)}
    x, y = run.targets(data['train'], system, noise, rng, regime)
    xv, yv = run.targets(data['validation'], system, noise, rng, regime)
    xc, yc = run.targets(data['calibration'], system, noise, rng, regime)
    run.targets(data['judge'], system, noise, rng, regime)  # keep the rng stream identical to confirm.py
    z_model, _ = run.fit(x, y, xv, yv, system, 'residual', 0, cfg, seed, dev)
    # ---- base pipeline (identical to confirm.py)
    pc, zc = roll(None, data['calibration'], system, dt, dev), roll(z_model, data['calibration'], system, dt, dev)
    pv, zv = roll(None, data['validation'], system, dt, dev), roll(z_model, data['validation'], system, dt, dev)
    pj, zj = roll(None, data['judge'], system, dt, dev), roll(z_model, data['judge'], system, dt, dev)
    s0 = data['calibration'][0][:, 0]; aobs = yc.reshape(-1, steps, 2)
    vv = s0[:, None, 2:] + dt * np.cumsum(aobs, axis=1); qq = s0[:, None, :2] + dt * np.cumsum(vv, axis=1)
    obs = np.concatenate([qq, vv], axis=-1)
    r, e = stack(zc - pc), stack(obs - pc)
    rv, ev = stack(zv - pv), stack(data['validation'][0][:, 1:] - pv)
    mats = {'prior': np.zeros((2, 2)), 'original': np.eye(2)}
    for kind in ('scalar', 'diagonal', 'matrix'):
        mats[kind], _ = select(r, e, rv, ev, kind, cfg['lambdas'])
    sel_loss = {c: float(np.mean((rv @ a - ev) ** 2)) for c, a in mats.items()}
    best = min(('prior', 'scalar', 'diagonal', 'matrix'), key=lambda c: (sel_loss[c], CAND.index(c)))
    truth_j = data['judge'][0][:, 1:]
    dj = episode_loss(apply(pj, zj, mats[best]), truth_j) - episode_loss(zj, truth_j)
    k_worse = int((dj > 0).sum()); ucb = cp_upper(k_worse, len(dj), cfg['ltt_epsilon'])
    ltt_pick = best if ucb <= cfg['ltt_delta'] else 'original'
    if reference is not None:
        assert ltt_pick == reference['ltt_pick'] and k_worse == reference['judge_worse'], (key, ltt_pick, reference['ltt_pick'], k_worse, reference['judge_worse'])
    # ---- (B-1) ensemble + conformal threshold on the judge set
    te = time.perf_counter()
    members = [z_model] + [run.fit(x, y, xv, yv, system, 'residual', 0, cfg, seed + r3['ensemble_seed_stride'] * j, dev)[0]
                           for j in range(1, r3['ensemble_members'])]
    ens_seconds = time.perf_counter() - te
    u_judge = ensemble_uncertainty([zj] + [roll(mm, data['judge'], system, dt, dev) for mm in members[1:]])
    tau_c = conformal_threshold(u_judge, r3['conformal_alpha'])
    # ---- (B-2) detector threshold on the judge set
    m = r3['detector_window_steps']
    tau_d = conformal_threshold(one_step_score(z_model, data['judge'], m, dt, dev), r3['detector_alpha'])
    mon, mex = r3['detector_monitor_episodes'], r3['detector_min_exceed']
    # ---- new test data
    rows, info = [], {}
    for i, split in enumerate(SPLITS, 6):
        scale, damp = {'id': (1.0, 0.0), 'ood_scale': (cfg['ood_scale'], 0.0), 'ood_damp': (1.0, cfg['ood_extra_damping'])}[split]
        td = generate(seed + 1000 * i + test_offset, cfg['test_episodes'], steps, system, dt, scale, regime, damp)
        truth = td[0][:, 1:]; p, z = roll(None, td, system, dt, dev), roll(z_model, td, system, dt, dev)
        cand = {c: apply(p, z, a) for c, a in mats.items()}; zz = cand['original']
        shift, n_exceed = shift_detected(one_step_score(z_model, td, m, dt, dev), tau_d, mon, mex)
        u_test = ensemble_uncertainty([z] + [roll(mm, td, system, dt, dev) for mm in members[1:]])
        gate = u_test <= tau_c
        conf_pred = np.where(gate[:, None, None], cand[best], zz)
        ltt_on = ltt_pick != 'original'
        preds = {'keep_original': (zz, False), 'ltt': (cand[ltt_pick], ltt_on),
                 'ltt_detect': (cand[ltt_pick] if (ltt_on and not shift) else zz, ltt_on and not shift),
                 'conformal_gate': (conf_pred, bool(gate[mon:].any()))}
        ev_idx = slice(mon, None)  # evaluation episodes 9..48
        tr = truth[ev_idx]; lz = episode_loss(zz[ev_idx], tr)
        for mname, (pred, applied) in preds.items():
            pe = pred[ev_idx]; le = episode_loss(pe, tr)
            rows.append({'key': key, 'system': system, 'regime': regime, 'n': n, 'noise': noise, 'seed': seed, 'split': split,
                         'method': mname, 'applied': int(applied), 'mse': float(((pe.astype('float64') - tr) ** 2).mean()),
                         'mse_original': float(((zz[ev_idx].astype('float64') - tr) ** 2).mean()),
                         'episode_worse_rate': float((le > lz).mean()),
                         'episode_applied_rate': float(gate[ev_idx].mean()) if mname == 'conformal_gate' else (1.0 if applied else 0.0),
                         'kinematic_max': kinematic(pe, td[0][ev_idx, 0], dt)})
        info[split] = {'shift_detected': bool(shift), 'n_exceed': n_exceed, 'conformal_applied_rate': float(gate[ev_idx].mean())}
    rep = {'key': key, 'system': system, 'regime': regime, 'n': n, 'noise': noise, 'seed': seed, 'rows': rows,
           'best_candidate': best, 'judge_worse': k_worse, 'cp_upper': ucb, 'ltt_pick': ltt_pick,
           'conformal_threshold': tau_c, 'detector_threshold': tau_d, 'splits': info,
           'seconds': {'ensemble_training': ens_seconds, 'case_total': time.perf_counter() - t0}}
    run.atomic_json(case / 'complete.json', rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('confirm', 'posthoc', 'dry_run'), required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--device', default='cpu')
    ap.add_argument('--reference', default='results_confirm', help='posthoc only: v14 results to check the LTT picks against')
    a = ap.parse_args()
    cfg = json.loads((ROOT / 'confirm_config.json').read_text()); r3 = json.loads((ROOT / 'confirm_r3_config.json').read_text())
    lock = ROOT / 'confirm_r3_protocol_lock.json'
    if a.mode != 'dry_run':
        if not lock.exists():
            raise SystemExit('confirm_r3_protocol_lock.json missing: lock the r3 protocol before running')
        L = json.loads(lock.read_text())
        bad = [f for f in LOCKED if L['sha256'].get(f) != sha(ROOT / f)]
        if bad:
            raise SystemExit(f'Protocol changed after locking ({", ".join(bad)}); use a new protocol/version')
    else:
        L = None
    spec = r3['dry_run'] if a.mode == 'dry_run' else r3['modes'][a.mode]
    out = (ROOT / a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['cpu_threads']); dev = run.choose_device(a.device)
    import sysinfo
    if not (out / 'system_env.json').exists():
        (out / 'system_env.json').write_text(json.dumps(sysinfo.collect(str(dev)), indent=2, ensure_ascii=False))
    refs = {}
    if a.mode == 'posthoc':
        refdir = (ROOT / a.reference / 'cases').resolve()
        refs = {d.name: json.loads((d / 'complete.json').read_text()) for d in refdir.iterdir()}
    systems = cfg['systems'] if a.mode != 'dry_run' else cfg['systems'][:1]
    regimes = cfg['regimes'] if a.mode != 'dry_run' else cfg['regimes'][:1]
    trains = cfg['train_episodes'] if a.mode != 'dry_run' else cfg['train_episodes'][:1]
    noises = cfg['noise'] if a.mode != 'dry_run' else cfg['noise'][:1]
    reps = []; start = time.perf_counter()
    for system in systems:
        for regime in regimes:
            for n in trains:
                for noise in noises:
                    for seed in spec['seeds']:
                        key = f'{system}_{regime}_n{n}_noise{noise}_seed{seed}'
                        rep = run_case(cfg, r3, system, regime, n, noise, seed, spec['test_seed_offset'], dev, out, refs.get(key))
                        reps.append(rep)
                        print(key, 'ltt', rep['ltt_pick'], 'shift', [rep['splits'][s]['shift_detected'] for s in SPLITS], flush=True)
    rows = [row for rp in reps for row in rp['rows']]
    with open(out / 'rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    run.atomic_json(out / 'run_summary.json', {'mode': a.mode, 'cases': len(reps), 'seconds_this_invocation': time.perf_counter() - start,
                                               'protocol_lock': L, 'device': str(dev), 'spec': spec})
    print('TOTAL', len(reps), round(time.perf_counter() - start, 1))


if __name__ == '__main__':
    main()
