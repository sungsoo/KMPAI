"""Paper assets for the C1 + C7 manuscript (manuscript-v1, A15): macros, tables and the gate figure.

    python make_paper_assets_c1c7.py [--paper ../paper] [--source mac]

Every number in the rewritten Sections 4-6 comes from results_c1{_mac}/ and results_c7{_mac}/ through this script;
the verdicts are read from the locked analyses (analysis/claims.json) and re-checked against the case files.
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SYS = ('planar_2link', 'planar_3link', 'franka_7dof')
SYSN = {'planar_2link': 'Planar 2-link', 'planar_3link': 'Planar 3-link', 'franka_7dof': 'Franka 7-DOF'}
TAG = {'planar_2link': 'Two', 'planar_3link': 'Three', 'franka_7dof': 'Fr'}
GATES = [('strict_ltt_c1', 'Count rule (C1)'), ('mag_whole_gate', 'Magnitude, whole'), ('mag_dir_gate', 'Magnitude + directions'),
         ('mag_dir_n36', '\\quad same, $n=36$'), ('mag_dir_tail', '\\quad same + tail'), ('mean_loss_rule', 'Mean-loss rule'),
         ('always_on', 'Always applied'), ('equal_budget_retrain', 'Retraining')]


def rd(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def pct(x, nd=1):
    s = f'{100 * x:.{nd}f}'
    return s.replace('-', '\\textminus{}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--paper', default='../paper'); ap.add_argument('--source', default='mac')
    ap.add_argument('--suffix', default=None, help='results suffix (overrides --source), e.g. _rerun'); ap.add_argument('--gen', default=None, help='output directory (default: <paper>/generated)')
    a = ap.parse_args(); suf = a.suffix if a.suffix is not None else ('_mac' if a.source == 'mac' else '')
    gen = Path(a.gen).resolve() if a.gen else (HERE / a.paper / 'generated').resolve(); gen.mkdir(parents=True, exist_ok=True)
    c1d, c7d = HERE / f'results_c1{suf}', HERE / f'results_c7{suf}'
    C1 = json.loads((c1d / 'analysis' / 'claims.json').read_text()); C7 = json.loads((c7d / 'analysis' / 'claims.json').read_text())
    R1, R7 = rd(c1d / 'rows.csv'), rd(c7d / 'rows.csv')
    rep1 = [json.loads(open(f).read()) for f in sorted(glob.glob(str(c1d / 'cases' / '*' / 'complete.json')))]
    v, checks = {}, {}
    sel1 = lambda s, sp, m: [r for r in R1 if r['system'] == s and r['split'] == sp and r['method'] == m]
    sel7 = lambda s, sp, m: [r for r in R7 if r['system'] == s and r['split'] == sp and r['method'] == m]
    runs = len(sel1('franka_7dof', 'id', 'keep_Z')); v['CxRuns'] = runs; v['CxCases'] = C1['cases']; v['CxSeeds'] = runs // 4
    # ---- C1
    acc_total = 0
    for s in SYS:
        t = TAG[s]; m1 = C1['M1'][s]; m2 = C1['M2'][s]
        v[f'CxOne{t}Worse'] = m1['worse']; v[f'CxOne{t}Share'] = pct(m1['share']); v[f'CxOne{t}Lo'] = pct(m1['lower95']); v[f'CxOne{t}Up'] = pct(m1['upper95'])
        v[f'CxOne{t}Verdict'] = m1['verdict'].replace('_', ' '); v[f'CxOne{t}Acc'] = m2['accepted']; acc_total += m2['accepted']
        ks = [x['judge']['k_full'] for x in rep1 if x['system'] == s]; v[f'CxOne{t}JudgeK'] = int(np.median(ks))
        rr = sel1(s, 'range', 'fixed_matrix_no_gate'); v[f'CxOne{t}MaxHarm'] = f"{100 * max(-float(r['rel_gain']) for r in rr):,.0f}".replace(',', '{,}')
        checks[f'C1 M1 recount {s}'] = sum(int(r['worse']) for r in rr) == m1['worse']
    v['CxOneAccTotal'] = acc_total
    fz = sel1('franka_7dof', 'id', 'fixed_matrix_no_gate')
    v['CxOneFrAlwaysNet'] = pct(1 - np.mean([float(r['mse']) for r in fz]) / np.mean([float(r['mse_Z']) for r in fz]))
    v['CxOneFrAlwaysIdWorse'] = sum(int(r['worse']) for r in fz)
    fr = sel1('franka_7dof', 'range', 'fixed_matrix_no_gate'); v['CxOneFrRangeMaxHarm'] = pct(max(-float(r['rel_gain']) for r in fr))
    v['CxOneMThreeVerdict'] = C1['M3']['verdict'].replace('_', ' '); v['CxOneMFourVerdict'] = C1['M4']['verdict'].replace('_', ' ')
    v['CxOneKin'] = f"{C1['kinematic_max_correction']:.1e}".replace('e-', '\\times10^{-') + '}'
    # ---- C7
    n1 = C7['N1']; v['CxSevNOneMed'] = pct(n1['median']); v['CxSevNOneLo'] = pct(n1['boot95'][0]); v['CxSevNOneHi'] = pct(n1['boot95'][1])
    v['CxSevNOneApplied'] = n1['applied']; v['CxSevNOneVerdict'] = n1['verdict'].replace('_', ' ')
    for s in SYS:
        t = TAG[s]; n2 = C7['N2'][s]
        v[f'CxSev{t}Applied'] = n2['applied']; v[f'CxSev{t}MRWorse'] = n2['mean_ratio_worse']; v[f'CxSev{t}Up'] = pct(n2['upper95']); v[f'CxSev{t}Verdict'] = n2['verdict'].replace('_', ' ')
        ap_ = [r for r in sel7(s, 'id', 'mag_dir_gate') if r['applied'] == '1']
        checks[f'C7 N2 recount {s}'] = sum(float(r['test_mean_ratio']) > 1 for r in ap_) == n2['mean_ratio_worse']
        n5 = C7['N5_descriptive'][s]['mag_dir_gate']; v[f'CxSev{t}RangeWorse'] = n5['worse']; v[f'CxSev{t}RangeApplied'] = n5['applied']
        v[f'CxSev{t}RangeMaxHarm'] = f"{100 * n5['max_harm']:,.0f}".replace(',', '{,}') if n5['max_harm'] >= 1 else pct(n5['max_harm'])
        ml = [r for r in sel7(s, 'id', 'mean_loss_rule') if r['applied'] == '1']; v[f'CxSev{t}MeanRuleMRWorse'] = sum(float(r['test_mean_ratio']) > 1 for r in ml)
    n3 = C7['N3']; v['CxSevNThreeDir'] = n3['applied_dir']; v['CxSevNThreeWhole'] = n3['applied_whole']; v['CxSevNThreeB'], v['CxSevNThreeC'] = n3['discordant_b_c']
    v['CxSevNThreeVerdict'] = n3['verdict'].replace('_', ' ')
    n4f = C7['N4_descriptive']['franka_7dof']
    v['CxSevFrNet'] = pct(n4f['mag_dir_gate']['net_gain_id']); v['CxSevFrAlwaysNet'] = pct(n4f['always_on']['net_gain_id'])
    v['CxSevFrNSmall'] = n4f['mag_dir_n36']['applied']; v['CxSevFrTail'] = n4f['mag_dir_tail']['applied']; v['CxSevFrStrict'] = n4f['strict_ltt_c1']['applied']
    p3 = [r for r in sel7('planar_3link', 'id', 'mag_dir_gate') if r['applied'] == '1']
    v['CxSevThreeRunWorse'] = sum(int(r['worse']) for r in p3); v['CxSevThreeRunMaxHarm'] = pct(max([-float(r['rel_gain']) for r in p3] + [0.0]))
    v['CxSevKin'] = f"{C7['kinematic_max_gates']:.1e}".replace('e-', '\\times10^{-') + '}'
    # cross-platform (Linux smoke vs Mac, same cases)
    xs, xd = True, 0.0
    for tag in ('c1', 'c7'):
        for f in glob.glob(str(HERE / f'results_{tag}_smoke' / 'cases' / '*' / 'complete.json')):
            A_ = json.loads(open(f).read()); B_ = json.loads(open(f.replace(f'results_{tag}_smoke', f'results_{tag}{suf}')).read())
            xs &= all(x['applied'] == y['applied'] for x, y in zip(A_['rows'], B_['rows']))
            xd = max(xd, max(abs(x['mse'] - y['mse']) / max(y['mse'], 1e-30) for x, y in zip(A_['rows'], B_['rows'])))
    v['CxXpCases'] = len(glob.glob(str(HERE / 'results_c1_smoke' / 'cases' / '*'))) + len(glob.glob(str(HERE / 'results_c7_smoke' / 'cases' / '*')))
    v['CxXpMaxRel'] = f'{xd:.0e}'.replace('e-', '\\times10^{-') + '}'
    checks['cross-platform decisions identical'] = xs
    checks['C7 N1 verdict consistent'] = (n1['boot95'][0] >= 0.05) == (n1['verdict'] == 'supported')
    # ---- tables (A16 / A17: plain-language claims table, compact gate table, full table for the supplement)
    MAIN = [('always_on', 'Always applied'), ('mean_loss_rule', 'Mean-loss rule'), ('strict_ltt_c1', 'Count rule'),
            ('mag_whole_gate', 'Magnitude rule'), ('mag_dir_gate', 'Magnitude + directions')]
    BRNG = np.random.default_rng(20261107)
    def boot_net(rows_, B=2000):
        seeds = sorted({r['seed'] for r in rows_}); by = {sd: [r for r in rows_ if r['seed'] == sd] for sd in seeds}
        vals = []
        for _ in range(B):
            pick = [r for sd in BRNG.choice(seeds, len(seeds)) for r in by[sd]]
            vals.append(1 - np.mean([float(r['mse']) for r in pick]) / np.mean([float(r['mse_Z']) for r in pick]))
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    def gate_row(g, s, with_range=False):
        idr = sel7(s, 'id', g); rr = sel7(s, 'range', g)
        ap_ = [r for r in idr if r['applied'] == '1']; apr = [r for r in rr if r['applied'] == '1']
        net = 1 - np.mean([float(r['mse']) for r in idr]) / np.mean([float(r['mse_Z']) for r in idr])
        lo_, hi_ = boot_net(idr) if not with_range else (None, None)
        gtxt = pct(net, 0) if with_range else f"{pct(net, 0)} [{pct(lo_, 0)}, {pct(hi_, 0)}]"
        out = [str(len(ap_)), gtxt, str(sum(float(r['test_mean_ratio']) > 1 for r in ap_))]
        return out + ([str(sum(int(r['worse']) for r in apr))] if with_range else [])
    T = ['\\begin{table*}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}',
         '\\caption{What each rule does on the same ' + str(runs) + ' runs per arm (Study 2). Applied: runs where the rule uses a correction. Gain: in-distribution net gain over $\\mZ$ (\\%) with a seed-cluster bootstrap 95\\% interval. Worse: applied runs that are worse than $\\mZ$ on average in-distribution (mean clipped episode ratio above 1). The count rule uses 36 judge episodes as in Study 1; the magnitude rules use 100. Range-widening outcomes are in Fig.~\\ref{fig:safety}b; all variants are in Supplementary Table~S2. \\textsuperscript{a}Post hoc: the count rule with the same 100 judge episodes as the magnitude rules.}\\label{tab:gates}',
         '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'rcr' * 3 + '@{}}\\toprule',
         ' & \\multicolumn{3}{c}{' + SYSN['planar_2link'] + ' arm} & \\multicolumn{3}{c}{' + SYSN['planar_3link'] + ' arm} & \\multicolumn{3}{c}{' + SYSN['franka_7dof'] + ' arm}\\\\',
         '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}',
         'Rule' + ' & Applied & Gain & Worse' * 3 + '\\\\\\midrule']
    # post hoc: count rule with the same 100 judge episodes (whole candidate; applied runs use the always-applied forecast)
    from c7.c7_rules import cp_upper as _cpu
    rep7c = {json.loads(f.read_text())['key']: json.loads(f.read_text()) for f in (c7d / 'cases').glob('*/complete.json')}
    def count_100(s):
        idr = sel7(s, 'id', 'always_on'); kz = sel7(s, 'id', 'keep_Z'); acc = {}
        for r in idr:
            k = int(np.sum(np.array(rep7c[r['key']]['judge_ratio_whole']) > 1)); acc[r['key']] = _cpu(k, 100) <= 0.1
        mse = [float(r['mse']) if acc[r['key']] else float(r['mse_Z']) for r in idr]
        worse = sum(acc[r['key']] and float(r['test_mean_ratio']) > 1 for r in idr)
        return sum(acc.values()), 1 - np.sum(mse) / np.sum([float(r['mse_Z']) for r in idr]), worse
    for g, name in MAIN:
        T.append(name + ' & ' + ' & '.join(c for s in SYS for c in gate_row(g, s)) + '\\\\')
        if g == 'strict_ltt_c1':
            cells = []
            for s in SYS:
                a_, n_, w_ = count_100(s); cells += [str(a_), pct(n_, 0), str(w_)]
            T.append('Count rule, 100 ep.\\textsuperscript{a} & ' + ' & '.join(cells) + '\\\\')
    v['CxCountHundredTotal'] = sum(count_100(s)[0] for s in SYS)
    T += ['\\bottomrule\\end{tabular*}', '\\end{table*}']
    (gen / 'table_gates.tex').write_text('\n'.join(T) + '\n')
    TF = ['% supplementary: all gates, including the n = 36 and tail variants and range widening',
          '\\begin{tabular}{l' + 'rrrr' * 3 + '}\\toprule', 'Rule' + ' & Appl. & Gain & Worse & Range worse' * 3 + '\\\\\\midrule']
    for g, name in GATES:
        TF.append(name + ' & ' + ' & '.join(c for s in SYS for c in gate_row(g, s, True)) + '\\\\')
    TF += ['\\bottomrule\\end{tabular}']
    (gen / 'supp_table_gates_full.tex').write_text('\n'.join(TF) + '\n')
    yn = lambda ok: 'Yes' if ok else 'No'
    m1 = C1['M1']; n2 = C7['N2']
    TC = ['\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{2.5pt}',
          '\\caption{What we pre-specified and what we found. Study 1 was fixed before its data; Study 2 was fixed after Study 1 and used new seeds. ``Unclear'' means too few accepted runs to judge.}\\label{tab:claims}',
          '\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}p{0.47\\columnwidth}ccc@{}}\\toprule',
          'Pre-specified statement & 2-link & 3-link & Franka\\\\\\midrule',
          '\\multicolumn{4}{@{}l}{\\emph{Study 1 (count rule, 36 judge episodes)}}\\\\',
          'Always applying the correction harms $\\ge$10\\% of runs at a wider range & ' + ' & '.join(f"{yn(m1[s]['verdict'] == 'supported')}, {pct(m1[s]['share'], 0)}\\%" for s in SYS) + '\\\\',
          'Corrections the count rule accepts are safe ('+'/'.join(str(C1['M2'][s]['accepted']) for s in SYS)+' accepted) & ' + ' & '.join(f"Unclear" for s in SYS) + '\\\\',
          'The count rule gains $\\ge$5\\% (accepted/runs) & -- & -- & ' + f"No ({C1['M2']['franka_7dof']['accepted']}/{runs})" + '\\\\',
          '\\multicolumn{4}{@{}l}{\\emph{Study 2 (magnitude rule, 100 judge episodes)}}\\\\',
          'The magnitude rule gains $\\ge$5\\% & -- & -- & ' + f"Yes, {pct(n1['median'], 0)}\\%" + '\\\\',
          'Applied corrections are not worse on average in $>$10\\% of runs & ' + ' & '.join(f"Yes (0/{n2[s]['applied']})" for s in SYS) + '\\\\',
          'Testing directions one by one applies $\\ge$10\\% more corrections & -- & -- & ' + f"No ({n3['applied_dir']}/{n3['applied_whole']})" + '\\\\',
          '\\bottomrule\\end{tabular*}', '\\end{table}']
    assert all(n2[s]['verdict'] == 'supported' for s in SYS) and n1['verdict'] == 'supported' and n3['verdict'] == 'not_supported'
    assert all(C1['M2'][s]['verdict'] == 'inconclusive' for s in SYS) and C1['M3']['verdict'] == 'not_supported'
    (gen / 'table_claims.tex').write_text('\n'.join(TC) + '\n')
    # ---- figures
    import matplotlib; matplotlib.use('Agg')
    import figure_ictx as FI
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    GC = {'always_on': '#D55E00', 'mean_loss_rule': '#E69F00', 'strict_ltt_c1': '#999999', 'mag_whole_gate': '#56B4E9', 'mag_dir_gate': '#0072B2'}
    GL = {'always_on': 'always', 'mean_loss_rule': 'mean-loss', 'strict_ltt_c1': 'count', 'mag_whole_gate': 'magnitude', 'mag_dir_gate': 'mag.+dir.'}
    MK = {'planar_2link': 'o', 'planar_3link': 's', 'franka_7dof': '^'}
    from scipy.stats import beta as _beta
    def cp2(k, n):
        return (0.0 if k == 0 else float(_beta.ppf(0.025, k, n - k + 1))), (1.0 if k == n else float(_beta.ppf(0.975, k + 1, n - k)))
    # H2: safety-benefit plane with 95% intervals and rule paths
    fig = FI._setup(1.6); ax, bx = fig.subplots(1, 2)
    path = ['strict_ltt_c1', 'mag_dir_gate', 'always_on']
    for s in SYS:
        pts = {}
        for g in GC:
            idr = sel7(s, 'id', g); rr = sel7(s, 'range', g)
            k1 = sum(r['applied'] == '1' and float(r['test_mean_ratio']) > 1 for r in idr); k2 = sum(r['applied'] == '1' and r['worse'] == '1' for r in rr)
            g1 = 1 - np.mean([float(r['mse']) for r in idr]) / np.mean([float(r['mse_Z']) for r in idr]); g2 = 1 - np.mean([float(r['mse']) for r in rr]) / np.mean([float(r['mse_Z']) for r in rr])
            (l1, h1), (l2, h2) = boot_net(idr, 1000), boot_net(rr, 1000); (a1, b1), (a2, b2) = cp2(k1, runs), cp2(k2, runs)
            pts[g] = ((100 * k1 / runs, 100 * g1), (100 * k2 / runs, 100 * g2))
            ax.errorbar(100 * k1 / runs, 100 * g1, xerr=[[100 * (k1 / runs - a1)], [100 * (b1 - k1 / runs)]], yerr=[[100 * (g1 - l1)], [100 * (h1 - g1)]], fmt='none', ecolor=GC[g], elinewidth=.6, capsize=0, zorder=2)
            bx.errorbar(100 * k2 / runs, 100 * g2, xerr=[[100 * (k2 / runs - a2)], [100 * (b2 - k2 / runs)]], yerr=[[100 * (g2 - l2)], [100 * (h2 - g2)]], fmt='none', ecolor=GC[g], elinewidth=.6, capsize=0, zorder=2)
            ax.scatter(100 * k1 / runs, 100 * g1, s=22, marker=MK[s], color=GC[g], edgecolor='k', linewidth=.3, zorder=3)
            bx.scatter(100 * k2 / runs, 100 * g2, s=22, marker=MK[s], color=GC[g], edgecolor='k', linewidth=.3, zorder=3)
        for axx, j in ((ax, 0), (bx, 1)):
            axx.plot([pts[g][j][0] for g in path], [pts[g][j][1] for g in path], color='#bbbbbb', lw=.6, zorder=1)
    ax.set_xlabel('worse on average (% of runs)'); ax.set_ylabel('net gain (%)'); ax.set_title('(a) judged conditions', fontsize=FI.FS)
    bx.set_xlabel('worse (% of runs)'); bx.set_title('(b) wider range (not covered)', fontsize=FI.FS)
    bx.set_yscale('symlog', linthresh=10); ax.axhline(0, color='#888', lw=.5); bx.axhline(0, color='#888', lw=.5)
    bx.set_yticks([-100, -10, 0, 10]); bx.set_yticklabels(['\u2212100', '\u221210', '0', '10'])   # plain labels: no small exponent glyphs (A46)
    ax.set_xlim(-1, 26); bx.set_xlim(-5, 100)
    hg = [Line2D([], [], ls='', marker='o', color=GC[g], label=GL[g]) for g in GC]
    hs = [Line2D([], [], ls='', marker=MK[s], color='w', markeredgecolor='k', label={'planar_2link': '2-link', 'planar_3link': '3-link', 'franka_7dof': 'Franka'}[s]) for s in SYS]
    ax.legend(handles=hg, ncol=1, frameon=False, fontsize=FI.FS - 1, loc='center right', handletextpad=0.2, borderaxespad=0.1)
    bx.legend(handles=hs, ncol=1, frameon=False, fontsize=FI.FS - 1, loc='lower left', handletextpad=0.2, borderaxespad=0.2)
    FI._save(fig, gen / 'fig_safety'); plt.close(fig)
    # H4: why the count rule rejects, with quadrant counts and both studies' count limits
    rep7 = [json.loads(open(f).read()) for f in sorted(glob.glob(str(c7d / 'cases' / '*' / 'complete.json')))]
    acc7 = {(r['key']): r['applied'] == '1' for r in R7 if r['method'] == 'mag_whole_gate' and r['split'] == 'id'}
    fig = FI._setup(1.65); gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05); hx = fig.add_subplot(gs[0]); cx = fig.add_subplot(gs[1], sharex=hx)
    allx = []
    for s in SYS:
        pts = [(100 * np.mean(np.array(x['judge_ratio_whole']) > 1), float(np.mean(np.minimum(x['judge_ratio_whole'], 2.0))), acc7[x['key']]) for x in rep7 if x['system'] == s]
        xs_, ys_, ac = zip(*pts); allx += list(xs_)
        cx.scatter([a for a, c in zip(xs_, ac) if c], [b for b, c in zip(ys_, ac) if c], s=10, marker=MK[s], color='#0072B2', alpha=.75, linewidth=0)
        cx.scatter([a for a, c in zip(xs_, ac) if not c], [b for b, c in zip(ys_, ac) if not c], s=10, marker=MK[s], facecolor='none', edgecolor='#999999', linewidth=.6)
    hx.hist(allx, bins=np.arange(0, 101, 4), color='#cccccc', edgecolor='white'); hx.axis('off')
    cx.axhline(1, color='k', lw=.6, ls='--'); cx.axvline(4, color='#D55E00', lw=.8, ls=':'); cx.axvline(0, color='#D55E00', lw=.8, ls='-')
    xs_all = [100 * np.mean(np.array(x['judge_ratio_whole']) > 1) for x in rep7]; ys_all = [float(np.mean(np.minimum(x['judge_ratio_whole'], 2.0))) for x in rep7]
    q = {'pass, below 1': sum(a <= 4 and b < 1 for a, b in zip(xs_all, ys_all)), 'fail, below 1': sum(a > 4 and b < 1 for a, b in zip(xs_all, ys_all)),
         'fail, above 1': sum(a > 4 and b >= 1 for a, b in zip(xs_all, ys_all)), 'pass, above 1': sum(a <= 4 and b >= 1 for a, b in zip(xs_all, ys_all))}
    v['CxSevQuadFailBelow'] = q['fail, below 1']; v['CxSevQuadPassBelow'] = q['pass, below 1']; v['CxSevQuadFailAbove'] = q['fail, above 1']
    cx.text(81, 1.27, f"{q['fail, below 1']} runs: count rule rejects,\nmean loss lower", fontsize=FI.FS - 1, ha='center', color='#0072B2')
    cx.text(81, 1.04, f"{q['fail, above 1']} runs: both reject", fontsize=FI.FS - 1, ha='center', color='#666666')
    cx.text(5.5, 1.56, 'count rule accepts only\nleft of 4% (100 episodes)\nor at 0 (36 episodes)', fontsize=FI.FS - 1, color='#D55E00', va='top')
    cx.set_xlabel('judge episodes made worse (%)'); cx.set_ylabel('mean clipped loss ratio'); cx.set_xlim(-2, 100); cx.set_ylim(0.2, 1.6)
    hh = [Line2D([], [], ls='', marker='o', color='#0072B2', label='magnitude rule accepts'), Line2D([], [], ls='', marker='o', mfc='none', mec='#999999', label='magnitude rule rejects')]
    cx.legend(handles=hh, ncol=1, frameon=False, fontsize=FI.FS - 1, loc='lower right', handletextpad=0.2, borderaxespad=0.3, labelspacing=0.25)
    FI._save(fig, gen / 'fig_ratios'); plt.close(fig)
    v['CxSevJudgeWorseMedFr'] = pct(np.median([np.mean(np.array(x['judge_ratio_whole']) > 1) for x in rep7 if x['system'] == 'franka_7dof']), 0)
    v['CxSevJudgeZeroWorse'] = sum(np.sum(np.array(x['judge_ratio_whole']) > 1) <= 4 for x in rep7)
    # H1: forest plot of the pre-specified statements
    rows_f = []
    for s in SYS:
        m1 = C1['M1'][s]; rows_f.append(('Always applied, harmed at wider range', s, 100 * m1['share'], 100 * m1['lower95'], 100 * m1['upper95'], 10, 'harm'))
    m3 = C1['M3']; rows_f.append(('Count rule, median gain', 'franka_7dof', 100 * m3['median'], 100 * m3['boot95'][0], 100 * m3['boot95'][1], 5, 'gain'))
    rows_f.append(('Magnitude rule, median gain', 'franka_7dof', 100 * n1['median'], 100 * n1['boot95'][0], 100 * n1['boot95'][1], 5, 'gain'))
    for s in SYS:
        n2s = C7['N2'][s]; rows_f.append(('Magnitude rule, worse on average', s, 100 * n2s['mean_ratio_worse'] / max(n2s['applied'], 1), 0.0, 100 * n2s['upper95'], 10, 'harm'))
    ad = [int(r['applied']) for r in sel7('franka_7dof', 'id', 'mag_dir_gate')]; aw = [int(r['applied']) for r in sel7('franka_7dof', 'id', 'mag_whole_gate')]
    seeds7 = sorted({r['seed'] for r in sel7('franka_7dof', 'id', 'mag_dir_gate')}); bd = []
    rd_ = {sd: [(int(x['applied']), int(y['applied'])) for x, y in zip(sel7('franka_7dof', 'id', 'mag_dir_gate'), sel7('franka_7dof', 'id', 'mag_whole_gate')) if x['seed'] == sd] for sd in seeds7}
    for _ in range(2000):
        pk = [t for sd in BRNG.choice(seeds7, len(seeds7)) for t in rd_[sd]]; bd.append(100 * (np.mean([t[0] for t in pk]) - np.mean([t[1] for t in pk])))
    rows_f.append(('Directions, extra applied runs', 'franka_7dof', 100 * (np.mean(ad) - np.mean(aw)), float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)), 10, 'gain'))
    fig = FI._setup(1.55); fx = fig.subplots(1, 1); ylab = []
    for i, (lab, s, est, lo, hi, thr, kind) in enumerate(rows_f):
        y = len(rows_f) - 1 - i
        fx.plot([lo, hi], [y, y], color='#0072B2' if kind == 'gain' else '#D55E00', lw=1.2)
        fx.scatter(est, y, marker=MK[s], s=22, color='#0072B2' if kind == 'gain' else '#D55E00', edgecolor='k', linewidth=.3, zorder=3)
        fx.plot([thr, thr], [y - .35, y + .35], color='k', lw=.9)
        ylab.append(lab + {'planar_2link': ' (2-link)', 'planar_3link': ' (3-link)', 'franka_7dof': ' (Franka)'}[s])
    fx.set_yticks(range(len(rows_f))); fx.set_yticklabels(ylab[::-1], fontsize=FI.FS - 1); fx.set_xlabel('% (95% CI; tick: threshold)')
    fx.axvline(0, color='#bbbbbb', lw=.5); fx.set_xlim(-12, 100)
    FI._save(fig, gen / 'fig_forest'); plt.close(fig)
    # H3: when does the correction pay (post hoc, descriptive)
    cfgs = [('12', '0.0'), ('12', '0.3'), ('36', '0.0'), ('36', '0.3')]
    M = np.zeros((3, 4)); T_ = [['' for _ in cfgs] for _ in SYS]
    for i, s in enumerate(SYS):
        for j, (nn, nz) in enumerate(cfgs):
            rr = [r for r in sel7(s, 'id', 'mag_dir_gate') if r['n'] == nn and r['noise'] == nz]
            M[i, j] = np.median([100 * float(r['rel_gain']) for r in rr]); T_[i][j] = f"{sum(r['applied'] == '1' for r in rr)}/{len(rr)}\n{M[i, j]:.0f}%"
    fig = FI._setup(1.3); hm = fig.subplots(1, 1)
    hm.imshow(M, cmap='Blues', vmin=0, vmax=45, aspect='auto')
    for i in range(3):
        for j in range(4):
            hm.text(j, i, T_[i][j], ha='center', va='center', fontsize=FI.FS - 1, color='white' if M[i, j] > 25 else 'black')
    hm.set_xticks(range(4)); hm.set_xticklabels(['12 ep., clean', '12 ep., noisy', '36 ep., clean', '36 ep., noisy'], fontsize=FI.FS - 1)
    hm.set_yticks(range(3)); hm.set_yticklabels(['2-link', '3-link', 'Franka'], fontsize=FI.FS - 1)
    hm.set_title('magnitude rule + directions: applied runs and median gain', fontsize=FI.FS - 1)
    FI._save(fig, gen / 'fig_conditions'); plt.close(fig)
    fr = lambda nn, nz: float(np.median([100 * float(r['rel_gain']) for r in sel7('franka_7dof', 'id', 'mag_dir_gate') if r['n'] == nn and r['noise'] == nz]))
    v['CxSevFrNoisyLo'] = f"{min(fr('12', '0.3'), fr('36', '0.3')):.0f}"; v['CxSevFrNoisyHi'] = f"{max(fr('12', '0.3'), fr('36', '0.3')):.0f}"
    v['CxSevFrCleanLo'] = f"{min(fr('12', '0.0'), fr('36', '0.0')):.0f}"; v['CxSevFrCleanHi'] = f"{max(fr('12', '0.0'), fr('36', '0.0')):.0f}"
    ap3 = lambda nn, nz: sum(r['applied'] == '1' for r in sel7('planar_3link', 'id', 'mag_dir_gate') if r['n'] == nn and r['noise'] == nz)
    v['CxSevThreeNoisySmallAcc'] = ap3('12', '0.3'); v['CxSevThreeCleanSmallAcc'] = ap3('12', '0.0'); v['CxCfgRuns'] = runs // 4
    # R5 / R6 (supplement): judge size (post hoc, descriptive) and per-run gains
    from c7.c7_rules import magnitude_gate, cp_upper
    fig = FI._setup(1.9); dx = fig.subplots(1, 1); ns = [20, 36, 60, 100]
    for s in SYS:
        xr = [x for x in rep7 if x['system'] == s]
        mag = [100 * np.mean([magnitude_gate(np.array(x['judge_ratio_whole'][:n]))[0] for x in xr]) for n in ns]
        cnt = [100 * np.mean([cp_upper(int(np.sum(np.array(x['judge_ratio_whole'][:n]) > 1)), n) <= 0.1 for x in xr]) for n in ns]
        dx.plot(ns, mag, marker=MK[s], color='#0072B2', lw=.9); dx.plot(ns, cnt, marker=MK[s], color='#999999', lw=.9, ls='--')
    dx.set_xlabel('judge episodes'); dx.set_ylabel('runs accepted (%)'); dx.set_title('post hoc, whole-matrix rules (blue: magnitude, grey: count)', fontsize=FI.FS - 1)
    dx.legend(handles=hs, frameon=False, fontsize=FI.FS - 1, loc='upper left')
    FI._save(fig, gen / 'supp_judge_size'); plt.close(fig)
    fig = FI._setup(2.2); axs = fig.subplots(1, 3, sharey=True)
    for k, s in enumerate(SYS):
        for j, g in enumerate(GC):
            yv = [100 * float(r['rel_gain']) for r in sel7(s, 'id', g)]
            axs[k].scatter(j + np.random.default_rng(j).uniform(-.25, .25, len(yv)), np.clip(yv, -60, 100), s=3, color=GC[g], alpha=.6, linewidth=0)
        axs[k].set_xticks(range(len(GC))); axs[k].set_xticklabels([GL[g] for g in GC], rotation=60, fontsize=FI.FS - 1); axs[k].set_title({'planar_2link': '2-link', 'planar_3link': '3-link', 'franka_7dof': 'Franka'}[s], fontsize=FI.FS)
        axs[k].axhline(0, color='#888', lw=.5)
    axs[0].set_ylabel('per-run gain, in-distribution (%, clipped at -60)')
    FI._save(fig, gen / 'supp_gains'); plt.close(fig)
    fig = FI._setup(2.0); ex = fig.subplots(1, 3, sharey=True)
    for k, s in enumerate(SYS):
        for g in GC:
            rr = [r for r in sel7(s, 'range', g) if r['applied'] == '1']
            if not rr:
                continue
            ratio = np.sort([float(r['mse']) / float(r['mse_Z']) for r in rr]); ex[k].step(ratio, np.arange(1, len(ratio) + 1) / len(ratio), color=GC[g], lw=.9, where='post', label=GL[g])
        ex[k].set_xscale('log'); ex[k].axvline(1, color='k', lw=.5, ls='--')
        from matplotlib.ticker import FuncFormatter as _FF, NullFormatter as _NF
        ex[k].xaxis.set_major_formatter(_FF(lambda v, _: f'{v:g}')); ex[k].xaxis.set_minor_formatter(_NF()); ex[k].set_title({'planar_2link': '2-link', 'planar_3link': '3-link', 'franka_7dof': 'Franka'}[s], fontsize=FI.FS)
        ex[k].set_xlabel('loss ratio (log)', fontsize=FI.FS - 1)
    ex[0].set_ylabel('share of applied runs'); ex[0].legend(handles=[Line2D([], [], color=GC[g_], lw=.9, label=GL[g_]) for g_ in GC], frameon=False, fontsize=FI.FS - 1, loc='lower right', handlelength=1.2)
    FI._save(fig, gen / 'supp_range_ecdf'); plt.close(fig)
    TM = ['% supplementary: median per-run gain [seed-cluster bootstrap 95% interval], in-distribution', '\\begin{tabular}{l' + 'c' * 3 + '}\\toprule', 'Rule & 2-link & 3-link & Franka\\\\\\midrule']
    for g, name in GATES:
        cells = []
        for s in SYS:
            rr = sel7(s, 'id', g); seeds = sorted({r['seed'] for r in rr}); by = {sd: [float(r['rel_gain']) for r in rr if r['seed'] == sd] for sd in seeds}
            meds = [np.median(np.concatenate([by[sd] for sd in BRNG.choice(seeds, len(seeds))])) for _ in range(1000)]
            cells.append(f"{pct(np.median([float(r['rel_gain']) for r in rr]), 0)} [{pct(np.percentile(meds, 2.5), 0)}, {pct(np.percentile(meds, 97.5), 0)}]")
        TM.append(name + ' & ' + ' & '.join(cells) + '\\\\')
    TM += ['\\bottomrule\\end{tabular}']
    (gen / 'supp_table_median_ci.tex').write_text('\n'.join(TM) + '\n')
    uc = json.loads((c7d / 'analysis' / 'ucb_check.json').read_text()) if (c7d / 'analysis' / 'ucb_check.json').exists() else None
    if uc:
        v['CxUcbDecisions'] = f"{uc['decisions_checked']:,}".replace(',', '{,}'); v['CxUcbDis'] = uc['decision_disagreements']; v['CxGateMs'] = f"{uc['ms_per_bound_n100_linux_1core']:.0f}"
        checks['bound definition does not change any decision'] = uc['decision_disagreements'] == 0
    # C2 (reduced): ICT cost measured on Linux (1 core) and Apple M5 Max (reports/ict_cost_*.json)
    icl, icd = HERE.parent / 'reports' / 'ict_cost_linux.json', HERE.parent / 'reports' / 'ict_cost_darwin.json'
    if icl.exists() and icd.exists():
        L_, D_ = json.loads(icl.read_text())['arms'], json.loads(icd.read_text())['arms']
        v['CxIctMacGate'] = f"{max(a['gate_ms_median'] for a in D_.values()):.1f}"; v['CxIctLinGate'] = f"{max(a['gate_ms_median'] for a in L_.values()):.1f}"
        v['CxIctMacApplyUs'] = f"{max(a['apply_us_median_per_forecast'] for a in D_.values()):.0f}"
        v['CxIctJudgeKB'] = f"{max(a['bytes_observed_judge_Y'] for a in D_.values()) / 1e3:.0f}"; v['CxIctMatrixB'] = max(a['bytes_matrix_A'] for a in D_.values())
        checks['ICT data volume equal on both machines'] = all(L_[k]['bytes_observed_judge_Y'] == D_[k]['bytes_observed_judge_Y'] for k in D_)
    # A35: seed-level bound (runs of one seed share judge and test episodes)
    from scipy.stats import beta as _b2
    for s_ in SYS:
        by = {}
        for r in sel7(s_, 'id', 'mag_dir_gate'):
            if r['applied'] == '1':
                by.setdefault(r['seed'], []).append(float(r['test_mean_ratio']) > 1)
        n_, k_ = len(by), sum(any(x) for x in by.values())
        up_ = 1.0 if k_ >= n_ else float(_b2.ppf(0.95, k_ + 1, n_ - k_))
        v[f'CxSevSeed{TAG[s_]}N'] = n_; v[f'CxSevSeed{TAG[s_]}K'] = k_; v[f'CxSevSeed{TAG[s_]}Up'] = pct(up_)
        checks[f'seed-level bound below 10% ({s_})'] = up_ <= 0.10
    # A37: post hoc sensitivity (impl/c7/c7_sensitivity.py)
    sp = c7d / 'analysis' / 'sensitivity.csv'
    if sp.exists():
        SR = rd(sp)
        v['CxSensTauZeroWorse'] = sum(int(r[f'{s_}_worse_avg']) for r in SR if float(r['tau']) == 0 for s_ in SYS)
        v['CxSensTauFiveWorseMax'] = max(int(r[f'{s_}_worse_avg']) for r in SR if float(r['tau']) == 0.05 for s_ in SYS)
        T7 = ['% supplementary: post hoc sensitivity of the whole-matrix magnitude rule (applied / applied and worse on average)', '\\begin{tabular}{rrr' + 'rr' * 3 + '}\\toprule', '$\\tau$ & $n$ & clip' + ''.join(f' & \\multicolumn{{2}}{{c}}{{{SYSN[s_]}}}' for s_ in SYS) + '\\\\\\midrule']
        for r in SR:
            T7.append(f"{r['tau']} & {r['n']} & {r['clip']} & " + ' & '.join(f"{r[s_ + '_applied']} & {r[s_ + '_worse_avg']}" for s_ in SYS) + '\\\\')
        T7 += ['\\bottomrule\\end{tabular}']; (gen / 'supp_table_sensitivity.tex').write_text('\n'.join(T7) + '\n')
    ag = c7d / 'analysis' / 'applied_only_gain.json'
    if ag.exists():
        AG = json.loads(ag.read_text()); v['CxSevFrAppliedOnlyMed'] = pct(AG['mag_dir_gate']['franka_7dof'], 0)
    if 'CxIctJudgeKB' in v:
        v['CxIctJudgeSecTen'] = f"{float(v['CxIctJudgeKB']) * 8 / 10000:.1f}"
    lines = ['% generated by impl/make_paper_assets_c1c7.py — do not edit']
    lines += [f'\\newcommand{{\\{k}}}{{{x}}}' for k, x in v.items()]
    (gen / 'numbers_c.tex').write_text('\n'.join(lines) + '\n')
    failed = [k for k, ok in checks.items() if not ok]
    (gen / 'provenance_c.json').write_text(json.dumps({'c1': str(c1d.name), 'c7': str(c7d.name), 'checks': checks}, indent=2))
    print(f'wrote {gen} | checks passed: {len(checks) - len(failed)}' + (f' FAILED: {failed}' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
