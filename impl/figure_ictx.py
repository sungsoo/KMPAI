"""English result figures of ICTExpress-v1 (elsarticle 5p column width 252 pt, printed at scale 1.0, text 8 pt).

    fig_risk(analysis_dir, out_stem)   (a) risk vs gain under the two shifts, (b) share of worse runs over time (damping)
    fig_eta(eta_summary_csv, out_stem) (a) applied runs vs eta, (b) ID episode-level deterioration (strict / eta-level)
Every glyph is Gentium Book Basic (figure_style.apply + check_pdf_fonts); no Hangul.
"""
import csv
from pathlib import Path
import figure_style
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

COLUMN_PT = 252.0
FS = 8 * 72 / 72.27
PLOT = {'keep_original': ('keep $\\mathbf{Z}$', '#777777', 'o'), 'fixed_matrix': ('fixed matrix', '#234D70', 's'),
        'mean_rule': ('mean-loss rule', '#B66A23', 'D'), 'bates_granger': ('per-step comb.', '#8E5572', 'v'),
        'ltt': ('LTT (ours)', '#187B73', '*'), 'equal_budget_retrain': ('retraining', '#4F7CAC', '^'),
        'oracle': ('oracle', '#999999', 'P')}


def read(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def _setup(height_in):
    figure_style.apply(FS)
    plt.rcParams.update({k: FS for k in ('font.size', 'axes.titlesize', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize')})
    return plt.figure(figsize=(COLUMN_PT / 72.27, height_in), layout='constrained')


def _save(fig, out_stem):
    out_stem = Path(out_stem); out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        figure_style.savefig(fig, out_stem.with_suffix('.' + ext), dpi=300)
    plt.close(fig)
    names = figure_style.pdf_fonts(out_stem.with_suffix('.pdf'))
    bad = [n for n in names if 'GentiumBookBasic' not in n]
    if bad:
        raise AssertionError(f'{out_stem}: non-Gentium fonts {bad}')
    return names


def fig_risk(analysis_dir, out_stem, horizon_s=10):
    an = Path(analysis_dir); S = {(r['split'], r['method']): r for r in read(an / 'summary.csv')}; C = read(an / 'curves.csv')
    fig = _setup(1.9); gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.28])
    ax, bx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for m, (lab, col, mk) in PLOT.items():
        if m == 'oracle':
            continue
        for split, fill in (('ood_scale', True), ('ood_damp', False)):
            r = S[(split, m)]
            ax.scatter(100 * float(r['worse_rate']), 100 * float(r['net_gain']), s=55 if m == 'ltt' else 22, marker=mk,
                       facecolors=col if fill else 'none', edgecolors=col, linewidths=.9, zorder=3 if m == 'ltt' else 2,
                       label=lab if split == 'ood_scale' else None)
    ax.axhline(0, color='#555555', lw=.6); ax.axvline(0, color='#555555', lw=.6); ax.margins(x=.08, y=.18)
    ax.set_xlabel('runs worse than $\\mathbf{Z}$ (%)'); ax.set_ylabel('net gain (%)'); ax.set_title('(a) risk and gain (OOD)')
    for m in ['fixed_matrix', 'mean_rule', 'bates_granger', 'ltt']:
        _, col, _ = PLOT[m]
        rr = [r for r in C if r['split'] == 'ood_damp' and r['method'] == m]
        bx.plot([float(r['time']) for r in rr], [float(r['worse_share']) for r in rr], color=col, lw=2.0 if m == 'ltt' else 1.0,
                ls='-' if m in ('ltt', 'fixed_matrix') else '--')
    bx.set_xlabel('forecast time (s)'); bx.set_ylabel('share of runs worse'); bx.set_ylim(0, 1); bx.set_title('(b) damping shift')
    bx.set_xticks(list(range(0, horizon_s + 1, 2 if horizon_s > 5 else 1))); bx.set_yticks([0, .5, 1])
    for a in (ax, bx):
        a.grid(alpha=.15)
    ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
    lx = fig.add_subplot(gs[1, :]); lx.axis('off'); h, l = ax.get_legend_handles_labels()
    lx.legend(h, l, loc='center', ncol=3, frameon=False, handletextpad=.2, columnspacing=.8, borderaxespad=0, labelspacing=.25)
    return _save(fig, out_stem)


def fig_eta(eta_summary_csv, out_stem, rows_csv=None):
    R = read(eta_summary_csv); RW = read(rows_csv) if rows_csv else None
    get = lambda split, tgt: sorted([r for r in R if r['split'] == split and r['target'] == tgt], key=lambda r: float(r['eta']))
    fig = _setup(1.42); ax, bx = fig.subplots(1, 2)
    for tgt, col, mk, lab in (('posthoc', '#187B73', 'o', 'post-hoc'), ('retrain', '#4F7CAC', '^', 'retraining')):
        rr = get('id', tgt)
        ax.plot([100 * float(r['eta']) for r in rr], [int(r['applied']) for r in rr], color=col, marker=mk, ms=4, lw=1.2, label=lab)
    ax.set_xlabel('tolerance eta (%)'); ax.set_ylabel('applied runs (of 80)'); ax.set_title('(a) acceptance'); ax.set_ylim(0, 80)
    ax.legend(loc='upper left', frameon=False, handlelength=1.4)
    rr = [r for r in get('id', 'posthoc') if int(r['ep_applied']) > 0]
    x = [100 * float(r['eta']) for r in rr]
    def strict_nt(eta):
        sel = [w for w in RW if w['split'] == 'id' and w['target'] == 'posthoc' and float(w['eta']) == eta and w['applied'] == '1'
               and abs(float(w['mse']) - float(w['mse_original'])) / float(w['mse_original']) >= 1e-12]
        n = sum(int(w['ep_n']) for w in sel); return 100 * sum(int(w['ep_worse_strict']) for w in sel) / n if n else 0.0
    ys = [strict_nt(float(r['eta'])) for r in rr] if RW else [100 * float(r['ep_worse_strict_rate']) for r in rr]
    bx.plot(x, ys, color='#B66A23', marker='s', ms=3.5, lw=1.2, label='worse than $\\mathbf{Z}$')
    bx.plot(x, [100 * float(r['ep_exceed_eta_rate']) for r in rr], color='#187B73', marker='o', ms=3.5, lw=1.2, label='worse by > eta')
    bx.axhline(10, color='#555555', lw=.7, ls=':'); bx.text(10.3, 10, 'delta', fontsize=FS, va='center', ha='left')
    bx.set_xlabel('tolerance eta (%)'); bx.set_ylabel('ID test episodes (%)'); bx.set_title('(b) post-hoc, applied runs')
    bx.set_ylim(0, 22); bx.legend(loc='upper left', frameon=False, handlelength=1.4)
    for a in (ax, bx):
        a.grid(alpha=.15); a.set_xticks([0, 5, 10])
    return _save(fig, out_stem)


if __name__ == '__main__':
    import sys
    print(fig_risk(sys.argv[1], sys.argv[2])); print(fig_eta(sys.argv[3], sys.argv[4]))
