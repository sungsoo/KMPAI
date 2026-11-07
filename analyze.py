"""Paired seed bootstrap and plots; no test-set model selection.
Intervals describe this finite synthetic benchmark; not corrected for multiplicity.
"""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze(root):
    root=Path(root);reports=[json.loads(p.read_text()) for p in sorted((root/'cases').glob('*/complete.json'))]
    groups={}
    for r in reports:groups.setdefault((r['system'],r.get('regime','heterogeneous'),r['n'],r['noise']),[]).append(r)
    rng=np.random.default_rng(7719);stats=[];mechanism=[]
    for key,rs in groups.items():
        for split in ['id','ood']:
            for baseline in ['residual','residual_equal_budget','global_gate','ridge','linear_id']:
                if split+'_'+baseline not in rs[0]['metrics']:continue
                a=np.array([r['metrics'][split+'_coordinate_gate']['mse'] for r in rs],dtype=float)
                b=np.array([r['metrics'][split+'_'+baseline]['mse'] for r in rs],dtype=float)
                valid=np.isfinite(a)&np.isfinite(b)
                record=dict(zip(['system','regime','n','noise'],key),split=split,baseline=baseline,seeds=len(rs),failed_seeds=int((~valid).sum()))
                if valid.all():
                    delta=a-b;draw=rng.integers(0,len(rs),(10000,len(rs)));boot=delta[draw].mean(1)
                    record.update(delta_mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),ratio_of_means=float(a.mean()/max(b.mean(),1e-15)),seed_wins=int((a<b).sum()))
                else:record.update(delta_mean=None,ci95=None,ratio_of_means=None,seed_wins=None)
                stats.append(record)
        axis={}
        for m in ['residual','coordinate_gate']:
            arr=np.array([r['metrics']['id_'+m]['axis_mse'] for r in rs],dtype=float)
            # Do not silently omit failed seeds/axes in a finite-only mean.
            axis[m]={'x_qv_mse':float(arr[:,[0,2]].mean()) if np.isfinite(arr[:,[0,2]]).all() else None,'y_qv_mse':float(arr[:,[1,3]].mean()) if np.isfinite(arr[:,[1,3]]).all() else None}
        mechanism.append(dict(zip(['system','regime','n','noise'],key),coordinate_alpha=np.mean([r['gates']['coordinate'] for r in rs],0).tolist(),global_alpha=float(np.mean([r['gates']['global'] for r in rs])),id_axis=axis))
    (root/'paired_statistics.json').write_text(json.dumps(stats,indent=2,allow_nan=False))
    (root/'mechanism.json').write_text(json.dumps(mechanism,indent=2,allow_nan=False))
    f,axs=plt.subplots(1,2,figsize=(7,2.7))
    for ax,split in zip(axs,['id','ood']):
        subset=[r for r in stats if r['split']==split and r['baseline']=='residual_equal_budget']
        vals=[r['ratio_of_means'] for r in subset]
        ax.bar(np.arange(len(vals)),[v if v is not None else np.nan for v in vals],color=['#aaaaaa' if v is None else '#346d9a' if v<1 else '#be785c' for v in vals]);ax.axhline(1,color='black',lw=.8)
        ax.set(title=f'{split.upper()}: coordinate / equal-budget residual',xlabel='Condition index (see JSON)',ylabel='Ratio of mean state MSE');ax.grid(axis='y',alpha=.2)
    f.tight_layout();f.savefig(root/'condition_ratios.pdf');f.savefig(root/'condition_ratios.png',dpi=170);plt.close(f)
    lines=['# 실측 결과 분석','', '각 조건에서 좌표별 보정과 기준선의 시드별 차이를 짝지어 10,000회 시드 재표집. 95% 구간은 탐색적이며 다중비교 보정 없음. 비유한 결과가 있으면 그 비교의 구간을 계산하지 않음.','', '|조건|ID 동일예산 대비 오차비|OOD 동일예산 대비 오차비|','|---|---:|---:|']
    for key in groups:
        rows=[r for r in stats if (r['system'],r['regime'],r['n'],r['noise'])==key and r['baseline']=='residual_equal_budget']
        vals={r['split']:r['ratio_of_means'] for r in rows}
        def display(v):return f'{v:.3f}' if v is not None else '실패/미정의'
        lines.append(f'|{key}|{display(vals["id"])}|{display(vals["ood"])}|')
    (root/'ANALYSIS.md').write_text('\n'.join(lines))
    return {'conditions':len(groups),'paired_comparisons':len(stats),'bootstrap_repetitions':10000,'multiple_comparison_correction':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);args=p.parse_args();print(analyze(args.results))
