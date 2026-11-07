"""Small, fully observed action-conditioned world-model study.

All simulated observations are four-dimensional physical states [qx,qy,vx,vy].
Training targets have synthetic acceleration-label noise; test rollouts are clean.
There is no learned image encoder, hidden-state estimation, or robotics claim.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, logging, os, platform, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

METHODS = ["prior", "direct", "physics_tuned", "residual", "global_gate", "coordinate_gate"]
CODE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def atomic_json(path,value):
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False));tmp.replace(path)

def file_hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def nominal_np(s, u, system):
    q,v=s[...,:2],s[...,2:]
    restore=q if system=="duffing" else np.sin(q)
    return -restore-0.1*v+u

def truth_np(s,u,system,regime='heterogeneous'):
    q,v=s[...,:2],s[...,2:]
    a=nominal_np(s,u,system).copy()
    if system=="duffing":
        a[...,1] += -0.5*q[...,1]-0.35*v[...,1]-0.25*q[...,1]**3+0.1*np.sin(q[...,0])
    else:
        a[...,1] += -0.6*np.sin(q[...,1])-0.25*v[...,1]-0.15*v[...,1]*np.abs(v[...,1])+0.1*np.sin(q[...,0])
    if regime=='both_wrong':
        a[...,0]+=-0.25*(q[...,0] if system=='duffing' else np.sin(q[...,0]))-0.2*v[...,0]-0.1*q[...,0]**3+0.1*np.sin(q[...,1])
    return a

def nominal_t(s,u,system):
    q,v=s[...,:2],s[...,2:]
    return -(q if system=="duffing" else torch.sin(q))-0.1*v+u

def step_np(s,u,system,dt,regime='heterogeneous'):
    # Explicit velocity update, followed by semi-implicit position update.
    v=s[...,2:]+dt*truth_np(s,u,system,regime)
    return np.concatenate((s[...,:2]+dt*v,v),axis=-1)

def generate(seed,n,steps,system,dt,scale=1.,regime='heterogeneous'):
    rng=np.random.default_rng(seed)
    states=np.empty((n,steps+1,4),dtype=np.float32)
    actions=np.empty((n,steps,2),dtype=np.float32)
    states[:,0]=rng.uniform(-scale,scale,(n,4))
    u=rng.normal(0,0.65*scale,(n,2))
    for t in range(steps):
        if t%8==0: u=rng.normal(0,0.65*scale,(n,2))
        actions[:,t]=u
        states[:,t+1]=step_np(states[:,t],u,system,dt,regime)
    return states,actions

def targets(data,system,noise,rng,regime='heterogeneous'):
    s,u=data
    x=np.concatenate((s[:,:-1],u),axis=-1).reshape(-1,6)
    clean=truth_np(s[:,:-1],u,system,regime).reshape(-1,2)
    y=clean+rng.normal(0,noise,clean.shape)
    return x.astype('float32'),y.astype('float32')

def choose_device(request):
    mps=hasattr(torch.backends,'mps') and torch.backends.mps.is_available()
    if request=='mps' and not mps: raise RuntimeError('Requested MPS unavailable; use --device cpu explicitly.')
    return torch.device('mps' if (request=='mps' or request=='auto' and mps) else 'cpu')

class Predictor(nn.Module):
    def __init__(self,x,y,system,mode,hidden,device):
        super().__init__()
        self.system,self.mode=system,mode
        self.register_buffer('xm',torch.tensor(x.mean(0),device=device))
        self.register_buffer('xs',torch.tensor(np.maximum(x.std(0),0.1),device=device))
        self.register_buffer('ys',torch.tensor(np.maximum(y.std(0),0.1),device=device))
        self.register_buffer('ym',torch.tensor(y.mean(0),device=device))
        self.net=nn.Sequential(nn.Linear(6,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,2)).to(device)
    def forward(self,x):
        raw=self.net((x-self.xm)/self.xs)*self.ys
        return raw+nominal_t(x[...,:4],x[...,4:],self.system) if self.mode=='residual' else raw+self.ym

def fit(x,y,xv,yv,system,mode,lam,cfg,seed,device,checkpoint=None,deadline=float('inf')):
    torch.manual_seed(seed)
    m=Predictor(x,y,system,mode,cfg['hidden'],device)
    opt=torch.optim.Adam(m.parameters(),lr=cfg['lr'])
    xt,yt,xvt,yvt=[torch.tensor(a,device=device) for a in [x,y,xv,yv]]
    prior=nominal_t(xt[:,:4],xt[:,4:],system)
    best,beststate,wait=float('inf'),None,0
    start_epoch,prior_seconds=0,0.
    if checkpoint and checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved['code_hash']!=CODE_HASH:raise ValueError('Checkpoint code changed')
        m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        best,beststate,wait=saved['best'],saved['beststate'],saved['wait']
        start_epoch,prior_seconds=saved['next_epoch'],saved['seconds']
    t0=time.perf_counter()
    def save(next_epoch):
        if checkpoint:
            tmp=checkpoint.with_suffix('.tmp')
            torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'best':best,'beststate':beststate,'wait':wait,'next_epoch':next_epoch,'seconds':prior_seconds+time.perf_counter()-t0,'code_hash':CODE_HASH},tmp);tmp.replace(checkpoint)
    epoch=start_epoch-1
    for next_epoch in range(start_epoch,cfg['epochs']):
        if wait>=cfg['patience']:break
        epoch=next_epoch
        if time.perf_counter()>deadline:
            save(epoch);raise TimeoutError('Budget exceeded inside training; checkpoint saved')
        m.train();opt.zero_grad()
        pred=m(xt)
        loss=(((pred-yt)/m.ys)**2).mean()+lam*(((pred-prior)/m.ys)**2).mean()
        loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): score=(((m(xvt)-yvt)/m.ys)**2).mean().item()
        if score<best-1e-6: best,beststate,wait=score,copy.deepcopy(m.state_dict()),0
        else: wait+=1
        if (epoch+1)%25==0:save(epoch+1)
        if wait>=cfg['patience']: break
    save(epoch+1)
    m.load_state_dict(beststate)
    return m,{'epochs':epoch+1,'validation_loss':best,'fit_seconds':prior_seconds+time.perf_counter()-t0,'lambda':lam}

def gate(m,xc,yc,system,device):
    with torch.no_grad():
        xt=torch.tensor(xc,device=device)
        p=nominal_t(xt[:,:4],xt[:,4:],system)
        r=(m(xt)-p).cpu().numpy()
        e=yc-p.cpu().numpy()
    denom=(r*r).sum(0)
    # Exactly solve constrained scalar SSE; zero denominator has any minimizer,
    # and we choose zero. No denominator floor that changes the argmin.
    coordinate=np.zeros_like(denom)
    np.divide((r*e).sum(0),denom,out=coordinate,where=denom>0)
    coordinate=np.clip(coordinate,0,1)
    global_=np.clip((r*e).sum()/float(denom.sum()),0,1) if denom.sum()>0 else 0.
    return coordinate, float(global_)

def features(x,system):
    q,v,u=x[...,:2],x[...,2:4],x[...,4:]
    return np.concatenate([np.ones((*x.shape[:-1],1)),q,v,u,q**3,np.sin(q),v*np.abs(v)],-1)

def ridge_fit(x,y,xv,yv,system,val_data,cfg,device,linear=False,deadline=float('inf')):
    f,fv=features(x,system),features(xv,system)
    if linear:f,fv=f[:,:7],fv[:,:7]
    best=(float('inf'),None,None)
    for lam in [1e-6,1e-3,0.1,1.,10.]:
        coef=np.linalg.solve(f.T@f+lam*np.eye(f.shape[1]),f.T@y)
        expanded=np.zeros((features(x[:1],system).shape[1],2));expanded[:len(coef)]=coef
        errs,_=rollout('ridge',None,None,val_data,system,cfg['dt'],device,expanded,deadline)
        score=float(errs.mean())
        if np.isfinite(score) and score<best[0]: best=(score,expanded,lam)
    if best[1] is None:raise RuntimeError('All ridge validation candidates diverged')
    return best[1],{'lambda':best[2],'validation_loss':float(best[0])}

def rollout(method,model,g,data,system,dt,device,ridgecoef=None,deadline=float('inf')):
    states,actions=data
    cur=torch.tensor(states[:,0],device=device)
    us=torch.tensor(actions,device=device)
    curves=[]
    with torch.no_grad():
        for t in range(actions.shape[1]):
            if time.perf_counter()>deadline:raise TimeoutError('Budget exceeded during rollout')
            u=us[:,t];p=nominal_t(cur,u,system)
            if method=='prior': a=p
            elif method in ['ridge','linear_id']:
                xx=torch.cat((cur,u),-1).cpu().numpy()
                a=torch.tensor(features(xx,system)@ridgecoef,dtype=torch.float32,device=device)
            else:
                a=model(torch.cat((cur,u),-1))
                if 'gate' in method: a=p+torch.tensor(g,dtype=torch.float32,device=device)*(a-p)
            v=cur[:,2:]+dt*a
            cur=torch.cat((cur[:,:2]+dt*v,v),-1)
            curves.append(cur.cpu().numpy())
    pred=np.stack(curves,1)
    errors=np.mean((pred-states[:,1:])**2,axis=-1) # episode x horizon
    return errors,pred

def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def run_case(cfg,system,n,noise,seed,device,root,deadline,regime='heterogeneous'):
    key=f'{system}_{regime}_n{n}_noise{noise}_seed{seed}'
    case=root/'cases'/key;case.mkdir(parents=True,exist_ok=True)
    if (case/'complete.json').exists():
        report=json.loads((case/'complete.json').read_text())
        if report.get('code_hash')!=CODE_HASH or report['config']!=cfg:raise ValueError('Stale case/code/config cache')
        for name,h in report['artifact_hashes'].items():
            if not (case/name).exists() or file_hash(case/name)!=h:raise ValueError(f'Corrupt case artifact: {name}')
        logging.info('Resume complete case %s',key)
        return report
    t0=time.perf_counter();rng=np.random.default_rng(seed+90000)
    # Disjoint seed streams; no transition-wise random split.
    sizes={'train':n,'validation':cfg['validation_episodes'],'calibration':cfg['calibration_episodes'],'id':cfg['test_episodes'],'ood':cfg['test_episodes']}
    data={k:generate(seed+1000*i+(cfg.get('test_seed_offset',0) if k in ['id','ood'] else 0),sizes[k],cfg['steps'],system,cfg['dt'],2.5 if k=='ood' else 1.,regime) for i,k in enumerate(sizes,1)}
    x,y=targets(data['train'],system,noise,rng,regime);xv,yv=targets(data['validation'],system,noise,rng,regime);xc,yc=targets(data['calibration'],system,noise,rng,regime)
    np.savez_compressed(case/'data.npz',**{f'{k}_{part}':d[j] for k,d in data.items() for j,part in enumerate(['states','actions'])},train_x=x,train_y=y,validation_x=xv,validation_y=yv,calibration_x=xc,calibration_y=yc)
    models={};training={}
    for mode in ['direct','residual']:
        if time.perf_counter()>deadline: raise TimeoutError('Four-hour total budget exceeded')
        models[mode],training[mode]=fit(x,y,xv,yv,system,mode,0,cfg,seed,device,case/f'progress-{mode}.pt',deadline)
    candidates=[]
    for lam in cfg['physics_weights']:
        if time.perf_counter()>deadline: raise TimeoutError('Four-hour total budget exceeded')
        m,info=fit(x,y,xv,yv,system,'direct',lam,cfg,seed,device,case/f'progress-physics-{lam}.pt',deadline)
        errs,_=rollout('direct',m,None,data['validation'],system,cfg['dt'],device,deadline=deadline)
        info['validation_rollout_mse']=float(errs.mean())
        candidates.append((info['validation_rollout_mse'],m,info))
    _,models['physics_tuned'],training['physics_tuned']=min(candidates,key=lambda t:t[0])
    cg,gg=gate(models['residual'],xc,yc,system,device)
    if cfg.get('equal_budget'):
        for mode in ['direct','residual']:
            name=mode+'_equal_budget'
            models[name],training[name]=fit(np.concatenate([x,xc]),np.concatenate([y,yc]),xv,yv,system,mode,0,cfg,seed,device,case/f'progress-{name}.pt',deadline)
    coef,ridge_info=(ridge_fit(np.concatenate([x,xc]),np.concatenate([y,yc]),xv,yv,system,data['validation'],cfg,device,deadline=deadline) if cfg.get('ridge') else (None,None))
    linear_coef,linear_info=(ridge_fit(np.concatenate([x,xc]),np.concatenate([y,yc]),xv,yv,system,data['validation'],cfg,device,linear=True,deadline=deadline) if cfg.get('ridge') else (None,None))
    methods=METHODS+(['ridge','linear_id'] if coef is not None else [])
    if cfg.get('equal_budget'):methods+=['direct_equal_budget','residual_equal_budget']
    metrics={};arrays={}
    for split in ['id','ood']:
        for method in methods:
            m=models.get(method,models['residual']);g=cg if method=='coordinate_gate' else gg
            errs,pred=rollout(method,m,g,data[split],system,cfg['dt'],device,linear_coef if method=='linear_id' else coef,deadline)
            arrays[f'{split}_{method}_errors']=errs;arrays[f'{split}_{method}_pred']=pred
            axis=np.mean((pred-data[split][0][:,1:])**2,axis=(0,1))
            metrics[f'{split}_{method}']={'mse':float(errs.mean()),'endpoint_mse':float(errs[:,-1].mean()),'axis_mse':axis.tolist(),'finite':bool(np.isfinite(pred).all()),'failed_episodes':int((~np.isfinite(pred).all(axis=(1,2))).sum())}
    np.savez_compressed(case/'rollouts.npz',**arrays)
    for name,m in models.items():
        torch.save({'state_dict':{k:v.cpu() for k,v in m.state_dict().items()},'mode':m.mode,'system':system,'hidden':cfg['hidden'],'seed':seed},case/f'{name}.pt')
    if coef is not None: np.save(case/'ridge.npy',coef)
    if linear_coef is not None:np.save(case/'linear_id.npy',linear_coef)
    report={'key':key,'system':system,'regime':regime,'n':n,'noise':noise,'seed':seed,'gates':{'coordinate':cg.tolist(),'global':gg},'metrics':metrics,'training':training,'physics_candidates':[c[2] for c in candidates],'ridge_info':ridge_info,'linear_info':linear_info,'data_hashes':{k:digest(v[0]) for k,v in data.items()},'seconds':time.perf_counter()-t0,'config':cfg,'code_hash':CODE_HASH,'artifact_hashes':{p.name:file_hash(p) for p in case.iterdir() if p.is_file() and not p.name.startswith('progress-') and p.name!='complete.json'}}
    atomic_json(case/'complete.json',report)
    logging.info('%s %.1fs gate=%s OOD residual=%.5f coord=%.5f',key,report['seconds'],cg,metrics['ood_residual']['mse'],metrics['ood_coordinate_gate']['mse'])
    return report

def summarize(reports,root):
    rows=[]
    for r in reports:
        for k,v in r['metrics'].items():
            split,method=k.split('_',1)
            rows.append({**{x:r[x] for x in ['system','regime','n','noise','seed']},'split':split,'method':method,**v})
    with (root/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups={}
    for row in rows:
        k=(row['system'],row['regime'],row['n'],row['noise'],row['split'],row['method'])
        groups.setdefault(k,[]).append(row['mse'] if row['mse'] is not None else float('nan'))
    summary=[dict(zip(['system','regime','n','noise','split','method'],k),mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)) if len(v)>1 else 0,seeds=len(v)) for k,v in groups.items()]
    atomic_json(root/'summary.json',summary)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')));parser.add_argument('--out',default='results');parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto');parser.add_argument('--threads',type=int,default=1);parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();cfg=json.loads(Path(a.config).read_text())
    if a.smoke: cfg.update(systems=['duffing'],regimes=['heterogeneous'],seeds=[11],train_episodes=[4],test_episodes=4,epochs=12,steps=20,noise=[0.3],calibration_episodes=4,validation_episodes=4)
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    existing=root/'config.json'
    if existing.exists() and json.loads(existing.read_text())!=cfg: raise ValueError('Output directory contains a different configuration; choose a new --out')
    existing.write_text(json.dumps(cfg,indent=2))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(root/'run.log')])
    torch.set_num_threads(a.threads);device=choose_device(a.device)
    torch.use_deterministic_algorithms(device.type=='cpu')
    env={'platform':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':str(device),'mps_available':torch.backends.mps.is_available(),'threads':a.threads,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    env['hardware_brand']=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() if platform.system()=='Darwin' else platform.processor()
    env['hardware_memory_bytes']=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)) if platform.system()=='Darwin' else None
    env['analysis_sha256']=file_hash(Path(__file__).with_name('analyze.py'))
    identity={k:env[k] for k in ['platform','machine','python','torch','numpy','device','threads','code_sha256','analysis_sha256']}
    if (root/'identity.json').exists() and json.loads((root/'identity.json').read_text())!=identity:raise ValueError('Environment/code changed; choose a new --out')
    (root/'environment.json').write_text(json.dumps(env,indent=2));logging.info('Environment %s',env)
    atomic_json(root/'identity.json',identity)
    history=json.loads((root/'run_history.json').read_text()) if (root/'run_history.json').exists() else []
    previous=sum(h['wall_seconds'] for h in history)
    t0=time.perf_counter();deadline=t0+cfg['budget_seconds']-previous;reports=[];complete=False;error=None
    try:
        if time.perf_counter()>deadline:raise TimeoutError('Cumulative budget exhausted')
        for system in cfg['systems']:
            for regime in cfg.get('regimes',['heterogeneous']):
                for n in cfg['train_episodes']:
                    for noise in cfg['noise']:
                        for seed in cfg['seeds']: reports.append(run_case(cfg,system,n,noise,seed,device,root,deadline,regime))
        summarize(reports,root)
        from analyze import analyze
        atomic_json(root/'analysis_summary.json',analyze(root))
        if time.perf_counter()>deadline:raise TimeoutError('Budget exceeded through analysis/export')
        complete=True
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';raise
    finally:
        elapsed=time.perf_counter()-t0
        history.append({'wall_seconds':elapsed,'complete':complete,'error':error});atomic_json(root/'run_history.json',history)
        status={'wall_seconds_this_invocation':elapsed,'cumulative_wall_seconds':previous+elapsed,'sum_case_seconds':sum(r['seconds'] for r in reports),'cases':len(reports),'within_4h_this_environment':previous+elapsed<14400,'mac_m3_max_measured':platform.system()=='Darwin' and 'M3 Max' in env['hardware_brand'] and env['hardware_memory_bytes']>=128*1024**3,'device':str(device),'complete':complete,'error':error,'scope':'generation, training, calibration, evaluation, statistics, plot export'}
        atomic_json(root/'runtime.json',status);logging.info('Completed %s',status)

if __name__=='__main__': main()
