"""Convex trajectory calibration. Arrays: [..., D], output A: [D,D]."""
import numpy as np

def scalar(r,e):
    den=np.sum(r*r)
    return float(np.clip(np.sum(r*e)/den,0,1)) if den>0 else 0.

def project(a):
    values,vectors=np.linalg.eigh((a+a.T)/2)
    return (vectors*np.clip(values,0,1))@vectors.T

def fit(r,e,lam=0.,kind='matrix',max_iter=1000,tol=1e-10):
    r=np.asarray(r,dtype=np.float64).reshape(-1,r.shape[-1])
    e=np.asarray(e,dtype=np.float64).reshape(r.shape)
    d=r.shape[1];alpha=scalar(r,e)
    if kind=='scalar':return alpha*np.eye(d),{'iterations':0,'alpha':alpha}
    if kind=='diagonal':
        den=(r*r).mean(0)+lam
        a=np.divide((r*e).mean(0)+lam*alpha,den,out=np.zeros(d),where=den>0)
        return np.diag(np.clip(a,0,1)),{'iterations':0,'alpha':alpha}
    gram=r.T@r/len(r);cross=r.T@e/len(r)
    lipschitz=2*(np.linalg.eigvalsh(gram)[-1]+lam)
    a=alpha*np.eye(d)
    if lipschitz==0:return a,{'iterations':0,'alpha':alpha}
    for k in range(max_iter):
        gradient=2*(gram@a-cross)+2*lam*(a-alpha*np.eye(d))
        nxt=project(a-gradient/lipschitz)
        delta=np.linalg.norm(nxt-a);a=nxt
        if delta<tol:break
    grad=2*(gram@a-cross)+2*lam*(a-alpha*np.eye(d))
    stationarity=np.linalg.norm(a-project(a-grad/lipschitz))*lipschitz
    return a,{'iterations':k+1,'alpha':alpha,'projected_gradient_residual':float(stationarity)}

def select(r,e,rv,ev,kind='matrix',lambdas=(0.,.001,.01,.1,1.,10.)):
    candidates=[]
    for lam in lambdas:
        a,info=fit(r,e,lam,kind)
        score=float(np.mean((rv@a-ev)**2))
        candidates.append((score,lam,a,info))
    score,lam,a,info=min(candidates,key=lambda x:x[0])
    return a,dict(info,lambda_=lam,selection_mse=score,candidates=[{'lambda':v[1],'mse':v[0]} for v in candidates])
