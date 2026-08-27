import warnings, json, numpy as np, pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
raw=pd.read_pickle('raw500.pkl'); C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=800); V=V.reindex(columns=C.columns)
r1=C.pct_change(); adv=(C*V).rolling(63).median(); liq=adv>5e7
mom=(C.shift(21)/C.shift(252)-1).where(liq); size=np.log(adv).where(liq)
days=C.loc['2016-01-01':].index
months=set(pd.Series(days).groupby([days.year,days.month]).first().values)
RF=0.06/252

def maxsharpe(mu,cov,cap):
    n=len(mu)
    def neg(w):
        v=np.sqrt(max(w@cov@w,1e-12)); return -(w@mu)/v
    c=({'type':'eq','fun':lambda w:w.sum()-1},)
    r=minimize(neg,np.repeat(1/n,n),bounds=[(0,cap)]*n,constraints=c,method='SLSQP',
               options={'maxiter':200})
    return r.x if r.success else np.repeat(1/n,n)

def black_litterman(cov,w_mkt,q,delta=2.5,tau=0.05):
    """pi = market-implied equilibrium returns; views Q = momentum-implied returns."""
    pi=delta*cov@w_mkt
    P=np.eye(len(q)); Om=np.diag(np.diag(tau*P@cov@P.T))+1e-10*np.eye(len(q))
    A=np.linalg.inv(tau*cov); B=np.linalg.inv(Om)
    M=np.linalg.inv(A+P.T@B@P)
    return M@(A@pi+P.T@B@q)

def run(mode,n=30,cost=20,cap=0.15):
    cur={}; out=[]; turn=[]
    for d in days:
        if np.datetime64(d) in months:
            s=mom.loc[d].dropna()
            if len(s)>=90:
                pick=list(s.nlargest(n).index)
                h=r1.loc[r1.index<d,pick].tail(252).dropna(axis=1)
                pick=list(h.columns)
                if len(pick)>2:
                    if mode=='equal':
                        w=np.repeat(1/len(pick),len(pick))
                    else:
                        cov=h.cov().values*252
                        a=adv.loc[d,pick].values; wm=a/a.sum()          # cap-weight proxy
                        z=s[pick].values; z=(z-z.mean())/(z.std()+1e-9)
                        q=0.02*z
                        if mode=='mvo':
                            mu=h.mean().values*252
                        else:                                            # 'bl'
                            mu=black_litterman(cov,wm,q)
                        w=maxsharpe(mu,cov,cap)
                    tgt=dict(zip(pick,w))
                    to=sum(abs(tgt.get(t,0)-cur.get(t,0)) for t in set(tgt)|set(cur))
                    turn.append(to); cur=tgt
                    out.append((d,-to*cost/1e4)); continue
        if cur:
            rr=sum(w*r1.loc[d,t] for t,w in cur.items() if pd.notna(r1.loc[d,t]))
            nv={t:w*(1+r1.loc[d,t]) for t,w in cur.items() if pd.notna(r1.loc[d,t])}
            g=sum(nv.values())
            if g>0: cur={t:v/g for t,v in nv.items()}
            out.append((d,float(rr)))
    return pd.Series(dict(out)), float(np.mean(turn))

def stat(s):
    n=len(s); a=(1+s).prod()**(252/n)-1; v=s.std()*np.sqrt(252)
    cum=(1+s).cumprod()
    return dict(ann=a*100,vol=v*100,sharpe=(a-0.06)/v,dd=float((cum/cum.cummax()-1).min())*100)

res={}
for m in ['equal','mvo','bl']:
    s,t=run(m); res[m]=stat(s); res[m]['turnover']=round(t,2); res[m]['_s']=s
    print(f"  {m:6} ann={res[m]['ann']:6.2f}%  vol={res[m]['vol']:5.2f}%  sharpe={res[m]['sharpe']:5.2f}  dd={res[m]['dd']:6.1f}%  turn={t:.2f}",flush=True)

# ---------- Fama-French / Carhart attribution ----------
print("\n=== Fama-French + Carhart factor attribution ===",flush=True)
sz=size.rank(axis=1,pct=True); mo=mom.rank(axis=1,pct=True)
def leg(mask): 
    w=mask.div(mask.sum(axis=1),axis=0); return (r1*w.shift(1)).sum(axis=1)
MKT=r1.where(liq).mean(axis=1)-RF
SMB=leg((sz<=0.3)&liq)-leg((sz>=0.7)&liq)
WML=leg((mo>=0.7)&liq)-leg((mo<=0.3)&liq)
F=pd.DataFrame({'MKT':MKT,'SMB':SMB,'WML':WML}).loc['2016-01-01':].dropna()
import numpy.linalg as la
for name in ['equal','bl']:
    y=(res[name]['_s']-RF).reindex(F.index).dropna(); Fx=F.loc[y.index]
    for spec in [['MKT','SMB'],['MKT','SMB','WML']]:
        X=np.column_stack([np.ones(len(y))]+[Fx[c].values for c in spec])
        b,*_=la.lstsq(X,y.values,rcond=None)
        resid=y.values-X@b; s2=resid@resid/(len(y)-X.shape[1])
        se=np.sqrt(np.diag(s2*la.inv(X.T@X)))
        al=b[0]*252*100; t=b[0]/se[0]
        print(f"  {name:6} ~ {'+'.join(spec):15}  alpha={al:+6.2f}%/yr (t={t:+5.2f})  "
              +"  ".join(f"{c}={bb:+.2f}" for c,bb in zip(spec,b[1:])),flush=True)
json.dump({k:{kk:vv for kk,vv in v.items() if kk!='_s'} for k,v in res.items()},open('v6.json','w'),indent=2)
print("DONE")
