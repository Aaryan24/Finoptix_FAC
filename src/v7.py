import warnings, json, numpy as np, pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
raw=pd.read_pickle('raw500.pkl'); C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=800); V=V.reindex(columns=C.columns)
r1=C.pct_change(); adv=(C*V).rolling(63).median(); liq=adv>5e7
mom=(C.shift(21)/C.shift(252)-1).where(liq)
days=C.loc['2016-01-01':].index
months=set(pd.Series(days).groupby([days.year,days.month]).first().values)

def opt(mu,cov,cap=1.0,gamma=0.0):
    n=len(mu)
    def neg(w):
        v=np.sqrt(max(w@cov@w,1e-12))
        return -(w@mu)/v + gamma*(w@w)          # L2 penalty pushes toward equal weight
    c=({'type':'eq','fun':lambda w:w.sum()-1},)
    r=minimize(neg,np.repeat(1/n,n),bounds=[(0,cap)]*n,constraints=c,method='SLSQP',options={'maxiter':300})
    return r.x if r.success else np.repeat(1/n,n)

def run(cap,gamma,n=30,cost=20):
    cur={}; out=[]; eff=[]
    for d in days:
        if np.datetime64(d) in months:
            s=mom.loc[d].dropna()
            if len(s)>=90:
                pick=list(s.nlargest(n).index)
                h=r1.loc[r1.index<d,pick].tail(252).dropna(axis=1); pick=list(h.columns)
                if len(pick)>2:
                    w=opt(h.mean().values*252,h.cov().values*252,cap,gamma)
                    eff.append(1/np.sum(w**2))
                    tgt=dict(zip(pick,w))
                    to=sum(abs(tgt.get(t,0)-cur.get(t,0)) for t in set(tgt)|set(cur))
                    cur=tgt; out.append((d,-to*cost/1e4)); continue
        if cur:
            rr=sum(w*r1.loc[d,t] for t,w in cur.items() if pd.notna(r1.loc[d,t]))
            nv={t:w*(1+r1.loc[d,t]) for t,w in cur.items() if pd.notna(r1.loc[d,t])}
            g=sum(nv.values())
            if g>0: cur={t:v/g for t,v in nv.items()}
            out.append((d,float(rr)))
    s=pd.Series(dict(out)); nn=len(s); a=(1+s).prod()**(252/nn)-1; v=s.std()*np.sqrt(252)
    cum=(1+s).cumprod()
    return dict(ann=a*100,vol=v*100,sharpe=(a-0.06)/v,dd=float((cum/cum.cummax()-1).min())*100,effN=float(np.mean(eff)))

cfg=[("no constraint (original bug)",1.0,0.0),
     ("cap 15% only",0.15,0.0),
     ("L2 gamma=0.5 only",1.0,0.5),
     ("L2 gamma=2 only",1.0,2.0),
     ("L2 gamma=5 only",1.0,5.0),
     ("cap 15% + L2 gamma=2",0.15,2.0)]
print(f"{'method':30}{'ann%':>8}{'vol%':>8}{'sharpe':>8}{'maxDD%':>9}{'effN':>7}")
R={}
for lab,cap,g in cfg:
    r=run(cap,g); R[lab]=r
    print(f"{lab:30}{r['ann']:8.2f}{r['vol']:8.2f}{r['sharpe']:8.2f}{r['dd']:9.1f}{r['effN']:7.1f}",flush=True)
json.dump(R,open('v7.json','w'),indent=2)
print("DONE")
