"""Full 3-stage architecture: transformer+momentum picks -> MVO max-Sharpe with L2 -> attribution."""
import warnings, json, numpy as np, pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
XA=pd.read_pickle('xattn_pred.pkl'); XA['date']=pd.to_datetime(XA['date'])
raw=pd.read_pickle('raw500.pkl'); C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=800); V=V.reindex(columns=C.columns)
r1=C.pct_change(); adv=(C*V).rolling(63).median(); liq=adv>5e7
mom=(C.shift(21)/C.shift(252)-1).where(liq)
RF=0.06/252

def opt(mu,cov,cap=1.0,gamma=0.0):
    n=len(mu)
    def neg(w):
        v=np.sqrt(max(w@cov@w,1e-12)); return -(w@mu)/v + gamma*(w@w)
    c=({'type':'eq','fun':lambda w:w.sum()-1},)
    r=minimize(neg,np.repeat(1/n,n),bounds=[(0,cap)]*n,constraints=c,method='SLSQP',options={'maxiter':300})
    return r.x if r.success else np.repeat(1/n,n)

sel={}
for d,g in XA.groupby('date'):
    if d not in C.index: continue
    g=g[g['tic'].isin(C.columns)]
    m=mom.loc[d].reindex(g['tic']).values
    g=g.assign(mom=m).dropna(subset=['mom'])
    if len(g)<80: continue
    g=g.assign(bl=(g['pred'].rank(pct=True)+g['mom'].rank(pct=True))/2)
    sel[d]={'blend':list(g.nlargest(30,'bl')['tic']), 'mom':list(g.nlargest(30,'mom')['tic'])}
dates=sorted(sel); days=r1.loc[dates[0]:dates[-1]+pd.DateOffset(months=1)].index

def bt(pick,weight,gamma=0.0,cap=1.0,cost=20):
    cur={}; out=[]; eff=[]
    for d in days:
        if d in sel:
            p=sel[d][pick]
            h=r1.loc[r1.index<d,p].tail(252).dropna(axis=1); p=list(h.columns)
            if len(p)>2:
                if weight=='equal': w=np.repeat(1/len(p),len(p))
                else: w=opt(h.mean().values*252,h.cov().values*252,cap,gamma)
                eff.append(1/np.sum(w**2))
                tgt=dict(zip(p,w))
                to=sum(abs(tgt.get(t,0)-cur.get(t,0)) for t in set(tgt)|set(cur))
                cur=tgt; out.append((d,-to*cost/1e4)); continue
        if cur:
            rr=sum(w*r1.loc[d,t] for t,w in cur.items() if pd.notna(r1.loc[d,t]))
            nv={t:w*(1+r1.loc[d,t]) for t,w in cur.items() if pd.notna(r1.loc[d,t])}
            g=sum(nv.values())
            if g>0: cur={t:v/g for t,v in nv.items()}
            out.append((d,float(rr)))
    s=pd.Series(dict(out)); n=len(s); a=(1+s).prod()**(252/n)-1; v=s.std()*np.sqrt(252)
    cum=(1+s).cumprod()
    return s, dict(ann=a*100,vol=v*100,sharpe=(a-0.06)/v,dd=float((cum/cum.cummax()-1).min())*100,
                   effN=float(np.mean(eff)))
cfg=[('blend','equal',0,1.0,'blend picks + equal weight'),
     ('blend','mvo',0.0,1.0,'blend picks + MVO (unconstrained)'),
     ('blend','mvo',2.0,1.0,'blend picks + MVO + L2 g=2'),
     ('blend','mvo',5.0,1.0,'blend picks + MVO + L2 g=5'),
     ('blend','mvo',2.0,0.15,'blend picks + MVO + L2 + cap15%'),
     ('mom','mvo',2.0,1.0,'momentum picks + MVO + L2 g=2')]
print(f"NIFTY500, {dates[0].date()} to {dates[-1].date()}, monthly, net 20bps\n")
print(f"{'configuration':38}{'ann%':>8}{'vol%':>8}{'sharpe':>8}{'maxDD%':>9}{'effN':>7}")
S={}
for pick,wt,g,cap,lab in cfg:
    s,r=bt(pick,wt,g,cap); S[lab]=s
    print(f"{lab:38}{r['ann']:8.2f}{r['vol']:8.2f}{r['sharpe']:8.2f}{r['dd']:9.1f}{r['effN']:7.1f}",flush=True)
# Fama-French / Carhart attribution on the winner
best=max(S,key=lambda k:(1+S[k]).prod())
sz=np.log(adv).where(liq).rank(axis=1,pct=True); mo=mom.rank(axis=1,pct=True)
def leg(mask):
    w=mask.div(mask.sum(axis=1),axis=0); return (r1*w.shift(1)).sum(axis=1)
MKT=r1.where(liq).mean(axis=1)-RF
SMB=leg((sz<=0.3)&liq)-leg((sz>=0.7)&liq)
WML=leg((mo>=0.7)&liq)-leg((mo<=0.3)&liq)
F=pd.DataFrame({'MKT':MKT,'SMB':SMB,'WML':WML}).dropna()
import numpy.linalg as la
print(f"\n=== Fama-French/Carhart attribution: {best} ===")
y=(S[best]-RF).reindex(F.index).dropna(); Fx=F.loc[y.index]
for spec in [['MKT','SMB'],['MKT','SMB','WML']]:
    X=np.column_stack([np.ones(len(y))]+[Fx[c].values for c in spec])
    b,*_=la.lstsq(X,y.values,rcond=None)
    res=y.values-X@b; s2=res@res/(len(y)-X.shape[1]); se=np.sqrt(np.diag(s2*la.inv(X.T@X)))
    print(f"  ~ {'+'.join(spec):15} alpha={b[0]*252*100:+6.2f}%/yr (t={b[0]/se[0]:+5.2f})  "
          +"  ".join(f"{c}={bb:+.2f}" for c,bb in zip(spec,b[1:])))
print("DONE")
