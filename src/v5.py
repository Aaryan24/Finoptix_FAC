import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
raw=pd.read_pickle('raw500.pkl'); C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=800); V=V.reindex(columns=C.columns)
r1=C.pct_change(); adv=(C*V).rolling(63).median(); liq=adv>5e7
mom=(C.shift(21)/C.shift(252)-1).where(liq)
days=C.loc['2016-01-01':].index
months=set(pd.Series(days).groupby([days.year,days.month]).first().values)

def run(rank_fn,n=30,cost=0,seed=None):
    rng=np.random.default_rng(seed); cur={}; out=[]
    for d in days:
        if np.datetime64(d) in months:
            s=mom.loc[d].dropna()
            if len(s)>=90:
                pick=rank_fn(s,rng,n)
                tgt={t:1/len(pick) for t in pick}
                to=sum(abs(tgt.get(t,0)-cur.get(t,0)) for t in set(tgt)|set(cur))
                cur=tgt; out.append((d,-to*cost/1e4)); continue
        if cur:
            rr=sum(w*r1.loc[d,t] for t,w in cur.items() if pd.notna(r1.loc[d,t]))
            nv={t:w*(1+r1.loc[d,t]) for t,w in cur.items() if pd.notna(r1.loc[d,t])}
            g=sum(nv.values())
            if g>0: cur={t:v/g for t,v in nv.items()}
            out.append((d,float(rr)))
    return pd.Series(dict(out))

def ann(s): return ((1+s).prod()**(252/len(s))-1)*100
def shp(s,rf=6): return (ann(s)-rf)/(s.std()*np.sqrt(252)*100)

top=run(lambda s,rng,n: list(s.nlargest(n).index))
print(f"momentum top-30 : ann={ann(top):6.2f}%  sharpe={shp(top):.2f}")
print("\n=== placebo: 200 RANDOM 30-name portfolios from the same liquid universe ===",flush=True)
ra=[];rs=[]
for i in range(120):
    r=run(lambda s,rng,n: list(rng.choice(s.index,size=n,replace=False)), seed=i)
    ra.append(ann(r)); rs.append(shp(r))
ra=np.array(ra); rs=np.array(rs)
print(f"  random ann%   mean={ra.mean():6.2f}  sd={ra.std():5.2f}  p95={np.percentile(ra,95):6.2f}  max={ra.max():6.2f}")
print(f"  random sharpe mean={rs.mean():6.2f}  sd={rs.std():5.2f}  p95={np.percentile(rs,95):6.2f}  max={rs.max():6.2f}")
print(f"  momentum percentile vs random: ann={100*(ra<ann(top)).mean():.1f}%   sharpe={100*(rs<shp(top)).mean():.1f}%")
print(f"  z-score: ann={(ann(top)-ra.mean())/ra.std():+.2f}   sharpe={(shp(top)-rs.mean())/rs.std():+.2f}")
print("\n=== sub-period stability (momentum top-30, net) ===")
for a,b in [('2016-01-01','2021-01-01'),('2021-01-01','2026-08-24')]:
    s=top.loc[a:b]; print(f"  {a[:4]}-{b[:4]}: ann={ann(s):6.2f}%  sharpe={shp(s):5.2f}")
print("DONE")
