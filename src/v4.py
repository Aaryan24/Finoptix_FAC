import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
raw=pd.read_pickle('raw500.pkl'); C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=800); V=V.reindex(columns=C.columns)
r1=C.pct_change(); adv=(C*V).rolling(63).median(); liq=adv>5e7
mom=(C.shift(21)/C.shift(252)-1).where(liq)
size=np.log(adv).where(liq)
bench=pd.read_pickle('bench2.pkl').squeeze()

def run(score, n_hold=30, cost=20, short=False):
    days=C.loc['2016-01-01':].index
    months=set(pd.Series(days).groupby([days.year,days.month]).first().values)
    cur={}; out=[]
    for d in days:
        if np.datetime64(d) in months:
            s=score.loc[d].dropna()
            if len(s)>=3*n_hold:
                lo=list(s.nlargest(n_hold).index)
                tgt={t:1/n_hold for t in lo}
                if short:
                    sh=list(s.nsmallest(n_hold).index)
                    tgt={t:0.5/n_hold for t in lo}; tgt.update({t:-0.5/n_hold for t in sh})
                to=sum(abs(tgt.get(t,0)-cur.get(t,0)) for t in set(tgt)|set(cur))
                cur=tgt; out.append((d,-to*cost/1e4)); continue
        if cur:
            rr=sum(w*r1.loc[d,t] for t,w in cur.items() if pd.notna(r1.loc[d,t]))
            nv={t:w*(1+r1.loc[d,t]) for t,w in cur.items() if pd.notna(r1.loc[d,t])}
            g=sum(abs(v) for v in nv.values())
            if g>0: cur={t:v/g for t,v in nv.items()}
            out.append((d,float(rr)))
    return pd.Series(dict(out))

def stat(s,rf=0.06):
    n=len(s); ann=(1+s).prod()**(252/n)-1; vol=s.std()*np.sqrt(252)
    cum=(1+s).cumprod(); dd=float((cum/cum.cummax()-1).min())
    return dict(ann=ann*100, vol=vol*100, sharpe=(ann-rf)/vol, dd=dd*100,
                total=float((1+s).prod()-1)*100, yrs=round(n/252,1))

R={}
R['momentum_top30']   = stat(run(mom,30))
R['momentum_top50']   = stat(run(mom,50))
R['mom_longshort']    = stat(run(mom,30,short=True))
comb=mom.rank(axis=1,pct=True)-size.rank(axis=1,pct=True)   # momentum + small size
R['mom_plus_size']    = stat(run(comb,30))
ew=r1.where(liq).mean(axis=1).loc['2016-01-01':].dropna()
R['equal_weight_univ']= stat(ew)
b=bench.pct_change().dropna(); b=b.loc['2016-01-01':]
R['NIFTY50']          = stat(b)
print(f"{'strategy':22}{'ann%':>8}{'vol%':>8}{'sharpe':>8}{'maxDD%':>9}{'total%':>10}{'yrs':>6}")
for k,v in R.items():
    print(f"{k:22}{v['ann']:8.2f}{v['vol']:8.2f}{v['sharpe']:8.2f}{v['dd']:9.1f}{v['total']:10.1f}{v['yrs']:6.1f}")
json.dump(R,open('v4_results.json','w'),indent=2)
# yearly for the headline strategy
s=run(mom,30); yr=(1+s).groupby(s.index.year).prod()-1
print("\nyearly (momentum top-30, net):"); print("  "+"  ".join(f"{y}:{v*100:+.1f}%" for y,v in yr.items()))
print("DONE")
