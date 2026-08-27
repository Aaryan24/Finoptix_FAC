"""Backtest every model's top-30 picks -> CAGR / Sharpe / maxDD / annual returns."""
import warnings, os, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
raw=pd.read_pickle('raw500.pkl'); C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=800); V=V.reindex(columns=C.columns)
r1=C.pct_change(); adv=(C*V).rolling(63).median(); liq=adv>5e7
mom=(C.shift(21)/C.shift(252)-1).where(liq)

WIN=('2021-01-01','2026-01-01')
def picks_from(pkl, blend_with_mom=False):
    R=pd.read_pickle(pkl); R['date']=pd.to_datetime(R['date'])
    if 'tic' not in R.columns: return None
    R=R[(R['date']>=WIN[0])&(R['date']<=WIN[1])]
    # daily prediction files (ridge/xgb) -> keep first trading day of each month only
    dd=sorted(R['date'].unique())
    if len(dd)>200:
        keep=set(pd.Series(pd.DatetimeIndex(dd)).groupby([pd.DatetimeIndex(dd).year,pd.DatetimeIndex(dd).month]).first().values)
        R=R[R['date'].isin([pd.Timestamp(x) for x in keep])]
    sel={}
    for d,g in R.groupby('date'):
        if d not in C.index: continue
        g=g[g['tic'].isin(C.columns)].copy()
        g['mom']=mom.loc[d].reindex(g['tic']).values
        g=g.dropna(subset=['mom'])
        if len(g)<80: continue
        if blend_with_mom:
            g['s']=(g['pred'].rank(pct=True)+g['mom'].rank(pct=True))/2
        else:
            g['s']=g['pred']
        sel[d]=list(g.nlargest(30,'s')['tic'])
    return sel
def mom_only(dates):
    sel={}
    for d in dates:
        s=mom.loc[d].dropna()
        if len(s)>=80: sel[d]=list(s.nlargest(30).index)
    return sel
def bt(sel,cost=20):
    dates=sorted(sel); days=r1.loc[dates[0]:dates[-1]+pd.DateOffset(months=1)].index
    cur={};out=[]
    for d in days:
        if d in sel:
            p=sel[d]; tgt={t:1/len(p) for t in p}
            to=sum(abs(tgt.get(t,0)-cur.get(t,0)) for t in set(tgt)|set(cur))
            cur=tgt; out.append((d,-to*cost/1e4)); continue
        if cur:
            rr=sum(w*r1.loc[d,t] for t,w in cur.items() if pd.notna(r1.loc[d,t]))
            nv={t:w*(1+r1.loc[d,t]) for t,w in cur.items() if pd.notna(r1.loc[d,t])}
            g=sum(nv.values())
            if g>0: cur={t:v/g for t,v in nv.items()}
            out.append((d,float(rr)))
    return pd.Series(dict(out))
def bench(days,cost=20):
    months=set(pd.Series(days).groupby([days.year,days.month]).first().values)
    cur={};out=[]
    for d in days:
        if np.datetime64(d) in months:
            p=[t for t in C.columns if liq.loc[d,t]]; tgt={t:1/len(p) for t in p}
            to=sum(abs(tgt.get(t,0)-cur.get(t,0)) for t in set(tgt)|set(cur))
            cur=tgt; out.append((d,-to*cost/1e4)); continue
        if cur:
            rr=sum(w*r1.loc[d,t] for t,w in cur.items() if pd.notna(r1.loc[d,t]))
            nv={t:w*(1+r1.loc[d,t]) for t,w in cur.items() if pd.notna(r1.loc[d,t])}
            g=sum(nv.values())
            if g>0: cur={t:v/g for t,v in nv.items()}
            out.append((d,float(rr)))
    return pd.Series(dict(out))
def stats(s):
    n=len(s); a=(1+s).prod()**(252/n)-1; v=s.std()*np.sqrt(252); cum=(1+s).cumprod()
    d={'CAGR':a*100,'Sharpe':(a-0.06)/v,'maxDD':float((cum/cum.cummax()-1).min())*100}
    for y in sorted(set(s.index.year)):
        sy=s[s.index.year==y]; d[y]=((1+sy).prod()-1)*100 if len(sy)>5 else np.nan
    return d
JOBS=[('Blend (transformer + momentum)','xattn_pred.pkl',True),
      ('Cross-sectional transformer','xattn_pred.pkl',False),
      ('LSTM (sequences + factors)','nn_pred_tic.pkl',False),
      ('MLP (factors only)','nn_statonly_tic.pkl',False),
      ('LSTM (sequences only)','nn_seqonly_tic.pkl',False),
      ('Ridge','v3_ridge.pkl',False),
      ('XGBoost','v3_xgb.pkl',False)]
rows=[]
for lab,f,bl in JOBS:
    if not os.path.exists(f): print("skip (missing):",lab); continue
    sel=picks_from(f,bl)
    if not sel: print("skip (no tickers):",lab); continue
    d=stats(bt(sel)); d['model']=lab; rows.append(d)
    print(f"  done {lab}",flush=True)
allsel=picks_from('xattn_pred.pkl',True)
d=stats(bt(mom_only(sorted(allsel)))); d['model']='12-1 momentum'; rows.append(d)
days=r1.loc[sorted(allsel)[0]:sorted(allsel)[-1]+pd.DateOffset(months=1)].index
d=stats(bench(days)); d['model']='Equal-weight NIFTY500 (benchmark)'; rows.append(d)
b=pd.read_pickle('bench2.pkl').squeeze().pct_change().loc[days[0]:days[-1]].dropna()
d=stats(b); d['model']='NIFTY 50'; rows.append(d)
T=pd.DataFrame(rows).set_index('model')
yrs=[c for c in T.columns if isinstance(c,int)]
T=T[['CAGR','Sharpe','maxDD']+yrs].sort_values('CAGR',ascending=False)
T.to_csv('perf_table.csv')
print("\n"+T.round(1).to_string())
