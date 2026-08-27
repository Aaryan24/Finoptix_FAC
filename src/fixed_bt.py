"""Corrected backtest: rebalance-day returns retained; true Sharpe (arithmetic excess)."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
raw=pd.read_pickle('raw500.pkl'); C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=800); V=V.reindex(columns=C.columns)
r1=C.pct_change(); adv=(C*V).rolling(63).median(); liq=adv>5e7
mom=(C.shift(21)/C.shift(252)-1).where(liq)
RF=0.06; WIN=('2021-01-01','2026-01-01')
def load(f):
    R=pd.read_pickle(f); R['date']=pd.to_datetime(R['date'])
    R=R[(R['date']>=WIN[0])&(R['date']<=WIN[1])]
    dd=pd.DatetimeIndex(sorted(R['date'].unique()))
    if len(dd)>200:
        keep={pd.Timestamp(x) for x in pd.Series(dd).groupby([dd.year,dd.month]).first().values}
        R=R[R['date'].isin(keep)]
    return R
def bt(sel,cost=20):
    """weights set at close of d; day d's return accrues to the PREVIOUS book, then rebalance."""
    dts=sorted(sel); days=r1.loc[dts[0]:dts[-1]+pd.DateOffset(months=1)].index
    cur={}; out=[]
    for d in days:
        rr=0.0
        if cur:   # previous book earns today's return
            rr=sum(w*r1.loc[d,t] for t,w in cur.items() if pd.notna(r1.loc[d,t]))
            nv={t:w*(1+r1.loc[d,t]) for t,w in cur.items() if pd.notna(r1.loc[d,t])}
            g=sum(nv.values())
            if g>0: cur={t:v/g for t,v in nv.items()}
        if d in sel:   # then rebalance at the close, paying costs
            p=sel[d]; tgt={t:1/len(p) for t in p}
            to=sum(abs(tgt.get(t,0)-cur.get(t,0)) for t in set(tgt)|set(cur))
            rr-= to*cost/1e4
            cur=tgt
        out.append((d,float(rr)))
    return pd.Series(dict(out))
def stats(s):
    n=len(s)
    cagr=(1+s).prod()**(252/n)-1
    arith=s.mean()*252                      # arithmetic annualised
    vol=s.std()*np.sqrt(252)
    cum=(1+s).cumprod()
    d={'CAGR':cagr*100,'Sharpe':(arith-RF)/vol,'vol':vol*100,
       'maxDD':float((cum/cum.cummax()-1).min())*100}
    for y in sorted(set(s.index.year)):
        sy=s[s.index.year==y]; d[y]=((1+sy).prod()-1)*100 if len(sy)>5 else np.nan
    return d
XA=load('xattn_pred.pkl'); RG=load('v3fix_ridge.pkl')
common=sorted(set(XA['date'])&set(RG['date']))
sig={}
for d in common:
    a=XA[XA['date']==d][['tic','pred']].rename(columns={'pred':'xa'})
    b=RG[RG['date']==d][['tic','pred']].rename(columns={'pred':'rg'})
    g=a.merge(b,on='tic',how='inner'); g=g[g['tic'].isin(C.columns)]
    g['mo']=mom.loc[d].reindex(g['tic']).values; g=g.dropna()
    if len(g)<80: continue
    for c in ['xa','rg','mo']: g['r_'+c]=g[c].rank(pct=True)
    sig[d]=g
COMBOS={'ridge+transformer':['r_rg','r_xa'],'transformer+momentum (blend)':['r_xa','r_mo'],
        'all three':['r_xa','r_mo','r_rg'],'transformer':['r_xa'],'ridge':['r_rg'],'momentum':['r_mo']}
rows=[]
for k,cs in COMBOS.items():
    sel={d:list(g.assign(s=g[cs].mean(axis=1)).nlargest(30,'s')['tic']) for d,g in sig.items()}
    st=stats(bt(sel)); st['strategy']=k; rows.append(st)
days=r1.loc[min(sig):max(sig)+pd.DateOffset(months=1)].index
months=set(pd.Series(days).groupby([days.year,days.month]).first().values)
bsel={pd.Timestamp(d):[t for t in C.columns if liq.loc[d,t]] for d in days if np.datetime64(d) in months}
st=stats(bt(bsel)); st['strategy']='equal-weight benchmark'; rows.append(st)
T=pd.DataFrame(rows).set_index('strategy')
yrs=[c for c in T.columns if isinstance(c,int)]
T=T[['CAGR','Sharpe','vol','maxDD']+yrs].sort_values('Sharpe',ascending=False)
T.to_csv('fixed_table.csv'); print(T.round(2).to_string())
