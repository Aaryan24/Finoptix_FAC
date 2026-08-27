import warnings, json, numpy as np, pandas as pd, xgboost as xg
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
H=21; EMBARGO=25; MIN_ADV=5e7          # Rs 5 crore median daily traded value
raw=pd.read_pickle('raw500.pkl')
C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=800); V=V.reindex(columns=C.columns)
print("panel",C.shape,flush=True)
r1=C.pct_change()
adv=(C*V).rolling(63).median()
liquid=adv>MIN_ADV                      # dynamic, point-in-time universe
mkt=r1.mean(axis=1)                     # equal-weight market proxy

F={}
F['mom_12_1']=C.shift(21)/C.shift(252)-1
F['mom_6_1'] =C.shift(21)/C.shift(126)-1
F['rev_1m']  =C/C.shift(21)-1
F['rev_1w']  =C/C.shift(5)-1
F['vol_63']  =r1.rolling(63).std()
F['dn_vol']  =r1.clip(upper=0).rolling(63).std()
F['max_21']  =r1.rolling(21).max()
F['skew_63'] =r1.rolling(63).skew()
F['dist_hi'] =C/C.rolling(252).max()-1
F['c_ma200'] =C/C.rolling(200).mean()-1
F['turn']    =(C*V).rolling(21).mean()/(C*V).rolling(252).mean()
F['amihud']  =(r1.abs()/(C*V)).rolling(63).mean()*1e11
F['size']    =np.log(adv)
cov=r1.rolling(252).cov(mkt); var=mkt.rolling(252).var()
F['beta']    =cov.div(var,axis=0)
F['ivol']    =(r1.sub(F['beta'].mul(mkt,axis=0))).rolling(126).std()
FN=list(F)
print("features",len(FN),flush=True)

X={k:v.where(liquid).rank(axis=1,pct=True) for k,v in F.items()}
fwd=C.shift(-H)/C-1
fwd=fwd.where(liquid)
y=fwd.sub(fwd.mean(axis=1),axis=0)

blocks=[]
for d in C.index[252:]:
    b=pd.DataFrame({k:X[k].loc[d] for k in FN})
    b['y']=y.loc[d]; b['date']=d
    blocks.append(b.dropna())
P=pd.concat(blocks); P['tic']=P.index
print("panel rows",len(P),"mean names/day",int(P.groupby('date').size().mean()),flush=True)

def ic_stats(df,col='pred'):
    ic=df.groupby('date').apply(lambda g: g[col].corr(g['y'],method='spearman')).dropna()
    n=len(ic); m=ic.mean(); x=ic.values-m; g0=(x@x)/n; nw=g0
    for L in range(1,H+1): nw+=2*(1-L/(H+1))*((x[:-L]@x[L:])/n)
    non=[ic.iloc[o::H] for o in range(3)]
    return dict(ic=float(m), t_naive=float(m/ic.std()*np.sqrt(n)),
                t_nw=float(m/np.sqrt(nw/n)),
                t_nonoverlap=[round(float(s.mean()/s.std()*np.sqrt(len(s))),2) for s in non],
                n=int(n), ic_series=ic)

# ---- single-factor ICs (full sample, descriptive only) ----
print("\n=== single-factor IC (descriptive) ===",flush=True)
single={}
for k in FN:
    s=ic_stats(P.rename(columns={k:'pred'}),'pred') if False else None
for k in FN:
    tmp=P[['date','y',k]].rename(columns={k:'pred'})
    st=ic_stats(tmp); single[k]=st
    print(f"  {k:10} IC={st['ic']:+.4f}  t_nw={st['t_nw']:+6.2f}  nonoverlap_t={st['t_nonoverlap']}",flush=True)

# ---- walk-forward models ----
starts=pd.date_range('2018-01-01','2026-08-01',freq='6MS')
out={m:[] for m in ['composite','ridge','xgb']}
for s in starts:
    tr=P[P['date']<=s-pd.Timedelta(days=EMBARGO)]
    te=P[(P['date']>=s)&(P['date']<s+pd.DateOffset(months=6))]
    if len(tr)<50000 or len(te)==0: continue
    # composite: sign each factor by TRAIN-ONLY IC, then average ranks
    signs={}
    for k in FN:
        c=tr[k].corr(tr['y'],method='spearman')
        signs[k]=np.sign(c) if np.isfinite(c) else 0.0
    comp=sum(signs[k]*(te[k]-0.5) for k in FN)/len(FN)
    o=te[['date','y']].copy(); o['pred']=comp.values; out['composite'].append(o)
    rg=Ridge(alpha=50.0).fit(tr[FN],tr['y'])
    o=te[['date','y']].copy(); o['pred']=rg.predict(te[FN]); out['ridge'].append(o)
    gb=xg.XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.7,min_child_weight=100,reg_lambda=5.0,random_state=0,n_jobs=4)
    gb.fit(tr[FN],tr['y'])
    o=te[['date','y']].copy(); o['pred']=gb.predict(te[FN]); out['xgb'].append(o)
    print(f"  {s.date()} train={len(tr)} test={len(te)}",flush=True)

res={}
print("\n=== walk-forward model IC ===",flush=True)
for m,v in out.items():
    if not v: continue
    df=pd.concat(v); df.to_pickle(f'v3_{m}.pkl')
    st=ic_stats(df); ser=st.pop('ic_series'); res[m]=st
    print(f"  {m:10} IC={st['ic']:+.4f}  t_naive={st['t_naive']:+.2f}  t_nw={st['t_nw']:+.2f}  nonoverlap={st['t_nonoverlap']}  n={st['n']}",flush=True)
json.dump({'single':{k:{kk:vv for kk,vv in v.items() if kk!='ic_series'} for k,v in single.items()},
           'models':res}, open('v3_results.json','w'), indent=2, default=str)
print("DONE",flush=True)
