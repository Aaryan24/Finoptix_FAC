import warnings, json, numpy as np, pandas as pd, yfinance as yf, xgboost as xg
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
TICKERS=['BHARTIARTL','HDFCLIFE','NTPC','MARUTI','NESTLEIND','BAJFINANCE','KOTAKBANK','TATASTEEL','ONGC',
 'BAJAJ-AUTO','LT','ITC','TCS','BRITANNIA','ADANIENT','CIPLA','WIPRO','INDUSINDBK','ULTRACEMCO','TATACONSUM',
 'BAJAJFINSV','RELIANCE','HEROMOTOCO','COALINDIA','TITAN','HINDALCO','APOLLOHOSP','TECHM','DRREDDY','DIVISLAB',
 'EICHERMOT','BPCL','SBILIFE','GRASIM','JSWSTEEL','ASIANPAINT','POWERGRID','ADANIPORTS','M&M','SUNPHARMA',
 'AXISBANK','HCLTECH','HINDUNILVR','INFY','SBIN','ICICIBANK','HDFCBANK','UPL','TMPV']
TK=[t+'.NS' for t in TICKERS]
START='2015-01-01'; END='2026-08-24'; H=21; EMBARGO=25
print("downloading", flush=True)
raw=yf.download(TK,start=START,end=END,interval='1d',progress=False,auto_adjust=True)
raw.to_pickle('raw2.pkl')
bench=yf.download('^NSEI',start=START,end=END,interval='1d',progress=False,auto_adjust=True)['Close']
bench.to_pickle('bench2.pkl')
C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=int(len(C)*0.6)); V=V[C.columns]
print("panel",C.shape,flush=True)

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).rolling(n).mean(); dn=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))

feats={}
r1=C.pct_change()
feats['mom_21']  = C/C.shift(21)-1
feats['mom_63']  = C/C.shift(63)-1
feats['mom_252_21']=C.shift(21)/C.shift(252)-1          # 12-1 momentum
feats['rev_5']   = C/C.shift(5)-1                        # short-term reversal
feats['vol_21']  = r1.rolling(21).std()*np.sqrt(252)
feats['vol_63']  = r1.rolling(63).std()*np.sqrt(252)
feats['c_ma10']  = C/C.rolling(10).mean()-1
feats['c_ma50']  = C/C.rolling(50).mean()-1
feats['c_ma200'] = C/C.rolling(200).mean()-1
sd20=C.rolling(20).std()
feats['bb_pos']  = (C-C.rolling(20).mean())/(2*sd20)
feats['rsi14']   = C.apply(rsi)/100
feats['volr']    = V.rolling(5).mean()/V.rolling(63).mean()
feats['dist_hi'] = C/C.rolling(252).max()-1
feats['dn_vol']  = r1.clip(upper=0).rolling(63).std()*np.sqrt(252)
feats['skew63']  = r1.rolling(63).skew()
FN=list(feats)

# cross-sectional rank-normalise each feature each day -> scale-free, stationary, comparable across names
X={k: v.rank(axis=1,pct=True) for k,v in feats.items()}
fwd=C.shift(-H)/C-1
y=fwd.sub(fwd.mean(axis=1),axis=0)                        # relative to cross-section

rows=[]
for d in C.index:
    if d not in y.index: continue
    blk=pd.DataFrame({k:X[k].loc[d] for k in FN})
    blk['y']=y.loc[d]; blk['date']=d; blk['tic']=blk.index
    rows.append(blk)
P=pd.concat(rows).dropna()
print("panel rows",len(P),flush=True)

def fit(tr):
    m=xg.XGBRegressor(objective='reg:squarederror',n_estimators=400,max_depth=4,learning_rate=0.05,
        subsample=0.8,colsample_bytree=0.7,min_child_weight=50,reg_lambda=2.0,random_state=0,n_jobs=4)
    m.fit(tr[FN],tr['y']); return m

dates=np.array(sorted(P['date'].unique()))
starts=pd.date_range('2019-01-01',END,freq='6MS')
preds=[]
for s in starts:
    e=s+pd.DateOffset(months=6)
    tr=P[P['date']<=s-pd.Timedelta(days=EMBARGO)]           # embargo: labels look H days ahead
    te=P[(P['date']>=s)&(P['date']<e)]
    if len(tr)<20000 or len(te)==0: continue
    m=fit(tr)
    o=te[['date','tic']].copy(); o['pred']=m.predict(te[FN]); o['y']=te['y'].values
    preds.append(o)
    print(f"  {s.date()} train={len(tr)} test={len(te)}",flush=True)
PR=pd.concat(preds)
PR.to_pickle('pred_v2.pkl')
ic=PR.groupby('date').apply(lambda g: g['pred'].corr(g['y'],method='spearman')).dropna()
print(f"\nV2 IC: mean={ic.mean():+.4f} std={ic.std():.3f} t={ic.mean()/ic.std()*np.sqrt(len(ic)):+.2f} n={len(ic)}",flush=True)
json.dump({'ic_mean':float(ic.mean()),'ic_t':float(ic.mean()/ic.std()*np.sqrt(len(ic))),'n':int(len(ic))},open('v2_ic.json','w'))
