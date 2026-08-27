import warnings, json, numpy as np, pandas as pd, yfinance as yf, xgboost as xg
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

TICKERS = ['BHARTIARTL','LTIM','HDFCLIFE','NTPC','MARUTI','NESTLEIND','BAJFINANCE','KOTAKBANK','TATASTEEL','ONGC',
 'BAJAJ-AUTO','LT','ITC','TCS','BRITANNIA','ADANIENT','CIPLA','WIPRO','INDUSINDBK','ULTRACEMCO','TATACONSUM',
 'BAJAJFINSV','RELIANCE','HEROMOTOCO','COALINDIA','TITAN','HINDALCO','APOLLOHOSP','TECHM','DRREDDY','DIVISLAB',
 'EICHERMOT','BPCL','SBILIFE','GRASIM','JSWSTEEL','ASIANPAINT','POWERGRID','ADANIPORTS','M&M','SUNPHARMA',
 'AXISBANK','HCLTECH','HINDUNILVR','INFY','SBIN','ICICIBANK','HDFCBANK','UPL','TMPV']   # 50 unique: UPL restored
                                                                                        # (LTIM was duplicated),
                                                                                        # TATAMOTORS -> TMPV
TK=[t+'.NS' for t in TICKERS]
TRAIN_END='2024-05-31'; TEST_START='2024-06-01'; TEST_END='2025-07-31'
FEATS=['volatility_20','ma_10','ma_50','momentum_10','momentum_50','upper_band','lower_band','returns_20',
       'corr_close_vol_20','return_lag_1','return_lag_2','return_lag_3','return_lag_5']

def calc(d, horizon=21):
    f=pd.DataFrame(index=d.index)
    c=d['Close']
    f['Close']=c; f['returns']=c.pct_change()
    f['returns_20']=c.rolling(20).mean().pct_change()
    f['volatility_20']=c.rolling(20).std()
    f['ma_10']=c.rolling(10).mean(); f['ma_50']=c.rolling(50).mean()
    f['momentum_10']=c.rolling(10).mean().pct_change(horizon)
    f['momentum_50']=c.rolling(50).mean().pct_change(horizon)
    f['upper_band']=c.rolling(20).mean()+2*c.rolling(20).std()
    f['lower_band']=c.rolling(20).mean()-2*c.rolling(20).std()
    f['corr_close_vol_20']=c.rolling(20).corr(d['Volume'])
    for l in [1,2,3,5]: f[f'return_lag_{l}']=f['returns'].shift(l)
    return f

print("downloading...", flush=True)
raw=yf.download(TK, start='2021-06-01', end=TEST_END, interval='1d', progress=False, auto_adjust=True)
bench=yf.download('^NSEI', start=TEST_START, end=TEST_END, interval='1d', progress=False, auto_adjust=True)['Close']
raw.to_pickle('raw.pkl'); bench.to_pickle('bench.pkl')
print("downloaded", raw['Close'].shape, "bench", bench.shape, flush=True)

def build(mode):
    """mode='asbuilt' -> y = same-bar return (their code). mode='fixed' -> y = NEXT-bar return."""
    preds={}
    for t in TK:
        try:
            d=pd.DataFrame({'Close':raw['Close'][t],'High':raw['High'][t],'Low':raw['Low'][t],'Volume':raw['Volume'][t]}).dropna()
        except Exception: continue
        f=calc(d)
        f['y']= f['returns'] if mode=='asbuilt' else f['returns'].shift(-1)
        f=f.dropna()
        tr=f[f.index<=TRAIN_END]; te=f[(f.index>=TEST_START)&(f.index<=TEST_END)]
        if len(tr)<200 or len(te)<50: continue
        m=xg.XGBRegressor(objective='reg:squarederror',n_estimators=2000,max_depth=12,
                          learning_rate=0.01,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=2)
        m.fit(tr[FEATS],tr['y'],verbose=False)
        preds[t]=pd.Series(m.predict(te[FEATS]),index=te.index)
    return pd.DataFrame(preds)

def maxsharpe(mu,cov):
    n=len(mu)
    if n==0: return np.array([])
    def neg(w):
        v=np.sqrt(max(w@cov@w,1e-12)); return -(w@mu)/v
    cons=({'type':'eq','fun':lambda w: w.sum()-1},)
    r=minimize(neg,np.repeat(1/n,n),bounds=[(0,1)]*n,constraints=cons,method='SLSQP')
    return r.x if r.success else np.repeat(1/n,n)

def backtest(pred, rets, topn=10, look=252, cost_bps=20):
    """Causal monthly rebalance. Signal & covariance use data strictly BEFORE the rebalance date.
    cost_bps charged on one-way turnover at each rebalance."""
    days=rets.loc[TEST_START:TEST_END].index
    months=pd.Series(days).groupby([days.year,days.month]).first().values
    cur={}   # ticker -> weight, drifts with returns between rebalances
    out=[]; turn_log=[]
    for d in days:
        if d in months:
            sig=pred.loc[pred.index<d].tail(21).mean().dropna()
            if len(sig)>=topn:
                cand=list(sig.nlargest(topn).index)
                hist=rets.loc[rets.index<d, cand].tail(look).dropna(axis=1)
                cand=list(hist.columns)
                if len(cand)>1:
                    w=maxsharpe(hist.mean().values*252, hist.cov().values*252)
                    tgt=dict(zip(cand,w))
                    names=set(tgt)|set(cur)
                    turnover=sum(abs(tgt.get(n,0.0)-cur.get(n,0.0)) for n in names)
                    turn_log.append(turnover)
                    cur=tgt
                    out.append((d, -turnover*cost_bps/1e4))  # charge cost on rebalance day
                    continue
        if cur:
            r=sum(w*rets.loc[d,n] for n,w in cur.items() if pd.notna(rets.loc[d,n]))
            nv={n:w*(1+rets.loc[d,n]) for n,w in cur.items() if pd.notna(rets.loc[d,n])}
            tot=sum(nv.values())
            if tot>0: cur={n:v/tot for n,v in nv.items()}
            out.append((d, float(r)))
    print(f"   rebalances={len(turn_log)} mean one-way turnover={np.mean(turn_log):.2f}", flush=True)
    return pd.Series(dict(out))

def stats(r, rf=0.06):
    n=len(r); ann=(1+r).prod()**(252/n)-1; vol=r.std()*np.sqrt(252)
    cum=(1+r).cumprod(); dd=(cum/cum.cummax()-1).min()
    return dict(total=float((1+r).prod()-1), ann_return=float(ann), ann_vol=float(vol),
                sharpe=float((ann-rf)/vol) if vol>0 else float('nan'), max_dd=float(dd), days=n)

rets=raw['Close'].pct_change()
res={}
for mode in ['asbuilt','fixed']:
    print("fitting",mode,flush=True)
    p=build(mode); p.to_pickle(f'pred_{mode}.pkl')
    r=backtest(p,rets); r.to_pickle(f'ret_{mode}.pkl')
    res[mode]=stats(r); print(mode,res[mode],flush=True)

br=bench.pct_change().dropna(); br=br[(br.index>=TEST_START)&(br.index<=TEST_END)]
res['NIFTY50']=stats(br.squeeze())
ew=rets.loc[TEST_START:TEST_END].mean(axis=1).dropna()
res['equal_weight']=stats(ew)
json.dump(res, open('results.json','w'), indent=2)
print(json.dumps(res, indent=2))
