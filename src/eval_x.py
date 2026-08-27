import warnings, numpy as np, pandas as pd, os
warnings.filterwarnings("ignore")
f='xattn_pred.pkl' if os.path.exists('xattn_pred.pkl') else 'xattn_partial.pkl'
R=pd.read_pickle(f); R['date']=pd.to_datetime(R['date'])
print(f"using {f}: {R['date'].nunique()} months\n")
raw=pd.read_pickle('raw_big.pkl'); C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=1500); V=V.reindex(columns=C.columns)
adv=(C*V).rolling(63).median(); liq=adv>3e7
mom=(C.shift(21)/C.shift(252)-1).where(liq)

def stat(s,lab):
    s=s.dropna(); print(f"  {lab:34}{s.mean():+9.4f}   t={s.mean()/s.std()*np.sqrt(len(s)):+5.2f}   n={len(s)}")
    return s
print("=== IC on the same monthly dates ===")
xic=R.groupby('date').apply(lambda g: g['pred'].corr(g['y'],method='spearman'))
mic={}
for d,g in R.groupby('date'):
    m=mom.loc[d].reindex(g['tic']).values
    ok=np.isfinite(m)&np.isfinite(g['y'].values)
    if ok.sum()>=80: mic[d]=pd.Series(m[ok]).corr(pd.Series(g['y'].values[ok]),method='spearman')
mic=pd.Series(mic)
X=stat(xic,'cross-sectional transformer'); M=stat(mic,'momentum (same dates)')
J=pd.concat([X.rename('x'),M.rename('m')],axis=1).dropna()
stat(J['x']-J['m'],'paired difference (x - mom)')

print("\n=== is the edge concentrated in illiquid names? ===")
for half,lab in [(True,'MORE liquid half'),(False,'LESS liquid half')]:
    ics={}
    for d,g in R.groupby('date'):
        a=adv.loc[d].reindex(g['tic']).values
        med=np.nanmedian(a); sel=(a>=med) if half else (a<med)
        gg=g[sel&np.isfinite(a)]
        if len(gg)>=60: ics[d]=gg['pred'].corr(gg['y'],method='spearman')
    stat(pd.Series(ics),lab)

print("\n=== portfolio: top-30 picks, forward 21d relative return ===")
rows=[]
for d,g in R.groupby('date'):
    m=mom.loc[d].reindex(g['tic']).values
    g=g.assign(mom=m).dropna(subset=['mom'])
    if len(g)<80: continue
    g=g.assign(blend=(g['pred'].rank(pct=True)+g['mom'].rank(pct=True))/2)
    x30=set(g.nlargest(30,'pred')['tic']); m30=set(g.nlargest(30,'mom')['tic'])
    rows.append(dict(date=d, x=g.nlargest(30,'pred')['y'].mean(),
                     m=g.nlargest(30,'mom')['y'].mean(),
                     b=g.nlargest(30,'blend')['y'].mean(), ov=len(x30&m30)))
P=pd.DataFrame(rows)
stat(P['m'],'momentum top-30'); stat(P['x'],'transformer top-30'); stat(P['b'],'BLEND top-30')
stat(P['b']-P['m'],'blend - momentum (paired)')
print(f"\n  avg overlap of top-30 with momentum: {P['ov'].mean():.1f}/30 ({P['ov'].mean()/30*100:.0f}%)")
