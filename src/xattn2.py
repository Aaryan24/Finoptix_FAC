"""Cross-sectional transformer: tokens = stocks on a date. Sector embeddings,
top-weighted ranking loss, multi-horizon heads. Evaluated on the same monthly dates."""
import warnings, json, sys, numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore"); torch.set_num_threads(8)
DEV=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PKL=sys.argv[1] if len(sys.argv)>1 else 'raw_big.pkl'
MINOBS=1500; MIN_ADV=3e7; HS=[5,21,63]; HMAIN=21; EMB=25
raw=pd.read_pickle(PKL); C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=MINOBS); V=V.reindex(columns=C.columns)
r1=C.pct_change(); adv=(C*V).rolling(63).median(); liq=adv>MIN_ADV
mkt=r1.mean(axis=1)
print("panel",C.shape,flush=True)

# ---------- sector map ----------
sec=pd.read_csv('sectors.csv'); sec['Symbol']=sec['Symbol'].str.strip()+'.NS'
smap={r.Symbol:r.Industry for r in sec.itertuples()}
inds=sorted(set(smap.values())); iidx={s:i for i,s in enumerate(inds)}
SEC=np.array([iidx.get(smap.get(c,None),len(inds)) for c in C.columns])   # last idx = unknown
NSEC=len(inds)+1
print("sectors:",NSEC,flush=True)

F={}
F['mom_12_1']=C.shift(21)/C.shift(252)-1; F['mom_6_1']=C.shift(21)/C.shift(126)-1
F['rev_1m']=C/C.shift(21)-1;              F['rev_1w']=C/C.shift(5)-1
F['vol_63']=r1.rolling(63).std();         F['dn_vol']=r1.clip(upper=0).rolling(63).std()
F['max_21']=r1.rolling(21).max();         F['skew_63']=r1.rolling(63).skew()
F['dist_hi']=C/C.rolling(252).max()-1;    F['c_ma200']=C/C.rolling(200).mean()-1
F['turn']=(C*V).replace(0,np.nan).rolling(21,min_periods=10).mean()/(C*V).replace(0,np.nan).rolling(252,min_periods=120).mean()
_dv=(C*V).replace(0,np.nan)
F['amihud']=(r1.abs()/_dv).rolling(63,min_periods=30).mean()*1e11
F['size']=np.log(adv)
cov=r1.rolling(252).cov(mkt); var=mkt.rolling(252).var(); F['beta']=cov.div(var,axis=0)
F['ivol']=(r1.sub(F['beta'].mul(mkt,axis=0))).rolling(126).std()
FN=list(F)
X={k:v.where(liq).rank(axis=1,pct=True)-0.5 for k,v in F.items()}
# sector-relative ranks for the 3 strongest factors
secser=pd.Series(SEC,index=C.columns)
for k in ['mom_12_1','c_ma200','size']:
    z=F[k].where(liq)
    X[k+'_sec']=z.groupby(secser,axis=1).rank(pct=True)-0.5
FN2=list(X)
Ys={h: (lambda f: f.sub(f.mean(axis=1),axis=0))((C.shift(-h)/C-1).where(liq)) for h in HS}
print("features",len(FN2),flush=True)

idx=C.index
mall=[pd.Timestamp(d) for d in pd.Series(idx).groupby([idx.year,idx.month]).first().values]
weeks=idx[::5]                                     # weekly sampling -> ~4x data
train_dates=[d for d in weeks if d>=idx[300] and d<=idx[-max(HS)-1]]
eval_dates=[d for d in mall if d>=pd.Timestamp('2021-01-01') and d<=idx[-HMAIN-1]]
Xv={k:X[k].values for k in FN2}; Yv={h:Ys[h].values for h in HS}
pos={d:i for i,d in enumerate(idx)}

def snap(d):
    i=pos[d]
    st=np.stack([Xv[k][i] for k in FN2],1)
    ys=np.stack([Yv[h][i] for h in HS],1)
    ok=np.isfinite(st).all(1)&np.isfinite(ys[:,HS.index(HMAIN)])
    if ok.sum()<80: return None
    yy=np.nan_to_num(ys[ok])
    return (st[ok].astype(np.float32), yy.astype(np.float32), SEC[ok],
            np.array(C.columns)[ok])
SNAP={}
for d in set(train_dates)|set(eval_dates):
    s=snap(d)
    if s: SNAP[d]=s
train_dates=[d for d in train_dates if d in SNAP]; eval_dates=[d for d in eval_dates if d in SNAP]
print("train snapshots",len(train_dates),"eval",len(eval_dates),
      "mean stocks/day",int(np.mean([SNAP[d][0].shape[0] for d in train_dates])),flush=True)

class XAttn(nn.Module):
    """tokens = stocks; self-attention over the cross-section (no positional encoding)."""
    def __init__(self,nf,nsec,d=64,heads=4,layers=2,p=0.2):
        super().__init__()
        self.inp=nn.Linear(nf,d); self.sec=nn.Embedding(nsec,d)
        L=nn.TransformerEncoderLayer(d,heads,d*2,dropout=p,batch_first=True,norm_first=True)
        self.tr=nn.TransformerEncoder(L,layers)
        self.heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(d),nn.Linear(d,1)) for _ in HS])
    def forward(self,x,s):
        h=self.inp(x)+self.sec(s)
        h=self.tr(h.unsqueeze(0)).squeeze(0)
        return [hd(h).squeeze(-1) for hd in self.heads]

def topw_loss(p,y,frac=0.2):
    """rank-correlation loss, weighted toward the top of the true distribution"""
    pc=p-p.mean(); yc=y-y.mean()
    base=-(pc*yc).sum()/(pc.norm()*yc.norm()+1e-8)
    k=max(10,int(len(y)*frac))
    idx=torch.topk(y,k).indices
    pt=p[idx]-p[idx].mean(); yt=y[idx]-y[idx].mean()
    top=-(pt*yt).sum()/(pt.norm()*yt.norm()+1e-8)
    return 0.6*base+0.4*top

def fit(tr,va,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net=XAttn(len(FN2),NSEC).to(DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=1e-3)
    best=-9;bs=None;bad=0
    for ep in range(30):
        net.train(); np.random.shuffle(tr)
        for d in tr:
            st,yy,sc,_=SNAP[d]
            x=torch.from_numpy(st).to(DEV); s=torch.from_numpy(sc).long().to(DEV)
            y=torch.from_numpy(yy).to(DEV)
            ps=net(x,s)
            loss=sum(topw_loss(ps[i],y[:,i]) for i in range(len(HS)))/len(HS)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        net.eval(); ics=[]
        with torch.no_grad():
            for d in va:
                st,yy,sc,_=SNAP[d]
                p=net(torch.from_numpy(st).to(DEV),torch.from_numpy(sc).long().to(DEV))[HS.index(HMAIN)].cpu().numpy()
                ics.append(pd.Series(p).corr(pd.Series(yy[:,HS.index(HMAIN)]),method='spearman'))
        v=np.nanmean(ics)
        if v>best: best=v;bs={k:t.clone() for k,t in net.state_dict().items()};bad=0
        else:
            bad+=1
            if bad>=5: break
    net.load_state_dict(bs); return net

rows=[]
for f in [pd.Timestamp(x) for x in pd.date_range('2021-01-01','2026-01-01',freq='12MS')]:
    tr=[d for d in train_dates if d<=f-pd.Timedelta(days=EMB+max(HS))]
    te=[d for d in eval_dates if f<=d<f+pd.DateOffset(months=12)]
    if len(tr)<200 or not te: continue
    cut=tr[-26]; va=[d for d in tr if d>=cut]; tr2=[d for d in tr if d<cut]
    P=[]
    for sd in [0,1,2]:
        net=fit(list(tr2),va,sd); net.eval()
        pr={}
        with torch.no_grad():
            for d in te:
                st,yy,sc,tk=SNAP[d]
                pr[d]=net(torch.from_numpy(st).to(DEV),torch.from_numpy(sc).long().to(DEV))[HS.index(HMAIN)].cpu().numpy()
        P.append(pr)
    for d in te:
        st,yy,sc,tk=SNAP[d]
        rows.append(pd.DataFrame({'date':d,'tic':tk,'pred':np.mean([p[d] for p in P],0),
                                  'y':yy[:,HS.index(HMAIN)]}))
    pd.concat(rows).to_pickle('xattn2_partial.pkl')          # checkpoint after every fold
    _ic=pd.concat(rows).groupby('date').apply(lambda g: g['pred'].corr(g['y'],method='spearman')).dropna()
    print(f"  fold {f.date()} train={len(tr2)} val={len(va)} months={len(te)}  running IC={_ic.mean():+.4f} n={len(_ic)}",flush=True)
R=pd.concat(rows); R.to_pickle('xattn2_pred.pkl')
ic=R.groupby('date').apply(lambda g: g['pred'].corr(g['y'],method='spearman')).dropna()
print(f"\nX-ATTN IC={ic.mean():+.4f}  t={ic.mean()/ic.std()*np.sqrt(len(ic)):+.2f}  n={len(ic)}",flush=True)
json.dump({'ic':float(ic.mean()),'t':float(ic.mean()/ic.std()*np.sqrt(len(ic))),'n':int(len(ic))},open('xattn2_ic.json','w'))
print("DONE",flush=True)
