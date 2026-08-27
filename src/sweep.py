"""Sweep target/loss variants for the cross-sectional transformer head."""
import warnings, json, sys, numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore"); torch.set_num_threads(8)
DEV=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
HS=[5,21,63]; HMAIN=21; EMB=25; MIN_ADV=3e7
raw=pd.read_pickle('raw_big.pkl'); C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=1500); V=V.reindex(columns=C.columns)
r1=C.pct_change(); adv=(C*V).rolling(63).median(); liq=adv>MIN_ADV; mkt=r1.mean(axis=1)
sec=pd.read_csv('sectors.csv'); sec['Symbol']=sec['Symbol'].str.strip()+'.NS'
smap={r.Symbol:r.Industry for r in sec.itertuples()}
inds=sorted(set(smap.values())); iidx={s:i for i,s in enumerate(inds)}
SEC=np.array([iidx.get(smap.get(c),len(inds)) for c in C.columns]); NSEC=len(inds)+1
F={}
F['mom_12_1']=C.shift(21)/C.shift(252)-1; F['mom_6_1']=C.shift(21)/C.shift(126)-1
F['rev_1m']=C/C.shift(21)-1; F['rev_1w']=C/C.shift(5)-1
F['vol_63']=r1.rolling(63).std(); F['dn_vol']=r1.clip(upper=0).rolling(63).std()
F['max_21']=r1.rolling(21).max(); F['skew_63']=r1.rolling(63).skew()
F['dist_hi']=C/C.rolling(252).max()-1; F['c_ma200']=C/C.rolling(200).mean()-1
F['turn']=(C*V).rolling(21).mean()/(C*V).rolling(252).mean()
F['amihud']=(r1.abs()/(C*V)).rolling(63).mean()*1e11; F['size']=np.log(adv)
cov=r1.rolling(252).cov(mkt); var=mkt.rolling(252).var(); F['beta']=cov.div(var,axis=0)
F['ivol']=(r1.sub(F['beta'].mul(mkt,axis=0))).rolling(126).std()
X={k:v.where(liq).rank(axis=1,pct=True)-0.5 for k,v in F.items()}
secser=pd.Series(SEC,index=C.columns)
for k in ['mom_12_1','c_ma200','size']:
    X[k+'_sec']=F[k].where(liq).groupby(secser,axis=1).rank(pct=True)-0.5
FN=list(X)
RAW={h:(C.shift(-h)/C-1).where(liq) for h in HS}
DEM={h:RAW[h].sub(RAW[h].mean(axis=1),axis=0) for h in HS}
VOLS=r1.rolling(63).std()*np.sqrt(252)
idx=C.index; weeks=idx[::5]
mall=[pd.Timestamp(d) for d in pd.Series(idx).groupby([idx.year,idx.month]).first().values]
tdates=[d for d in weeks if d>=idx[300] and d<=idx[-max(HS)-1]]
edates=[d for d in mall if d>=pd.Timestamp('2021-01-01') and d<=idx[-HMAIN-1]]
Xv={k:X[k].values for k in FN}; Dv={h:DEM[h].values for h in HS}; VV=VOLS.values
pos={d:i for i,d in enumerate(idx)}
SNAP={}
for d in set(tdates)|set(edates):
    i=pos[d]; st=np.stack([Xv[k][i] for k in FN],1)
    ys=np.stack([Dv[h][i] for h in HS],1); vv=VV[i]
    ok=np.isfinite(st).all(1)&np.isfinite(ys[:,HS.index(HMAIN)])&np.isfinite(vv)&(vv>1e-6)
    if ok.sum()<80: continue
    SNAP[d]=(st[ok].astype(np.float32), np.nan_to_num(ys[ok]).astype(np.float32),
             SEC[ok], np.array(C.columns)[ok], vv[ok].astype(np.float32))
tdates=[d for d in tdates if d in SNAP]; edates=[d for d in edates if d in SNAP]
print(f"train {len(tdates)} eval {len(edates)} feats {len(FN)}",flush=True)

class Net(nn.Module):
    def __init__(self,nf,nsec,d=64,heads=4,layers=2,p=0.2,nout=len(HS)):
        super().__init__()
        self.inp=nn.Linear(nf,d); self.sec=nn.Embedding(nsec,d)
        L=nn.TransformerEncoderLayer(d,heads,d*2,dropout=p,batch_first=True,norm_first=True)
        self.tr=nn.TransformerEncoder(L,layers)
        self.heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(d),nn.Linear(d,1)) for _ in range(nout)])
    def forward(self,x,s):
        h=self.tr((self.inp(x)+self.sec(s)).unsqueeze(0)).squeeze(0)
        return [hd(h).squeeze(-1) for hd in self.heads]

def corr(p,y):
    p=p-p.mean(); y=y-y.mean(); return (p*y).sum()/(p.norm()*y.norm()+1e-8)
def make_target(mode,y,vol):
    if mode=='demean': return y
    if mode=='rank':   return (torch.argsort(torch.argsort(y)).float()/(len(y)-1)-0.5)
    if mode=='volscl': return y/ (vol+1e-6)
    if mode=='clip':   return torch.clamp(y,-0.25,0.25)
    return y
def loss_fn(mode,p,t):
    if mode=='corr':   return -corr(p,t)
    if mode=='topw':
        k=max(10,int(len(t)*0.2)); i=torch.topk(t,k).indices
        return -(0.6*corr(p,t)+0.4*corr(p[i],t[i]))
    if mode=='softk':                      # differentiable top-k portfolio return
        w=torch.softmax(p*8.0,0); return -(w*t).sum()
    if mode=='bce':                        # is it top decile?
        lab=(t>=torch.quantile(t,0.9)).float()
        return nn.functional.binary_cross_entropy_with_logits(p,lab)
    return -corr(p,t)

CFG=[('demean','topw','BASELINE demean+topw'),
     ('rank','corr','rank target + corr'),
     ('rank','topw','rank target + topw'),
     ('volscl','topw','vol-scaled target + topw'),
     ('clip','topw','clipped target + topw'),
     ('demean','softk','demean + soft top-k portfolio'),
     ('demean','bce','top-decile classification')]
def fit(tr,va,seed,tmode,lmode):
    torch.manual_seed(seed); np.random.seed(seed)
    net=Net(len(FN),NSEC).to(DEV); opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=1e-3)
    best=-9;bs=None;bad=0
    for ep in range(25):
        net.train(); np.random.shuffle(tr)
        for d in tr:
            st,yy,sc,_,vv=SNAP[d]
            x=torch.from_numpy(st).to(DEV); s=torch.from_numpy(sc).long().to(DEV)
            y=torch.from_numpy(yy).to(DEV); v=torch.from_numpy(vv).to(DEV)
            ps=net(x,s)
            L=sum(loss_fn(lmode,ps[i],make_target(tmode,y[:,i],v)) for i in range(len(HS)))/len(HS)
            opt.zero_grad(); L.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        net.eval(); sc_=[]
        with torch.no_grad():
            for d in va:
                st,yy,s2,_,vv=SNAP[d]
                p=net(torch.from_numpy(st).to(DEV),torch.from_numpy(s2).long().to(DEV))[HS.index(HMAIN)].cpu().numpy()
                k=np.argsort(-p)[:30]; sc_.append(yy[k,HS.index(HMAIN)].mean())   # select on TOP-30 return
        v_=np.nanmean(sc_)
        if v_>best: best=v_;bs={k:t.clone() for k,t in net.state_dict().items()};bad=0
        else:
            bad+=1
            if bad>=5: break
    net.load_state_dict(bs); return net
res={}
for tmode,lmode,lab in CFG:
    rows=[]
    for f in [pd.Timestamp(x) for x in pd.date_range('2021-01-01','2026-01-01',freq='12MS')]:
        tr=[d for d in tdates if d<=f-pd.Timedelta(days=EMB+max(HS))]
        te=[d for d in edates if f<=d<f+pd.DateOffset(months=12)]
        if len(tr)<200 or not te: continue
        cut=tr[-26]; va=[d for d in tr if d>=cut]; tr2=[d for d in tr if d<cut]
        P=[]
        for sd in [0,1]:
            net=fit(list(tr2),va,sd,tmode,lmode); net.eval()
            pr={}
            with torch.no_grad():
                for d in te:
                    st,yy,s2,tk,vv=SNAP[d]
                    pr[d]=net(torch.from_numpy(st).to(DEV),torch.from_numpy(s2).long().to(DEV))[HS.index(HMAIN)].cpu().numpy()
            P.append(pr)
        for d in te:
            st,yy,s2,tk,vv=SNAP[d]
            rows.append(pd.DataFrame({'date':d,'tic':tk,'pred':np.mean([p[d] for p in P],0),
                                      'y':yy[:,HS.index(HMAIN)]}))
    R=pd.concat(rows); R.to_pickle(f'sweep_{tmode}_{lmode}.pkl')
    ic=R.groupby('date').apply(lambda g: g['pred'].corr(g['y'],method='spearman')).dropna()
    top=R.groupby('date').apply(lambda g: g.nlargest(30,'pred')['y'].mean()).dropna()
    res[lab]=dict(ic=float(ic.mean()), t_ic=float(ic.mean()/ic.std()*np.sqrt(len(ic))),
                  top=float(top.mean()*100), t_top=float(top.mean()/top.std()*np.sqrt(len(top))))
    print(f"  {lab:34} IC={res[lab]['ic']:+.4f}  top30={res[lab]['top']:+.3f}%/mo  t={res[lab]['t_top']:+.2f}",flush=True)
json.dump(res,open('sweep.json','w'),indent=2)
print("DONE",flush=True)
