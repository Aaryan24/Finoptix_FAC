import warnings, json, numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
torch.set_num_threads(8)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEQ=60; H=21; EMB_DAYS=25
raw=pd.read_pickle('raw500.pkl'); C,V=raw['Close'],raw['Volume']
C=C.dropna(axis=1,thresh=800); V=V.reindex(columns=C.columns)
r1=C.pct_change(); adv=(C*V).rolling(63).median(); liq=adv>5e7
mkt=r1.mean(axis=1)

# ---- 15 static cross-sectional factors (same as v3) ----
F={}
F['mom_12_1']=C.shift(21)/C.shift(252)-1; F['mom_6_1']=C.shift(21)/C.shift(126)-1
F['rev_1m']=C/C.shift(21)-1;              F['rev_1w']=C/C.shift(5)-1
F['vol_63']=r1.rolling(63).std();         F['dn_vol']=r1.clip(upper=0).rolling(63).std()
F['max_21']=r1.rolling(21).max();         F['skew_63']=r1.rolling(63).skew()
F['dist_hi']=C/C.rolling(252).max()-1;    F['c_ma200']=C/C.rolling(200).mean()-1
F['turn']=(C*V).rolling(21).mean()/(C*V).rolling(252).mean()
F['amihud']=(r1.abs()/(C*V)).rolling(63).mean()*1e11
F['size']=np.log(adv)
cov=r1.rolling(252).cov(mkt); var=mkt.rolling(252).var(); F['beta']=cov.div(var,axis=0)
F['ivol']=(r1.sub(F['beta'].mul(mkt,axis=0))).rolling(126).std()
FN=list(F)
X={k:v.where(liq).rank(axis=1,pct=True) for k,v in F.items()}
fwd=(C.shift(-H)/C-1).where(liq); Y=fwd.sub(fwd.mean(axis=1),axis=0)

# ---- monthly sample dates ----
idx=C.index
mdates=[d for d in pd.Series(idx).groupby([idx.year,idx.month]).first().values]
mdates=[pd.Timestamp(d) for d in mdates if pd.Timestamp(d)>=idx[300] and pd.Timestamp(d)<=idx[-H-1]]
logv=np.log(V.replace(0,np.nan)); dlv=logv.diff()
cm20=(C/C.rolling(20).mean()-1)

seqs=[];stats=[];ys=[];dts=[];tks=[]
pos={d:i for i,d in enumerate(idx)}
Rv,Dv,Mv=r1.values,dlv.values,cm20.values
Xv={k:X[k].values for k in FN}; Yv=Y.values
cols=list(C.columns)
for d in mdates:
    i=pos[d]
    if i<SEQ+1: continue
    sl=slice(i-SEQ+1,i+1)
    blockR,blockD,blockM=Rv[sl],Dv[sl],Mv[sl]
    yrow=Yv[i]
    st=np.stack([Xv[k][i] for k in FN],1)
    ok=np.isfinite(yrow)&np.isfinite(st).all(1)&np.isfinite(blockR).all(0)&np.isfinite(blockD).all(0)&np.isfinite(blockM).all(0)
    if ok.sum()<60: continue
    s=np.stack([blockR[:,ok],blockD[:,ok],blockM[:,ok]],-1)        # (SEQ, n, 3)
    s=np.transpose(s,(1,0,2))                                       # (n, SEQ, 3)
    mu=s.mean(1,keepdims=True); sd=s.std(1,keepdims=True)+1e-8
    s=(s-mu)/sd                                                     # per-sample z-score => scale free
    seqs.append(s.astype(np.float32)); stats.append(st[ok].astype(np.float32))
    ys.append(yrow[ok].astype(np.float32)); dts += [d]*int(ok.sum())
S=np.concatenate(seqs); ST=np.concatenate(stats); YY=np.concatenate(ys); DT=np.array(dts)
print("samples",S.shape,"dates",len(set(DT)),flush=True)

class Net(nn.Module):
    def __init__(self,nf=3,ns=len(FN),hid=32,emb=16,p=0.3):
        super().__init__()
        self.lstm=nn.LSTM(nf,hid,batch_first=True)
        self.proj=nn.Linear(hid,emb)
        self.head=nn.Sequential(nn.Linear(emb+ns,32),nn.ReLU(),nn.Dropout(p),nn.Linear(32,1))
    def forward(self,s,st):
        _,(h,_)=self.lstm(s)
        return self.head(torch.cat([self.proj(h[-1]),st],-1)).squeeze(-1)

def corr_loss(p,y):
    p=p-p.mean(); y=y-y.mean()
    return -(p*y).sum()/(p.norm()*y.norm()+1e-8)

def train_fold(tr_idx,va_idx,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net=Net().to(DEV); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    trd=sorted(set(DT[tr_idx])); vad=sorted(set(DT[va_idx]))
    best=-9; bstate=None; bad=0
    for ep in range(40):
        net.train(); np.random.shuffle(trd)
        for d in trd:
            m=tr_idx[DT[tr_idx]==d]
            if len(m)<40: continue
            s=torch.from_numpy(S[m]).to(DEV); st=torch.from_numpy(ST[m]).to(DEV); y=torch.from_numpy(YY[m]).to(DEV)
            opt.zero_grad(); loss=corr_loss(net(s,st),y); loss.backward(); opt.step()
        net.eval(); ics=[]
        with torch.no_grad():
            for d in vad:
                m=va_idx[DT[va_idx]==d]
                if len(m)<40: continue
                p=net(torch.from_numpy(S[m]).to(DEV),torch.from_numpy(ST[m]).to(DEV)).cpu().numpy()
                ics.append(pd.Series(p).corr(pd.Series(YY[m]),method='spearman'))
        v=np.nanmean(ics)
        if v>best: best=v; bstate={k:t.clone() for k,t in net.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=6: break
    net.load_state_dict(bstate); return net,best

alld=np.array(sorted(set(DT)))
folds=[pd.Timestamp(x) for x in pd.date_range('2019-01-01','2026-01-01',freq='12MS')]
rows=[]
for f in folds:
    tr=np.where(DT<=f-pd.Timedelta(days=EMB_DAYS+H))[0]
    te=np.where((DT>=f)&(DT<f+pd.DateOffset(months=12)))[0]
    if len(tr)<8000 or len(te)==0: continue
    cut=sorted(set(DT[tr]))[-12]
    va=tr[DT[tr]>=cut]; tr2=tr[DT[tr]<cut]
    preds=[]
    for sd in [0,1,2]:
        net,bv=train_fold(tr2,va,sd)
        net.eval()
        with torch.no_grad():
            p=net(torch.from_numpy(S[te]).to(DEV),torch.from_numpy(ST[te]).to(DEV)).cpu().numpy()
        preds.append(p)
    p=np.mean(preds,0)
    rows.append(pd.DataFrame({'date':DT[te],'pred':p,'y':YY[te]}))
    print(f"  fold {f.date()} train={len(tr2)} val={len(va)} test={len(te)}",flush=True)
R=pd.concat(rows); R.to_pickle('nn_pred.pkl')
ic=R.groupby('date').apply(lambda g: g['pred'].corr(g['y'],method='spearman')).dropna()
n=len(ic); m=ic.mean(); x=ic.values-m; g0=(x@x)/n; nw=g0
for L in range(1,2): nw+=2*(1-L/2)*((x[:-L]@x[L:])/n)
print(f"\nLSTM ensemble IC={m:+.4f}  t={m/ic.std()*np.sqrt(n):+.2f}  n={n} (monthly obs, non-overlapping)",flush=True)
json.dump({'ic':float(m),'t':float(m/ic.std()*np.sqrt(n)),'n':int(n)},open('nn_ic.json','w'))
print("DONE",flush=True)
