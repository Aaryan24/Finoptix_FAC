import warnings, time, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
TK=[l.strip() for l in open('all_tickers.txt') if l.strip()]
print(f"downloading {len(TK)} tickers from 2010", flush=True)
parts=[]
for i in range(0,len(TK),50):
    ch=TK[i:i+50]
    for a in range(3):
        try:
            d=yf.download(ch,start='2010-01-01',end='2026-08-24',interval='1d',
                          progress=False,auto_adjust=True,threads=True)
            if len(d): parts.append(d); break
        except Exception as e:
            print("retry",i,type(e).__name__,flush=True); time.sleep(4)
    if i%250==0: print(f"  {i+len(ch)}/{len(TK)}",flush=True)
    time.sleep(0.5)
big=pd.concat(parts,axis=1); big=big.loc[:,~big.columns.duplicated()]
big.to_pickle('raw_big.pkl')
C=big['Close']
print("panel",C.shape,"cols with >1500 obs:",int((C.notna().sum()>1500).sum()),flush=True)
print("DONE",flush=True)
