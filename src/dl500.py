import warnings, time, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
TK=[l.strip() for l in open('n500_tickers.txt') if l.strip()]
print(f"downloading {len(TK)} tickers 2015->2026", flush=True)
parts=[]
for i in range(0,len(TK),40):
    ch=TK[i:i+40]
    for attempt in range(3):
        try:
            d=yf.download(ch,start='2015-01-01',end='2026-08-24',interval='1d',
                          progress=False,auto_adjust=True,threads=True)
            if len(d): parts.append(d); break
        except Exception as e:
            print("  retry",i,type(e).__name__,flush=True); time.sleep(5)
    print(f"  {i+len(ch)}/{len(TK)}",flush=True)
    time.sleep(1)
big=pd.concat(parts,axis=1)
big=big.loc[:,~big.columns.duplicated()]
big.to_pickle('raw500.pkl')
C=big['Close']
print("panel",C.shape,"non-null cols",int((C.notna().sum()>1000).sum()),flush=True)
print("DONE",flush=True)
