from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import backtest_micro as b
import replicate_2025 as r

ASSET=b.ASSET
OUT=Path(f'micro_swing25_{ASSET}'); OUT.mkdir(exist_ok=True)
HORIZONS={'60s':600,'120s':1200,'300s':3000}
MOVE_BP=10.0
DIR_TH=.65; BIG_TH=.60


def add_labels(df):
    out=[]
    for (day,win),g in df.groupby(['day','window'],sort=False):
        g=g.copy()
        for name,h in HORIZONS.items():
            f=g.mid.shift(-h); rb=(f/g.mid-1)*10000
            yb=pd.Series(np.nan,index=g.index); ok=f.notna(); yb[ok]=(rb[ok].abs()>=MOVE_BP).astype(int); g[f'ybig_{name}']=yb
            yd=pd.Series(np.nan,index=g.index); big=ok&(rb.abs()>=MOVE_BP); yd[big]=(rb[big]>0).astype(int); g[f'ydir_{name}']=yd
        out.append(g)
    return pd.concat(out,ignore_index=True)


def xy(df,target,n=350000,seed=1):
    m=df[target].notna(); x=df.loc[m,b.FEATURES].replace([np.inf,-np.inf],np.nan).dropna(); y=df.loc[x.index,target].astype(int)
    if len(x)>n:
        rng=np.random.default_rng(seed); ii=np.sort(rng.choice(len(x),n,replace=False)); x=x.iloc[ii]; y=y.iloc[ii]
    return x,y


def fit(x,y):
    return Pipeline([('scale',StandardScaler()),('m',LogisticRegression(C=.5,max_iter=250,class_weight='balanced'))]).fit(x,y)


def pred(m,df):
    x=df[b.FEATURES].replace([np.inf,-np.inf],np.nan); ok=x.notna().all(axis=1); p=np.full(len(df),np.nan)
    if ok.any(): p[ok.to_numpy()]=m.predict_proba(x.loc[ok])[:,1]
    return p


def auc(y,p):
    ok=np.isfinite(y)&np.isfinite(p)
    return float(roc_auc_score(y[ok].astype(int),p[ok])) if ok.sum()>1 and len(np.unique(y[ok]))>1 else None


def main():
    fs=[]; logs=[]
    for day,split in r.DATES:
        print(ASSET,day,split,flush=True); x=b.build_day(day,split)
        if x.empty: continue
        logs.append({'day':day,'split':split,'rows':len(x),'spread_bp':float(x.spread_bps.median()),'collector_lag_ms':float(x.collector_lag_ms.median())})
        fs.append(x[r.KEEP].copy())
    df=add_labels(pd.concat(fs,ignore_index=True)); train=df[df.split.eq('TRAIN')]
    cls=[]; trades=[]
    for name,h in HORIZONS.items():
        xb,yb=xy(train,f'ybig_{name}',seed=701+h); xd,yd=xy(train,f'ydir_{name}',seed=801+h); mb=fit(xb,yb); md=fit(xd,yd); pb=pred(mb,df); pdn=pred(md,df)
        for split in ['VALID','TEST']:
            q=df.split.eq(split).to_numpy(); ybig=df[f'ybig_{name}'].to_numpy(); ydir=df[f'ydir_{name}'].to_numpy(); ob=q&np.isfinite(ybig)&np.isfinite(pb); od=q&np.isfinite(ydir)&np.isfinite(pdn)
            cls.append({'asset':ASSET,'split':split,'horizon':name,'rows':int(q.sum()),'bigmove_rate':float(np.nanmean(ybig[q])),'big_auc':auc(ybig[ob],pb[ob]),'dirbig_auc':auc(ydir[od],pdn[od]),'dirbig_acc':float(accuracy_score(ydir[od].astype(int),pdn[od]>=.5)) if od.any() else None,'dirbig_n':int(od.sum())})
        valid=np.isfinite(pb)&np.isfinite(pdn); act=np.zeros(len(df),dtype=np.int8); act[valid&(pb>=BIG_TH)&(pdn>=DIR_TH)]=1; act[valid&(pb>=BIG_TH)&(pdn<=1-DIR_TH)]=-1
        for lat in [1,3]:
            for fee in [0,2,5,10]:
                z=b.simulate(df,act,h,lat,fee,'TEST'); z.update({'horizon':name,'move_label_bp':MOVE_BP,'model':'LOGIT_FROZEN','pbig_threshold':BIG_TH,'pdir_threshold':DIR_TH}); trades.append(z)
    c=pd.DataFrame(cls); t=pd.DataFrame(trades); c.to_csv(OUT/'classification.csv',index=False); t.to_csv(OUT/'trading.csv',index=False); (OUT/'data_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    p=t[(t.latency_ms==100)&(t.fee_bp_roundtrip.isin([0,2,5,10]))]
    lines=[f'# HYDRA Micro-to-minute frozen replication 2025 — {ASSET}','',
           'Microsecond raw quotes/trades -> 100ms features -> predict 10bp+ moves at 60s/120s/300s.','Jan-Jun TRAIN, Jul-Aug VALID, Sep-Dec TEST. Fixed p(big)>=0.60 and p(direction)>=0.65/<=0.35.','Execution starts next 100ms bucket, crosses quoted spread, then explicit fee stress is subtracted.','','## TEST classification',c[c.split.eq('TEST')].to_markdown(index=False),'','## TEST economics, 100ms latency',p.to_markdown(index=False)]
    (OUT/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8'); print('\n'.join(lines),flush=True)

if __name__=='__main__': main()
