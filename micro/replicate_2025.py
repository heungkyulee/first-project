from __future__ import annotations

import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import backtest_micro as b

ASSET=b.ASSET
OUT=Path(f'micro_rep25_{ASSET}'); OUT.mkdir(exist_ok=True)
DATES=[]
for m in range(1,13):
    split='TRAIN' if m<=6 else ('VALID' if m<=8 else 'TEST')
    DATES.append((f'2025-{m:02d}-01',split))
HORIZONS={'5s':50,'10s':100,'30s':300,'60s':600}
MOVE_BP=5.0
DIR_TH=.65; BIG_TH=.60
KEEP=b.FEATURES+['ask_price','bid_price','mid','day','window','split']


def labels(df):
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
    mask=df[target].notna(); x=df.loc[mask,b.FEATURES].replace([np.inf,-np.inf],np.nan).dropna(); y=df.loc[x.index,target].astype(int)
    if len(x)>n:
        rng=np.random.default_rng(seed); ii=np.sort(rng.choice(len(x),n,replace=False)); x=x.iloc[ii]; y=y.iloc[ii]
    return x,y


def fit(x,y):
    return Pipeline([('scale',StandardScaler()),('m',LogisticRegression(C=.5,max_iter=250,class_weight='balanced'))]).fit(x,y)


def pred(m,df):
    x=df[b.FEATURES].replace([np.inf,-np.inf],np.nan); ok=x.notna().all(axis=1); p=np.full(len(df),np.nan)
    if ok.any(): p[ok.to_numpy()]=m.predict_proba(x.loc[ok])[:,1]
    return p


def safe_auc(y,p):
    ok=np.isfinite(y)&np.isfinite(p)
    return float(roc_auc_score(y[ok].astype(int),p[ok])) if ok.sum()>1 and len(np.unique(y[ok]))>1 else None


def main():
    fs=[]; logs=[]
    for day,split in DATES:
        print(ASSET,day,split,flush=True); x=b.build_day(day,split)
        if x.empty: continue
        logs.append({'day':day,'split':split,'rows':len(x),'spread_bp':float(x.spread_bps.median()),'collector_lag_ms':float(x.collector_lag_ms.median())})
        fs.append(x[KEEP].copy())
    df=labels(pd.concat(fs,ignore_index=True)); train=df[df.split.eq('TRAIN')]
    cls=[]; trades=[]
    for name,h in HORIZONS.items():
        xb,yb=xy(train,f'ybig_{name}',seed=101+h); xd,yd=xy(train,f'ydir_{name}',seed=201+h)
        mb=fit(xb,yb); md=fit(xd,yd); pb=pred(mb,df); pdn=pred(md,df)
        for split in ['VALID','TEST']:
            q=df.split.eq(split).to_numpy(); ybig=df[f'ybig_{name}'].to_numpy(); ydir=df[f'ydir_{name}'].to_numpy(); ob=q&np.isfinite(ybig)&np.isfinite(pb); od=q&np.isfinite(ydir)&np.isfinite(pdn)
            cls.append({'asset':ASSET,'split':split,'horizon':name,'rows':int(q.sum()),'bigmove_rate':float(np.nanmean(ybig[q])),
                        'big_auc':safe_auc(ybig[ob],pb[ob]),'dirbig_auc':safe_auc(ydir[od],pdn[od]),
                        'dirbig_acc':float(accuracy_score(ydir[od].astype(int),pdn[od]>=.5)) if od.any() else None,'dirbig_n':int(od.sum())})
        valid=np.isfinite(pb)&np.isfinite(pdn); act=np.zeros(len(df),dtype=np.int8); act[valid&(pb>=BIG_TH)&(pdn>=DIR_TH)]=1; act[valid&(pb>=BIG_TH)&(pdn<=1-DIR_TH)]=-1
        for lat in [0,1,3]:
            for fee in [0,2,5]:
                r=b.simulate(df,act,h,lat,fee,'TEST'); r.update({'horizon':name,'model':'LOGIT_FROZEN','dir_threshold':DIR_TH,'big_threshold':BIG_TH}); trades.append(r)
    c=pd.DataFrame(cls); t=pd.DataFrame(trades); c.to_csv(OUT/'classification.csv',index=False); t.to_csv(OUT/'trading.csv',index=False); (OUT/'data_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    p=t[(t.latency_ms==100)&(t.fee_bp_roundtrip.isin([0,5]))]
    lines=[f'# HYDRA Micro frozen historical replication 2025 — {ASSET}','',
           'Architecture and thresholds frozen before this run. 2025 Jan-Jun TRAIN, Jul-Aug VALID, Sep-Dec TEST.','Label: >=5bp absolute mid move; direction conditional on large move. 100ms grid, microsecond raw quotes/trades.','',
           '## TEST classification',c[c.split.eq('TEST')].to_markdown(index=False),'','## TEST economics (100ms latency)',p.to_markdown(index=False)]
    (OUT/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8'); print('\n'.join(lines),flush=True)

if __name__=='__main__': main()
