from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import backtest_micro as b
import replicate_2025 as r

ASSET=b.ASSET
OUT=Path(f'micro_edge25_{ASSET}'); OUT.mkdir(exist_ok=True)
QUANTILES=[0.50,0.70,0.80,0.90,0.95,0.97,0.98,0.99,0.995,0.999]
MIN_VALID_TRADES=30
LATENCY_STEPS=1
FEE_BP=5


def main():
    fs=[]; logs=[]
    for day,split in r.DATES:
        print(ASSET,day,split,flush=True)
        x=b.build_day(day,split)
        if x.empty: continue
        logs.append({'day':day,'split':split,'rows':len(x),'spread_bp':float(x.spread_bps.median()),'collector_lag_ms':float(x.collector_lag_ms.median())})
        fs.append(x[r.KEEP].copy())
    df=r.labels(pd.concat(fs,ignore_index=True)); train=df[df.split.eq('TRAIN')]
    selections=[]; test_rows=[]; candidate_rows=[]
    for hname,hsteps in r.HORIZONS.items():
        xb,yb=r.xy(train,f'ybig_{hname}',seed=501+hsteps); xd,yd=r.xy(train,f'ydir_{hname}',seed=601+hsteps)
        mb=r.fit(xb,yb); md=r.fit(xd,yd); pbig=r.pred(mb,df); pdir=r.pred(md,df)
        edge=pbig*(2*pdir-1)
        valid_mask=df.split.eq('VALID').to_numpy() & np.isfinite(edge)
        abs_valid=np.abs(edge[valid_mask])
        cand=[]
        for q in QUANTILES:
            thr=float(np.quantile(abs_valid,q))
            action=np.zeros(len(df),dtype=np.int8)
            ok=np.isfinite(edge)&(np.abs(edge)>=thr)
            action[ok & (edge>0)]=1; action[ok & (edge<0)]=-1
            vr=b.simulate(df,action,hsteps,LATENCY_STEPS,FEE_BP,'VALID')
            vr.update({'asset':ASSET,'horizon':hname,'quantile':q,'edge_threshold':thr,'phase':'VALID_SELECT'})
            candidate_rows.append(vr); cand.append(vr)
        eligible=[x for x in cand if x['trades']>=MIN_VALID_TRADES and x['net_mean_bp'] is not None]
        chosen=max(eligible,key=lambda z:z['net_mean_bp']) if eligible else max(cand,key=lambda z:(z['trades']>=1,z['net_mean_bp'] if z['net_mean_bp'] is not None else -1e99))
        thr=chosen['edge_threshold']
        action=np.zeros(len(df),dtype=np.int8); ok=np.isfinite(edge)&(np.abs(edge)>=thr); action[ok&(edge>0)]=1; action[ok&(edge<0)]=-1
        tr=b.simulate(df,action,hsteps,LATENCY_STEPS,FEE_BP,'TEST')
        tr.update({'asset':ASSET,'horizon':hname,'selected_quantile':chosen['quantile'],'edge_threshold':thr,'valid_trades':chosen['trades'],'valid_net_mean_bp':chosen['net_mean_bp'],'valid_pf':chosen['profit_factor'],'test_phase':'FROZEN_TEST'})
        selections.append({'asset':ASSET,'horizon':hname,'selected_quantile':chosen['quantile'],'edge_threshold':thr,'valid_trades':chosen['trades'],'valid_net_mean_bp':chosen['net_mean_bp'],'valid_pf':chosen['profit_factor']})
        test_rows.append(tr)
    pd.DataFrame(candidate_rows).to_csv(OUT/'validation_candidates.csv',index=False); pd.DataFrame(selections).to_csv(OUT/'selection.csv',index=False); tt=pd.DataFrame(test_rows); tt.to_csv(OUT/'frozen_test.csv',index=False); (OUT/'data_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    lines=[f'# HYDRA Micro validation-selected edge replication 2025 — {ASSET}','',
           'Architecture fixed before this run. Jan-Jun TRAIN, Jul-Aug VALID selects only the |P(big)*(2P(up|big)-1)| threshold, Sep-Dec frozen TEST.','Selection objective: highest VALID net mean after quoted-spread execution, 100ms added latency, 5bp round-trip explicit cost, minimum 30 non-overlapping trades.','TEST is not used to select the threshold.','','## Selected thresholds',pd.DataFrame(selections).to_markdown(index=False),'','## Frozen TEST',tt.to_markdown(index=False)]
    (OUT/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8'); print('\n'.join(lines),flush=True)

if __name__=='__main__': main()
