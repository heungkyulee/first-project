from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import backtest_micro as b

ASSET = b.ASSET
OUT = Path(f"micro_econ_{ASSET}")
OUT.mkdir(exist_ok=True)
HORIZONS = {"5s": 50, "10s": 100, "30s": 300, "60s": 600}
MOVE_BP = 5.0
DIR_TH = 0.65
BIG_TH = 0.60
TRAIN_MAX = 350_000


def add_econ_labels(df):
    out = []
    for (day, win), g in df.groupby(["day", "window"], sort=False):
        g = g.copy()
        for name, h in HORIZONS.items():
            f = g["mid"].shift(-h)
            retbp = (f / g["mid"] - 1.0) * 10_000
            g[f"future_ret_bp_{name}"] = retbp
            big = retbp.abs() >= MOVE_BP
            yb = pd.Series(np.nan, index=g.index)
            yb[f.notna()] = big[f.notna()].astype(int)
            g[f"ybig_{name}"] = yb
            yd = pd.Series(np.nan, index=g.index)
            eligible = big & f.notna()
            yd[eligible] = (retbp[eligible] > 0).astype(int)
            g[f"ydirbig_{name}"] = yd
        out.append(g)
    return pd.concat(out).sort_index()


def xy(df, target):
    mask = df[target].notna()
    x = df.loc[mask, b.FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    y = df.loc[x.index, target].astype(int)
    return x, y


def subsample(x, y, n, seed):
    if len(x) <= n: return x, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(x), size=n, replace=False))
    return x.iloc[idx], y.iloc[idx]


def fit(train, h):
    xb, yb = xy(train, f"ybig_{h}")
    xd, yd = xy(train, f"ydirbig_{h}")
    xb, yb = subsample(xb, yb, TRAIN_MAX, 31)
    xd, yd = subsample(xd, yd, TRAIN_MAX, 32)
    m = {}
    m["LOGIT_BIG"] = Pipeline([("scale", StandardScaler()), ("m", LogisticRegression(C=.5,max_iter=250,class_weight="balanced"))]).fit(xb,yb)
    m["LOGIT_DIR"] = Pipeline([("scale", StandardScaler()), ("m", LogisticRegression(C=.5,max_iter=250,class_weight="balanced"))]).fit(xd,yd)
    xb2,yb2=subsample(xb,yb,min(250_000,len(xb)),33); xd2,yd2=subsample(xd,yd,min(250_000,len(xd)),34)
    m["HGB_BIG"] = HistGradientBoostingClassifier(max_iter=80,learning_rate=.07,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=2,random_state=41).fit(xb2,yb2)
    m["HGB_DIR"] = HistGradientBoostingClassifier(max_iter=80,learning_rate=.07,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=2,random_state=42).fit(xd2,yd2)
    return m


def predict(model, df):
    x=df[b.FEATURES].replace([np.inf,-np.inf],np.nan); valid=x.notna().all(axis=1); p=np.full(len(df),np.nan)
    if valid.any(): p[valid.to_numpy()]=model.predict_proba(x.loc[valid])[:,1]
    return p


def auc(y,p):
    m=np.isfinite(y)&np.isfinite(p)
    return float(roc_auc_score(y[m].astype(int),p[m])) if m.sum()>1 and len(np.unique(y[m]))>1 else None


def classify(df,h,family,pdir,pbig,split):
    m=df["split"].eq(split).to_numpy(); yb=df[f"ybig_{h}"].to_numpy(); yd=df[f"ydirbig_{h}"].to_numpy()
    mb=m&np.isfinite(yb)&np.isfinite(pbig); md=m&np.isfinite(yd)&np.isfinite(pdir)
    return {"asset":ASSET,"split":split,"horizon":h,"model":family,"rows":int(m.sum()),
            "bigmove_rate":float(np.nanmean(yb[m])),"big_auc":auc(yb[mb],pbig[mb]),
            "big_brier":float(brier_score_loss(yb[mb].astype(int),pbig[mb])) if mb.any() else None,
            "dirbig_auc":auc(yd[md],pdir[md]),"dirbig_acc":float(accuracy_score(yd[md].astype(int),pdir[md]>=.5)) if md.any() else None,
            "dirbig_n":int(md.sum())}


def main():
    frames=[]; log=[]
    for day,split in b.DATES:
        print(f"[{ASSET}] build {day}",flush=True); x=b.build_day(day,split)
        if x.empty: continue
        log.append({"day":day,"split":split,"rows":len(x),"median_spread_bp":float(x.spread_bps.median()),"median_collector_lag_ms":float(x.collector_lag_ms.median())})
        frames.append(x)
    df=add_econ_labels(pd.concat(frames,ignore_index=True)); train=df[df.split.eq("TRAIN")].copy(); cls=[]; trades=[]; sens=[]
    for h,steps in HORIZONS.items():
        print(f"[{ASSET}] economic horizon {h}",flush=True); mods=fit(train,h)
        for family in ["LOGIT","HGB"]:
            pbig=predict(mods[f"{family}_BIG"],df); pdir=predict(mods[f"{family}_DIR"],df)
            for split in ["VALID","TEST"]: cls.append(classify(df,h,family,pdir,pbig,split))
            valid=np.isfinite(pbig)&np.isfinite(pdir); act=np.zeros(len(df),dtype=np.int8)
            act[valid&(pbig>=BIG_TH)&(pdir>=DIR_TH)]=1; act[valid&(pbig>=BIG_TH)&(pdir<=1-DIR_TH)]=-1
            for lat in [0,1,3]:
                for fee in [0,2,5,10]:
                    r=b.simulate(df,act,steps,lat,fee,"TEST"); r.update({"horizon":h,"model":family,"dir_threshold":DIR_TH,"big_threshold":BIG_TH,"move_label_bp":MOVE_BP}); trades.append(r)
            for dth in [.60,.65,.70,.75,.80,.85,.90]:
                for bth in [.50,.60,.70,.80,.90]:
                    a=np.zeros(len(df),dtype=np.int8); a[valid&(pbig>=bth)&(pdir>=dth)]=1; a[valid&(pbig>=bth)&(pdir<=1-dth)]=-1
                    r=b.simulate(df,a,steps,1,5,"TEST"); r.update({"horizon":h,"model":family,"dir_threshold":dth,"big_threshold":bth,"move_label_bp":MOVE_BP}); sens.append(r)
    c=pd.DataFrame(cls); t=pd.DataFrame(trades); s=pd.DataFrame(sens)
    c.to_csv(OUT/"classification.csv",index=False); t.to_csv(OUT/"trading.csv",index=False); s.to_csv(OUT/"sensitivity.csv",index=False)
    (OUT/"data_log.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
    primary=t[(t.latency_ms==100)&(t.fee_bp_roundtrip==5)]; bestsens=s.sort_values("net_mean_bp",ascending=False).head(20)
    lines=[f"# HYDRA Micro economic-move OOS — {ASSET}","","Label: absolute future mid move >= 5bp, then direction conditional on such a move.","TRAIN Mar-May, VALID Jun, untouched TEST Jul-Aug. Raw events use microsecond local_timestamp.","Execution starts next 100ms bucket; bid/ask spread is paid, then explicit fee stress is subtracted.","","## Classification TEST",c[c.split.eq("TEST")].to_markdown(index=False),"","## Primary: 100ms extra latency + 5bp RT",primary.to_markdown(index=False),"","## Best threshold sensitivities on TEST (diagnostic only, NOT a promoted rule)",bestsens.to_markdown(index=False)]
    (OUT/"SUMMARY.md").write_text("\n".join(lines),encoding="utf-8"); print("\n".join(lines),flush=True)

if __name__=="__main__": main()
