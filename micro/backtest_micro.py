from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ASSET = os.environ.get("ASSET", "BTCUSDT")
OUT = Path(f"micro_results_{ASSET}")
OUT.mkdir(exist_ok=True)

BASE = "https://datasets.tardis.dev/v1/binance-futures"
DATES = [
    ("2026-03-01", "TRAIN"),
    ("2026-04-01", "TRAIN"),
    ("2026-05-01", "TRAIN"),
    ("2026-06-01", "VALID"),
    ("2026-07-01", "TEST"),
    ("2026-08-01", "TEST"),
]
WINDOWS_H = [(0, 2), (8, 10), (16, 18)]
STEP_US = 100_000
HORIZONS = {"500ms": 5, "1s": 10, "2s": 20, "5s": 50}
FEATURES = [
    "qi", "spread_bps", "log_depth_ratio", "depth_notional_log",
    "ofi_100ms", "ofi_1s", "ofi_5s",
    "trade_imb_100ms", "trade_imb_1s", "trade_imb_5s",
    "trade_rate_1s", "trade_rate_5s", "quote_rate_1s",
    "ret_100ms_bps", "ret_500ms_bps", "ret_1s_bps", "ret_5s_bps",
    "rv_1s_bps", "rv_5s_bps", "rv_30s_bps", "flow_z",
]
PRIMARY_DIR_TH = 0.65
PRIMARY_VOL_TH = 0.60
TRAIN_MAX = 350_000

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


def download(kind: str, day: str) -> Path:
    yyyy, mm, dd = day.split("-")
    url = f"{BASE}/{kind}/{yyyy}/{mm}/{dd}/{ASSET}.csv.gz"
    path = Path("/tmp") / f"{ASSET}_{day}_{kind}.csv.gz"
    if path.exists() and path.stat().st_size > 100:
        return path
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            shutil.copyfileobj(r.raw, f, length=1024 * 1024)
    return path


def day_start_us(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp() * 1_000_000)


def load_windows(path: Path, kind: str, day: str) -> pd.DataFrame:
    base = day_start_us(day)
    usecols = (
        ["timestamp", "local_timestamp", "ask_amount", "ask_price", "bid_price", "bid_amount"]
        if kind == "quotes"
        else ["timestamp", "local_timestamp", "side", "price", "amount"]
    )
    kept = []
    for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=400_000):
        lt = pd.to_numeric(chunk["local_timestamp"], errors="coerce")
        for w, (h0, h1) in enumerate(WINDOWS_H):
            lo = base + h0 * 3_600_000_000
            hi = base + h1 * 3_600_000_000
            mask = (lt >= lo) & (lt < hi)
            if mask.any():
                x = chunk.loc[mask].copy()
                x["window"] = w
                kept.append(x)
    if not kept:
        return pd.DataFrame(columns=usecols + ["window"])
    out = pd.concat(kept, ignore_index=True)
    out["local_timestamp"] = pd.to_numeric(out["local_timestamp"], errors="coerce").astype("int64")
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").astype("int64")
    return out.sort_values(["window", "local_timestamp"]).reset_index(drop=True)


def ofi_top(bid, bqty, ask, aqty):
    pb = bid.shift(1)
    pbq = bqty.shift(1)
    pa = ask.shift(1)
    paq = aqty.shift(1)
    bid_e = np.select([bid > pb, bid < pb], [bqty, -pbq], default=bqty - pbq)
    ask_e = np.select([ask < pa, ask > pa], [-aqty, paq], default=-(aqty - paq))
    return pd.Series(bid_e + ask_e, index=bid.index, dtype="float64").fillna(0.0)


def build_window(q: pd.DataFrame, t: pd.DataFrame, day: str, win: int) -> pd.DataFrame:
    h0, h1 = WINDOWS_H[win]
    lo = day_start_us(day) + h0 * 3_600_000_000
    n = int((h1 - h0) * 3600 * 1_000_000 // STEP_US)
    q = q[q["window"] == win].copy()
    t = t[t["window"] == win].copy()
    if q.empty:
        return pd.DataFrame()

    q["bucket"] = ((q["local_timestamp"] - lo) // STEP_US).astype("int64")
    q["collector_lag_us"] = q["local_timestamp"] - q["timestamp"]
    q_last = q.groupby("bucket", sort=True).agg(
        ask_amount=("ask_amount", "last"), ask_price=("ask_price", "last"),
        bid_price=("bid_price", "last"), bid_amount=("bid_amount", "last"),
        collector_lag_us=("collector_lag_us", "last"), quote_count=("local_timestamp", "size"),
    )

    idx = pd.RangeIndex(n, name="bucket")
    g = q_last.reindex(idx)
    qcols = ["ask_amount", "ask_price", "bid_price", "bid_amount", "collector_lag_us"]
    g[qcols] = g[qcols].ffill()
    g["quote_count"] = g["quote_count"].fillna(0.0)

    if not t.empty:
        t["bucket"] = ((t["local_timestamp"] - lo) // STEP_US).astype("int64")
        t["price"] = pd.to_numeric(t["price"], errors="coerce")
        t["amount"] = pd.to_numeric(t["amount"], errors="coerce")
        sign = np.where(t["side"].astype(str).str.lower().eq("buy"), 1.0, -1.0)
        t["notional"] = t["price"] * t["amount"]
        t["signed_notional"] = t["notional"] * sign
        tg = t.groupby("bucket", sort=True).agg(
            signed_notional=("signed_notional", "sum"), total_notional=("notional", "sum"),
            trade_count=("local_timestamp", "size"),
        ).reindex(idx).fillna(0.0)
        g = g.join(tg)
    else:
        g["signed_notional"] = 0.0
        g["total_notional"] = 0.0
        g["trade_count"] = 0.0

    g = g.dropna(subset=["ask_price", "bid_price", "ask_amount", "bid_amount"]).copy()
    if g.empty:
        return g
    for c in ["ask_price", "bid_price", "ask_amount", "bid_amount"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    g["mid"] = (g["ask_price"] + g["bid_price"]) / 2.0
    g["spread"] = g["ask_price"] - g["bid_price"]
    depth = g["bid_amount"] + g["ask_amount"]
    g["qi"] = (g["bid_amount"] - g["ask_amount"]) / (depth + 1e-12)
    g["spread_bps"] = g["spread"] / g["mid"] * 10_000
    g["log_depth_ratio"] = np.log((g["bid_amount"] + 1e-12) / (g["ask_amount"] + 1e-12))
    g["depth_notional_log"] = np.log1p(depth * g["mid"])

    e = ofi_top(g["bid_price"], g["bid_amount"], g["ask_price"], g["ask_amount"])
    avg_depth_1s = depth.rolling(10, min_periods=3).mean()
    avg_depth_5s = depth.rolling(50, min_periods=10).mean()
    g["ofi_100ms"] = e / (depth + 1e-12)
    g["ofi_1s"] = e.rolling(10, min_periods=3).sum() / (avg_depth_1s + 1e-12)
    g["ofi_5s"] = e.rolling(50, min_periods=10).sum() / (avg_depth_5s + 1e-12)

    sn = g["signed_notional"]
    tn = g["total_notional"]
    g["trade_imb_100ms"] = sn / (tn + 1e-12)
    sn1 = sn.rolling(10, min_periods=1).sum()
    tn1 = tn.rolling(10, min_periods=1).sum()
    sn5 = sn.rolling(50, min_periods=1).sum()
    tn5 = tn.rolling(50, min_periods=1).sum()
    g["trade_imb_1s"] = sn1 / (tn1 + 1e-12)
    g["trade_imb_5s"] = sn5 / (tn5 + 1e-12)
    g["trade_rate_1s"] = g["trade_count"].rolling(10, min_periods=1).sum()
    g["trade_rate_5s"] = g["trade_count"].rolling(50, min_periods=1).sum()
    g["quote_rate_1s"] = g["quote_count"].rolling(10, min_periods=1).sum()

    logmid = np.log(g["mid"])
    r = logmid.diff()
    g["ret_100ms_bps"] = r * 10_000
    g["ret_500ms_bps"] = (logmid - logmid.shift(5)) * 10_000
    g["ret_1s_bps"] = (logmid - logmid.shift(10)) * 10_000
    g["ret_5s_bps"] = (logmid - logmid.shift(50)) * 10_000
    g["rv_1s_bps"] = np.sqrt(r.pow(2).rolling(10, min_periods=3).sum()) * 10_000
    g["rv_5s_bps"] = np.sqrt(r.pow(2).rolling(50, min_periods=10).sum()) * 10_000
    g["rv_30s_bps"] = np.sqrt(r.pow(2).rolling(300, min_periods=50).sum()) * 10_000

    prior_mean = sn1.shift(1).rolling(600, min_periods=100).mean()
    prior_std = sn1.shift(1).rolling(600, min_periods=100).std()
    g["flow_z"] = (sn1 - prior_mean) / (prior_std + 1e-12)
    g["day"] = day
    g["window"] = win
    g["collector_lag_ms"] = g["collector_lag_us"] / 1000.0

    for name, h in HORIZONS.items():
        future = g["mid"].shift(-h)
        delta = future - g["mid"]
        threshold = g["spread"] / 2.0
        moved = delta.abs() >= threshold
        g[f"yvol_{name}"] = moved.astype("float")
        yd = pd.Series(np.nan, index=g.index)
        eligible = moved & future.notna()
        yd[eligible] = (delta[eligible] > 0).astype(int)
        g[f"ydir_{name}"] = yd

    return g.iloc[650:].reset_index(drop=True)


def build_day(day: str, split: str) -> pd.DataFrame:
    qpath = download("quotes", day)
    tpath = download("trades", day)
    q = load_windows(qpath, "quotes", day)
    t = load_windows(tpath, "trades", day)
    frames = []
    for w in range(len(WINDOWS_H)):
        x = build_window(q, t, day, w)
        if not x.empty:
            x["split"] = split
            frames.append(x)
    qpath.unlink(missing_ok=True)
    tpath.unlink(missing_ok=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def finite_xy(df: pd.DataFrame, target: str):
    mask = df[target].notna()
    x = df.loc[mask, FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    y = df.loc[x.index, target].astype(int)
    return x, y


def subsample(x, y, n, seed):
    if len(x) <= n:
        return x, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(x), size=n, replace=False))
    return x.iloc[idx], y.iloc[idx]


def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
    except Exception:
        return None


def fit_models(train: pd.DataFrame, horizon: str):
    xv, yv = finite_xy(train, f"yvol_{horizon}")
    xd, yd = finite_xy(train, f"ydir_{horizon}")
    xv, yv = subsample(xv, yv, TRAIN_MAX, 11)
    xd, yd = subsample(xd, yd, TRAIN_MAX, 12)
    models = {}
    models["LOGIT_VOL"] = Pipeline([
        ("scale", StandardScaler()),
        ("m", LogisticRegression(C=0.5, max_iter=250, class_weight="balanced")),
    ]).fit(xv, yv)
    models["LOGIT_DIR"] = Pipeline([
        ("scale", StandardScaler()),
        ("m", LogisticRegression(C=0.5, max_iter=250, class_weight="balanced")),
    ]).fit(xd, yd)
    xv2, yv2 = subsample(xv, yv, min(250_000, len(xv)), 13)
    xd2, yd2 = subsample(xd, yd, min(250_000, len(xd)), 14)
    models["HGB_VOL"] = HistGradientBoostingClassifier(
        max_iter=70, learning_rate=0.07, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=2.0, random_state=21,
    ).fit(xv2, yv2)
    models["HGB_DIR"] = HistGradientBoostingClassifier(
        max_iter=70, learning_rate=0.07, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=2.0, random_state=22,
    ).fit(xd2, yd2)
    return models


def predict_all(model, df):
    x = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    valid = x.notna().all(axis=1)
    p = np.full(len(df), np.nan)
    if valid.any():
        p[valid.to_numpy()] = model.predict_proba(x.loc[valid])[:, 1]
    return p


def classification_row(df, horizon, model_name, pdir, pvol, split):
    q = df["split"].eq(split).to_numpy()
    yv = df[f"yvol_{horizon}"].to_numpy()
    yd = df[f"ydir_{horizon}"].to_numpy()
    mv = q & np.isfinite(pvol) & np.isfinite(yv)
    md = q & np.isfinite(pdir) & np.isfinite(yd)
    return {
        "asset": ASSET, "split": split, "horizon": horizon, "model": model_name,
        "rows": int(q.sum()), "movement_rate": float(np.nanmean(yv[q])) if q.any() else None,
        "vol_auc": safe_auc(yv[mv].astype(int), pvol[mv]) if mv.any() else None,
        "vol_brier": float(brier_score_loss(yv[mv].astype(int), pvol[mv])) if mv.any() else None,
        "dir_auc": safe_auc(yd[md].astype(int), pdir[md]) if md.any() else None,
        "dir_acc_50": float(accuracy_score(yd[md].astype(int), pdir[md] >= 0.5)) if md.any() else None,
        "dir_n": int(md.sum()),
    }


def pf(arr):
    arr = np.asarray(arr, float)
    pos, neg = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return float(pos / neg) if neg > 0 else None


def mdd(arr):
    arr = np.asarray(arr, float)
    if not len(arr): return None
    eq = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1))


def tstat(arr):
    arr = np.asarray(arr, float)
    if len(arr) < 2: return None
    sd = np.std(arr, ddof=1)
    return float(np.mean(arr) / (sd / np.sqrt(len(arr)))) if np.isfinite(sd) and sd > 0 else None


def simulate(df, action, horizon_steps, latency_steps, fee_bp, split="TEST"):
    gross, directions, entry_days = [], [], []
    for (day, win), idxs in df.loc[df["split"].eq(split)].groupby(["day", "window"], sort=True).groups.items():
        idxs = np.asarray(list(idxs), dtype=int)
        local_action = action[idxs]
        next_free = 0
        for loc_i in np.flatnonzero(local_action != 0):
            if loc_i < next_free: continue
            entry = loc_i + 1 + latency_steps
            exit_i = loc_i + horizon_steps
            if entry >= exit_i or exit_i >= len(idxs): continue
            ge, gx = idxs[entry], idxs[exit_i]
            side = int(local_action[loc_i])
            if side > 0:
                entry_px, exit_px = float(df.at[ge, "ask_price"]), float(df.at[gx, "bid_price"])
                r = exit_px / entry_px - 1.0
            else:
                entry_px, exit_px = float(df.at[ge, "bid_price"]), float(df.at[gx, "ask_price"])
                r = (entry_px - exit_px) / entry_px
            if np.isfinite(r):
                gross.append(r); directions.append(side); entry_days.append(day); next_free = exit_i + 1
    gross = np.asarray(gross, float)
    net = gross - fee_bp / 10_000
    return {
        "asset": ASSET, "split": split, "horizon_steps": horizon_steps,
        "latency_ms": latency_steps * 100, "fee_bp_roundtrip": fee_bp, "trades": int(len(net)),
        "long_share": float((np.asarray(directions) > 0).mean()) if len(net) else None,
        "gross_mean_bp": float(np.mean(gross) * 10_000) if len(net) else None,
        "net_mean_bp": float(np.mean(net) * 10_000) if len(net) else None,
        "median_net_bp": float(np.median(net) * 10_000) if len(net) else None,
        "win_rate": float((net > 0).mean()) if len(net) else None,
        "profit_factor": pf(net) if len(net) else None, "t_stat": tstat(net) if len(net) else None,
        "break_even_fee_bp": float(np.mean(gross) * 10_000) if len(net) else None,
        "sum_net_bp": float(np.sum(net) * 10_000) if len(net) else None,
        "compounded_return": float(np.prod(1 + net) - 1) if len(net) else None,
        "mdd": mdd(net) if len(net) else None, "entry_days": len(set(entry_days)),
    }


def baseline_row(df, horizon):
    q = df[df["split"].eq("TEST")]
    yd = q[f"ydir_{horizon}"]
    mask = yd.notna()
    if not mask.any(): return {"asset": ASSET, "split": "TEST", "horizon": horizon, "model": "BASELINE_SIGNS"}
    y = yd[mask].astype(int).to_numpy()
    qi = (q.loc[mask, "qi"].to_numpy() > 0).astype(int)
    ti = (q.loc[mask, "trade_imb_1s"].to_numpy() > 0).astype(int)
    return {"asset": ASSET, "split": "TEST", "horizon": horizon, "model": "BASELINE_SIGNS",
            "qi_sign_accuracy": float((qi == y).mean()),
            "trade_imb_1s_sign_accuracy": float((ti == y).mean()), "dir_n": int(len(y))}


def main():
    frames, data_log = [], []
    for day, split in DATES:
        print(f"[{ASSET}] building {day} {split}", flush=True)
        x = build_day(day, split)
        if x.empty:
            data_log.append({"day": day, "split": split, "rows": 0}); continue
        data_log.append({"day": day, "split": split, "rows": int(len(x)),
                         "median_collector_lag_ms": float(x["collector_lag_ms"].median()),
                         "p95_collector_lag_ms": float(x["collector_lag_ms"].quantile(.95)),
                         "median_spread_bps": float(x["spread_bps"].median())})
        frames.append(x)
    if not frames: raise RuntimeError("no features")
    df = pd.concat(frames, ignore_index=True)
    (OUT / "data_log.json").write_text(json.dumps(data_log, indent=2), encoding="utf-8")
    train = df[df["split"].eq("TRAIN")].copy()
    cls, trade_rows, sensitivity = [], [], []

    for hname, hsteps in HORIZONS.items():
        print(f"[{ASSET}] fitting {hname}", flush=True)
        models = fit_models(train, hname)
        for family in ["LOGIT", "HGB"]:
            pvol = predict_all(models[f"{family}_VOL"], df)
            pdir = predict_all(models[f"{family}_DIR"], df)
            for split in ["VALID", "TEST"]:
                cls.append(classification_row(df, hname, family, pdir, pvol, split))
            valid = np.isfinite(pdir) & np.isfinite(pvol)
            action = np.zeros(len(df), dtype=np.int8)
            action[valid & (pvol >= PRIMARY_VOL_TH) & (pdir >= PRIMARY_DIR_TH)] = 1
            action[valid & (pvol >= PRIMARY_VOL_TH) & (pdir <= 1 - PRIMARY_DIR_TH)] = -1
            for latency in [0, 1, 3]:
                for fee in [0, 2, 5, 10]:
                    rec = simulate(df, action, hsteps, latency, fee, "TEST")
                    rec.update({"horizon": hname, "model": family,
                                "dir_threshold": PRIMARY_DIR_TH, "vol_threshold": PRIMARY_VOL_TH})
                    trade_rows.append(rec)
            for dth in [0.55, 0.60, 0.65, 0.70]:
                for vth in [0.50, 0.60, 0.70]:
                    a = np.zeros(len(df), dtype=np.int8)
                    a[valid & (pvol >= vth) & (pdir >= dth)] = 1
                    a[valid & (pvol >= vth) & (pdir <= 1 - dth)] = -1
                    rec = simulate(df, a, hsteps, 1, 5, "TEST")
                    rec.update({"horizon": hname, "model": family, "dir_threshold": dth, "vol_threshold": vth})
                    sensitivity.append(rec)
        cls.append(baseline_row(df, hname))

    cls_df, tr_df, sens_df = pd.DataFrame(cls), pd.DataFrame(trade_rows), pd.DataFrame(sensitivity)
    cls_df.to_csv(OUT / "classification.csv", index=False)
    tr_df.to_csv(OUT / "trading_primary.csv", index=False)
    sens_df.to_csv(OUT / "threshold_sensitivity.csv", index=False)
    primary = tr_df[(tr_df["latency_ms"] == 100) & (tr_df["fee_bp_roundtrip"] == 5)].copy()
    best_case = tr_df[(tr_df["latency_ms"] == 0) & (tr_df["fee_bp_roundtrip"] == 0)].copy()
    harsh = tr_df[(tr_df["latency_ms"] == 300) & (tr_df["fee_bp_roundtrip"] == 10)].copy()
    lines = [
        f"# HYDRA Micro OOS — {ASSET}", "",
        "Raw source: Tardis free first-day-of-month Binance Futures quotes + trades, microsecond timestamp/local_timestamp.",
        "Observation grid: 100ms. Three 2-hour UTC windows/day. Features use local_timestamp (observable arrival time).",
        "Chronology: Mar-May TRAIN, Jun VALID, Jul-Aug TEST. TEST is not used for fitting.",
        "Primary gate: p(direction)>=0.65 or <=0.35 AND p(move)>=0.60.",
        "Execution starts no earlier than the next 100ms bucket. Spread is paid through ask/bid; explicit fee stress is additional.", "",
        "## Data", pd.DataFrame(data_log).to_markdown(index=False), "",
        "## Classification TEST", cls_df[cls_df["split"].eq("TEST")].to_markdown(index=False), "",
        "## Primary economics: 100ms extra latency + 5bp round-trip fee",
        primary[["horizon","model","trades","gross_mean_bp","net_mean_bp","win_rate","profit_factor","t_stat","break_even_fee_bp","sum_net_bp","compounded_return","mdd"]].to_markdown(index=False), "",
        "## Best-case upper bound: 0ms extra latency + 0bp explicit fee",
        best_case[["horizon","model","trades","gross_mean_bp","win_rate","profit_factor","t_stat","break_even_fee_bp"]].to_markdown(index=False), "",
        "## Harsh stress: 300ms extra latency + 10bp round-trip fee",
        harsh[["horizon","model","trades","net_mean_bp","win_rate","profit_factor","t_stat","sum_net_bp","mdd"]].to_markdown(index=False),
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
