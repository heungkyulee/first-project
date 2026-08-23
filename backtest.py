from __future__ import annotations

import json
import math
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

OUT = Path('results')
OUT.mkdir(exist_ok=True)
TICKERS = ['005930','000660']


def jdump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def fetch_naver(probe_only=False):
    """Probe Naver's public intraday chart endpoints. No auth/cookies/secrets."""
    logs = []
    all_frames = []
    # Probe possible granularities. minute10 is confirmed by an open-source client;
    # minute5/minute1 are exploratory and may 404.
    for ticker in TICKERS:
        for gran in ['minute1','minute5','minute10']:
            url = f'https://api.stock.naver.com/chart/domestic/item/{ticker}/{gran}'
            # First try one recent trading day; then a longer historical range if data comes back.
            for start, end in [
                ('202608210900','202608211530'),
                ('202605270900','202605271530'),
                ('202601050900','202601051530'),
            ]:
                try:
                    r = requests.get(url, params={'startDateTime': start, 'endDateTime': end}, timeout=20,
                                     headers={'User-Agent':'Mozilla/5.0'})
                    rec = {'ticker':ticker,'granularity':gran,'start':start,'end':end,'status':r.status_code,
                           'url':r.url,'text_head':r.text[:250]}
                    if r.ok:
                        try:
                            p = r.json()
                            items = p.get('items') or p.get('chartData') or p.get('datas') or [] if isinstance(p,dict) else []
                            rec['rows'] = len(items) if isinstance(items,list) else None
                            rec['keys'] = list(p.keys()) if isinstance(p,dict) else None
                        except Exception as e:
                            rec['json_error'] = repr(e)
                    logs.append(rec)
                except Exception as e:
                    logs.append({'ticker':ticker,'granularity':gran,'start':start,'end':end,'error':repr(e)})
    jdump('naver_probe.json', logs)
    return logs


def download_hf():
    # Known public KOSPI 200 5-minute parquet from SCOTS dataset commit (May 27 2026).
    urls = [
        'https://huggingface.co/datasets/alansynn/scots/resolve/68a846241692d7c865c479b218d9d689cf9dd212/parquet/5min/kr/ohlcv.parquet?download=true',
        'https://huggingface.co/datasets/alansynn/scots/resolve/main/parquet/5min/kr/ohlcv.parquet?download=true',
    ]
    dest = Path('/tmp/kr_5min.parquet')
    logs=[]
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=60, allow_redirects=True,
                              headers={'User-Agent':'Mozilla/5.0'}) as r:
                logs.append({'url':url,'status':r.status_code,'final_url':r.url,
                             'content_length':r.headers.get('content-length'),'content_type':r.headers.get('content-type')})
                if not r.ok:
                    continue
                with dest.open('wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk: f.write(chunk)
                if dest.stat().st_size > 1_000_000:
                    jdump('hf_download.json', logs)
                    return dest
        except Exception as e:
            logs.append({'url':url,'error':repr(e)})
    jdump('hf_download.json', logs)
    raise RuntimeError('HF parquet download failed')


def load_hf_filtered(path: Path):
    import duckdb
    con = duckdb.connect()
    schema = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
    schema.to_csv(OUT/'hf_schema.csv', index=False)
    cols = schema['column_name'].tolist()
    lower = {c.lower():c for c in cols}
    sym = next((lower[k] for k in ['symbol','ticker','code','stock_code'] if k in lower), None)
    ts = next((lower[k] for k in ['timestamp','datetime','date_time','dt','time'] if k in lower), None)
    if not sym or not ts:
        raise RuntimeError(f'Could not identify symbol/timestamp columns: {cols}')
    # inventory
    inv = con.execute(f"SELECT CAST({sym} AS VARCHAR) symbol, min({ts}) min_ts, max({ts}) max_ts, count(*) n FROM read_parquet(?) WHERE CAST({sym} AS VARCHAR) IN ('005930','000660') GROUP BY 1 ORDER BY 1", [str(path)]).fetchdf()
    inv.to_csv(OUT/'hf_inventory.csv', index=False)
    # extract target rows
    q = f"SELECT * FROM read_parquet(?) WHERE CAST({sym} AS VARCHAR) IN ('005930','000660') ORDER BY {sym},{ts}"
    df = con.execute(q,[str(path)]).fetchdf()
    df.to_parquet(OUT/'targets_5min.parquet', index=False)
    return df, sym, ts


def normalize(df, symcol, tscol):
    d = df.copy()
    d['symbol'] = d[symcol].astype(str).str.extract(r'(\d{6})', expand=False).fillna(d[symcol].astype(str))
    t = pd.to_datetime(d[tscol], utc=True, errors='coerce')
    # If the source already parsed as tz-aware UTC, convert to KST. If parsing was naive but utc=True,
    # this may shift incorrectly; detect by observed session hours and compare an unshifted alternative.
    kst = t.dt.tz_convert('Asia/Seoul')
    h_kst = kst.dt.hour.dropna()
    # Most bars should land in 09..15 KST. If not, fall back to naive localization as KST.
    if len(h_kst) and ((h_kst>=9)&(h_kst<=15)).mean() < 0.5:
        raw = pd.to_datetime(d[tscol], errors='coerce')
        if getattr(raw.dt, 'tz', None) is None:
            kst = raw.dt.tz_localize('Asia/Seoul')
    d['ts_kst'] = kst
    # standard OHLCV names
    cmap = {c.lower():c for c in d.columns}
    for k in ['open','high','low','close','volume']:
        if k not in cmap: raise RuntimeError(f'Missing {k}; columns={list(d.columns)}')
        d[k] = pd.to_numeric(d[cmap[k]], errors='coerce')
    d = d.dropna(subset=['symbol','ts_kst','open','close']).sort_values(['symbol','ts_kst'])
    return d[['symbol','ts_kst','open','high','low','close','volume']]


def build_sessions(d):
    d=d.copy()
    d['date']=d['ts_kst'].dt.date
    d['hm']=d['ts_kst'].dt.hour*60+d['ts_kst'].dt.minute
    rows=[]
    for (sym, date), g in d.groupby(['symbol','date'], sort=True):
        g=g.sort_values('ts_kst')
        # regular session only
        rg=g[(g.hm>=540)&(g.hm<=930)]
        if len(rg)<20: continue
        first=rg.iloc[0]
        last=rg.iloc[-1]
        # exact/nearest <= target bar
        def at_or_before(minute):
            x=rg[rg.hm<=minute]
            if x.empty: return None
            return x.iloc[-1]
        b1400=at_or_before(14*60)
        b0935=at_or_before(9*60+35)
        if b1400 is None or b0935 is None: continue
        vwin=rg[(rg.hm>=14*60)&(rg.hm<15*60)]['volume'].sum()
        rows.append({
            'symbol':sym,'date':pd.Timestamp(date),
            'open':float(first.open),'close':float(last.close),
            'close1400':float(b1400.close),'close0935':float(b0935.close),
            'vol1400_1500':float(vwin), 'n_bars':len(rg),
            'first_time':first.ts_kst.isoformat(),'last_time':last.ts_kst.isoformat(),
            'bar1400_time':b1400.ts_kst.isoformat(),'bar0935_time':b0935.ts_kst.isoformat(),
        })
    s=pd.DataFrame(rows).sort_values(['symbol','date'])
    s.to_csv(OUT/'sessions.csv', index=False)
    return s


def pf(x):
    pos=x[x>0].sum(); neg=-x[x<0].sum()
    return float(pos/neg) if neg>0 else None


def mdd(returns):
    if len(returns)==0:return None
    eq=np.cumprod(1+np.asarray(returns,float)); peak=np.maximum.accumulate(eq)
    return float(np.min(eq/peak-1))


def tstat(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<2 or np.std(x,ddof=1)==0:return None
    return float(np.mean(x)/(np.std(x,ddof=1)/np.sqrt(len(x))))


def bootstrap_ci(x, n=20000, seed=42):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<2:return [None,None]
    rng=np.random.default_rng(seed)
    means=np.empty(n)
    for i in range(n): means[i]=rng.choice(x, size=len(x), replace=True).mean()
    return [float(np.quantile(means,.025)), float(np.quantile(means,.975))]


def permutation_p(signal, base, n=20000, seed=43):
    signal=np.asarray(signal,float); base=np.asarray(base,float)
    signal=signal[np.isfinite(signal)]; base=base[np.isfinite(base)]
    if len(signal)<2 or len(base)<len(signal): return None
    obs=signal.mean()-base.mean(); rng=np.random.default_rng(seed)
    c=0
    for _ in range(n):
        samp=rng.choice(base,size=len(signal),replace=False)
        if abs(samp.mean()-base.mean())>=abs(obs): c+=1
    return float((c+1)/(n+1))


def run_backtest(s):
    all_tr=[]; summaries=[]
    for sym,g in s.groupby('symbol'):
        g=g.sort_values('date').copy()
        # Reconstructed relative-volume feature: today's 14:00-15:00 volume divided by the
        # median of the PRIOR 20 sessions' same-window volume. The public CAND1 source exposes
        # the column name and threshold but not the upstream feature builder, so this is explicit
        # reconstruction, not claimed exact replication.
        g['vol_med20_prev']=g['vol1400_1500'].shift(1).rolling(20,min_periods=10).median()
        g['rel_v_w1400_1500_recon']=g['vol1400_1500']/g['vol_med20_prev']
        g['r_1400_close']=g['close']/g['close1400']-1
        g['next_date']=g['date'].shift(-1)
        g['next_open']=g['open'].shift(-1)
        g['next_c0935']=g['close0935'].shift(-1)
        g['next_morning_gross']=g['next_c0935']/g['next_open']-1
        # Require actual next observed trading session, not calendar next day.
        primary=(g['r_1400_close']<=-0.02)&(g['rel_v_w1400_1500_recon']<1.5)&g['next_open'].notna()&g['next_c0935'].notna()
        unfiltered=(g['r_1400_close']<=-0.02)&g['next_open'].notna()&g['next_c0935'].notna()
        for name,mask in [('PMCRASH_RECON',primary),('PMCRASH_NO_VOLUME',unfiltered)]:
            tr=g.loc[mask].copy()
            if tr.empty:
                summaries.append({'symbol':sym,'variant':name,'trades':0})
                continue
            for cost in [20,25,30,35,40,50]:
                net=tr['next_morning_gross']-cost/10000
                rec={
                    'symbol':sym,'variant':name,'cost_bp':cost,'trades':len(tr),
                    'win_rate':float((net>0).mean()),'mean_net':float(net.mean()),'median_net':float(net.median()),
                    'profit_factor':pf(net),'t_stat':tstat(net),'bootstrap_mean_ci95':bootstrap_ci(net),
                    'cum_trade_return':float(np.prod(1+net)-1),'mdd_trade_seq':mdd(net),
                    'mean_gross':float(tr['next_morning_gross'].mean()),
                    'unconditional_next_morning_mean':float(g['next_morning_gross'].dropna().mean()),
                    'permutation_p_abs':permutation_p(tr['next_morning_gross'].dropna(), g['next_morning_gross'].dropna()),
                    'date_min':str(tr['date'].min().date()),'date_max':str(tr['date'].max().date()),
                    'data_date_min':str(g['date'].min().date()),'data_date_max':str(g['date'].max().date()),
                }
                summaries.append(rec)
            tr['variant']=name
            all_tr.append(tr)
        # Buy & hold benchmark over source coverage
        bh=float(g.iloc[-1]['close']/g.iloc[0]['open']-1) if len(g)>1 else None
        summaries.append({'symbol':sym,'variant':'BUY_HOLD_SOURCE_WINDOW','trades':None,'return':bh,
                          'data_date_min':str(g['date'].min().date()),'data_date_max':str(g['date'].max().date())})
    # Combined equal-weight-by-entry-date primary strategy
    if all_tr:
        trades=pd.concat(all_tr,ignore_index=True)
        trades.to_csv(OUT/'trades.csv',index=False)
        prim=trades[trades.variant=='PMCRASH_RECON'].copy()
        if not prim.empty:
            for cost in [20,25,30,35,40,50]:
                prim[f'net{cost}']=prim['next_morning_gross']-cost/10000
                daily=prim.groupby('next_date')[f'net{cost}'].mean().sort_index()
                summaries.append({'symbol':'COMBINED_EW','variant':'PMCRASH_RECON','cost_bp':cost,
                                  'trades':len(prim),'entry_days':len(daily),'mean_daily_net':float(daily.mean()),
                                  'cum_return':float(np.prod(1+daily)-1),'mdd':mdd(daily),'t_stat_daily':tstat(daily),
                                  'bootstrap_mean_daily_ci95':bootstrap_ci(daily)})
    pd.DataFrame(summaries).to_json(OUT/'summary.json',orient='records',force_ascii=False,indent=2)
    return summaries


def write_md(summaries, sessions):
    sdf=pd.DataFrame(summaries)
    lines=['# Samsung / SK hynix intraday backtest\n',
           '\nData source: public SCOTS KOSPI200 5-minute KIS parquet plus Naver endpoint probe.\n',
           '\nPrimary signal: 14:00→close <= -2%, reconstructed relative volume <1.5; buy next open, sell 09:35.\n',
           '\nImportant: relative-volume builder is reconstructed from prior-20-session median because the public CAND1 repository exposes the threshold/column but not its upstream feature-builder. `PMCRASH_NO_VOLUME` is included as a no-reconstruction sensitivity check.\n',
           '\nCosts are round-trip stress assumptions in basis points.\n\n']
    inv=sessions.groupby('symbol').agg(first=('date','min'),last=('date','max'),sessions=('date','count')).reset_index()
    lines.append('## Coverage\n\n'+inv.to_markdown(index=False)+'\n\n')
    cols=[c for c in ['symbol','variant','cost_bp','trades','win_rate','mean_net','median_net','profit_factor','t_stat','cum_trade_return','mdd_trade_seq','return'] if c in sdf.columns]
    lines.append('## Results\n\n'+sdf[cols].to_markdown(index=False)+'\n')
    (OUT/'SUMMARY.md').write_text(''.join(lines),encoding='utf-8')


def main():
    print('1) probing Naver public intraday endpoints...')
    fetch_naver()
    print('2) downloading public SCOTS KOSPI200 5m parquet...')
    path=download_hf()
    print('downloaded',path,path.stat().st_size)
    print('3) extracting 005930 / 000660...')
    df,sym,ts=load_hf_filtered(path)
    print('rows',len(df),'schema symbol=',sym,'timestamp=',ts)
    d=normalize(df,sym,ts)
    sessions=build_sessions(d)
    print(sessions.groupby('symbol').date.agg(['min','max','count']))
    summaries=run_backtest(sessions)
    write_md(summaries,sessions)
    print((OUT/'SUMMARY.md').read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
