from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import duckdb

OUT=Path('results2'); OUT.mkdir(exist_ok=True)
TICKERS=['005930','000660']


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def naver_retention_probe():
    dates=['20260821','20260820','20260819','20260814','20260807','20260731','20260724','20260717','20260710','20260630','20260615','20260527']
    out=[]
    for t in TICKERS:
        for d in dates:
            url=f'https://api.stock.naver.com/chart/domestic/item/{t}/minute5'
            try:
                r=requests.get(url,params={'startDateTime':d+'0900','endDateTime':d+'1530'},timeout=20,headers={'User-Agent':'Mozilla/5.0'})
                p=r.json() if r.ok else None
                rows=len(p) if isinstance(p,list) else None
                first=p[0] if isinstance(p,list) and p else None
                last=p[-1] if isinstance(p,list) and p else None
                out.append({'ticker':t,'date':d,'status':r.status_code,'rows':rows,'first':first,'last':last})
            except Exception as e: out.append({'ticker':t,'date':d,'error':repr(e)})
    # multi-day ranges
    for t in TICKERS:
        for a,b in [('202608010900','202608211530'),('202607010900','202608211530'),('202606010900','202608211530')]:
            try:
                url=f'https://api.stock.naver.com/chart/domestic/item/{t}/minute5'
                r=requests.get(url,params={'startDateTime':a,'endDateTime':b},timeout=30,headers={'User-Agent':'Mozilla/5.0'})
                p=r.json() if r.ok else None
                out.append({'ticker':t,'range':[a,b],'status':r.status_code,'rows':len(p) if isinstance(p,list) else None,
                            'first':p[0] if isinstance(p,list) and p else None,'last':p[-1] if isinstance(p,list) and p else None})
            except Exception as e: out.append({'ticker':t,'range':[a,b],'error':repr(e)})
    dump('naver_retention.json',out)
    return out


def download_bigdb():
    urls=[
      'https://huggingface.co/datasets/alansynn/scots/resolve/dfc41730d51b957c34a6fd41865780fde302e157/duckdb/historical_5min_kospi_kis.duckdb?download=true',
      'https://huggingface.co/datasets/alansynn/scots/resolve/dfc41730d51b957c34a6fd41865780fde302e157/historical_5min_kospi_kis.duckdb?download=true',
      'https://huggingface.co/datasets/alansynn/scots/resolve/main/duckdb/historical_5min_kospi_kis.duckdb?download=true',
    ]
    dest=Path('/tmp/kospi5.duckdb'); logs=[]
    for u in urls:
        try:
            with requests.get(u,stream=True,timeout=90,allow_redirects=True,headers={'User-Agent':'Mozilla/5.0'}) as r:
                logs.append({'url':u,'status':r.status_code,'final':r.url,'length':r.headers.get('content-length')})
                if not r.ok: continue
                with dest.open('wb') as f:
                    for ch in r.iter_content(1024*1024):
                        if ch:f.write(ch)
                if dest.stat().st_size>10_000_000:
                    dump('bigdb_download.json',logs); return dest
        except Exception as e: logs.append({'url':u,'error':repr(e)})
    dump('bigdb_download.json',logs); return None


def inspect_bigdb(path):
    if path is None: return None
    con=duckdb.connect(str(path),read_only=True)
    tables=con.execute('show tables').fetchall(); dump('bigdb_tables.json',tables)
    result={}
    for table in ['bars_5min','ohlcv','bars']:
        try:
            cols=con.execute(f'describe {table}').fetchdf();
            result['table']=table; result['columns']=cols.to_dict('records')
            inv=con.execute(f"select cast(symbol as varchar) symbol,min(timestamp) min_ts,max(timestamp) max_ts,count(*) n from {table} where cast(symbol as varchar) in ('005930','000660') group by 1 order by 1").fetchdf()
            inv.to_csv(OUT/'bigdb_inventory.csv',index=False)
            df=con.execute(f"select symbol,timestamp,open,high,low,close,volume from {table} where cast(symbol as varchar) in ('005930','000660') order by symbol,timestamp").fetchdf()
            df.to_parquet(OUT/'bigdb_targets.parquet',index=False)
            result['rows']=len(df)
            dump('bigdb_inspect.json',result)
            return df
        except Exception as e:
            result.setdefault('errors',[]).append({'table':table,'error':repr(e)})
    dump('bigdb_inspect.json',result); return None


def yfinance_fetch():
    import yfinance as yf
    frames=[]; logs=[]
    for ticker in ['005930.KS','000660.KS']:
        try:
            d=yf.download(ticker,period='60d',interval='5m',auto_adjust=False,prepost=False,progress=False,threads=False)
            logs.append({'ticker':ticker,'rows':len(d),'start':str(d.index.min()) if len(d) else None,'end':str(d.index.max()) if len(d) else None,'columns':str(d.columns)})
            if len(d):
                if isinstance(d.columns,pd.MultiIndex): d.columns=[c[0] for c in d.columns]
                d=d.reset_index(); d['symbol']=ticker.split('.')[0]; frames.append(d)
        except Exception as e: logs.append({'ticker':ticker,'error':repr(e)})
    dump('yfinance_log.json',logs)
    if not frames:return None
    df=pd.concat(frames,ignore_index=True)
    df.to_parquet(OUT/'yahoo_targets.parquet',index=False)
    return df


def normalize_any(df, source):
    x=df.copy(); cmap={str(c).lower():c for c in x.columns}
    if source=='bigdb':
        x['symbol']=x[cmap['symbol']].astype(str).str.zfill(6); raw=x[cmap['timestamp']]
    else:
        raw=x[cmap.get('datetime',cmap.get('date'))]
    ts=pd.to_datetime(raw,errors='coerce',utc=True)
    kst=ts.dt.tz_convert('Asia/Seoul')
    x['ts_kst']=kst
    def col(name,alt=None):
        key=name.lower(); return cmap.get(key,cmap.get((alt or '').lower()))
    if source!='bigdb':
        x['open']=pd.to_numeric(x[col('open')],errors='coerce');x['high']=pd.to_numeric(x[col('high')],errors='coerce');x['low']=pd.to_numeric(x[col('low')],errors='coerce');x['close']=pd.to_numeric(x[col('close')],errors='coerce');x['volume']=pd.to_numeric(x[col('volume')],errors='coerce')
    return x[['symbol','ts_kst','open','high','low','close','volume']].dropna().sort_values(['symbol','ts_kst'])


def sessions(d):
    x=d.copy();x['date']=x.ts_kst.dt.date;x['hm']=x.ts_kst.dt.hour*60+x.ts_kst.dt.minute
    rows=[]
    for (s,dt),g in x.groupby(['symbol','date']):
        g=g[(g.hm>=540)&(g.hm<=930)].sort_values('ts_kst')
        if len(g)<20:continue
        def bef(m):
            q=g[g.hm<=m]; return None if q.empty else q.iloc[-1]
        a=bef(840); b=bef(575)
        if a is None or b is None:continue
        rows.append({'symbol':s,'date':pd.Timestamp(dt),'open':float(g.iloc[0].open),'close':float(g.iloc[-1].close),'c1400':float(a.close),'c0935':float(b.close),'bars':len(g)})
    return pd.DataFrame(rows).sort_values(['symbol','date'])


def evaluate(s, label):
    records=[]; trades=[]
    for sym,g in s.groupby('symbol'):
        g=g.sort_values('date').copy();g['crash']=g['close']/g['c1400']-1;g['next_open']=g.open.shift(-1);g['next_0935']=g.c0935.shift(-1);g['gross']=g.next_0935/g.next_open-1
        for thr in [-0.01,-0.015,-0.02,-0.025,-0.03]:
            t=g[(g.crash<=thr)&g.gross.notna()].copy(); t['threshold']=thr;t['source']=label;trades.append(t)
            for cost in [20,30,40]:
                net=t.gross-cost/10000 if len(t) else pd.Series(dtype=float)
                records.append({'source':label,'symbol':sym,'threshold':thr,'cost_bp':cost,'trades':len(t),'mean_gross':float(t.gross.mean()) if len(t) else None,'mean_net':float(net.mean()) if len(t) else None,'win':float((net>0).mean()) if len(t) else None,'median_net':float(net.median()) if len(t) else None})
        records.append({'source':label,'symbol':sym,'threshold':'BH','return':float(g.iloc[-1].close/g.iloc[0].open-1),'sessions':len(g),'start':str(g.date.min().date()),'end':str(g.date.max().date())})
    if trades:pd.concat(trades,ignore_index=True).to_csv(OUT/f'{label}_trades.csv',index=False)
    return records


def crosscheck(a,b):
    if a is None or b is None:return
    sa=sessions(normalize_any(a,'bigdb')); sb=sessions(normalize_any(b,'yahoo'))
    m=sa.merge(sb,on=['symbol','date'],suffixes=('_kis','_yahoo'))
    if len(m):
        for c in ['open','close','c1400','c0935']:
            m[c+'_relerr']=(m[c+'_yahoo']/m[c+'_kis']-1).abs()
        m.to_csv(OUT/'crosscheck_kis_yahoo.csv',index=False)
        dump('crosscheck_summary.json',{'rows':len(m),**{c:float(m[c+'_relerr'].median()) for c in ['open','close','c1400','c0935']}})


def main():
    print('Naver retention...'); naver_retention_probe()
    print('Big DB...'); p=download_bigdb(); big=inspect_bigdb(p)
    print('Yahoo...'); y=yfinance_fetch()
    rec=[]
    if big is not None and len(big):
        sb=sessions(normalize_any(big,'bigdb')); sb.to_csv(OUT/'bigdb_sessions.csv',index=False); rec+=evaluate(sb,'BIGDB')
    if y is not None and len(y):
        sy=sessions(normalize_any(y,'yahoo')); sy.to_csv(OUT/'yahoo_sessions.csv',index=False); rec+=evaluate(sy,'YAHOO60D')
    crosscheck(big,y)
    pd.DataFrame(rec).to_csv(OUT/'event_study.csv',index=False)
    print(pd.DataFrame(rec).to_string(index=False))
    print((OUT/'naver_retention.json').read_text()[:12000])

if __name__=='__main__':main()
