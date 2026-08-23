from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
import yfinance as yf
import requests

OUT=Path('yahoo_results'); OUT.mkdir(exist_ok=True)
SYMS={'005930':'005930.KS','000660':'000660.KS'}


def tstat(x):
    x=np.array(pd.Series(x).dropna(),float)
    return float(x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))) if len(x)>=2 and x.std(ddof=1)>0 else None

def ci(x,n=10000,seed=7):
    x=np.array(pd.Series(x).dropna(),float)
    if len(x)<2:return [None,None]
    rng=np.random.default_rng(seed); m=np.empty(n)
    for i in range(n):m[i]=rng.choice(x,len(x),replace=True).mean()
    return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]

def pf(x):
    x=pd.Series(x);p=x[x>0].sum();n=-x[x<0].sum();return float(p/n) if n>0 else None


def flat_download(ticker,interval,period):
    d=yf.download(ticker,interval=interval,period=period,auto_adjust=False,prepost=False,progress=False,threads=False)
    if isinstance(d.columns,pd.MultiIndex):d.columns=[c[0] for c in d.columns]
    return d


def build(ticker, ysym):
    m=flat_download(ysym,'5m','60d')
    day=flat_download(ysym,'1d','6mo')
    # intraday index is tz-aware Asia/Seoul on yfinance; enforce KST
    idx=pd.to_datetime(m.index)
    if idx.tz is None: idx=idx.tz_localize('Asia/Seoul')
    else: idx=idx.tz_convert('Asia/Seoul')
    m=m.copy();m['ts']=idx;m['date']=idx.date;m['hm']=idx.hour*60+idx.minute
    day=day.copy(); day.index=pd.to_datetime(day.index).date
    rows=[]
    for dt,g in m.groupby('date'):
        g=g.sort_values('ts')
        a=g[g.hm<=14*60]
        b=g[g.hm<=9*60+35]
        if a.empty or b.empty or dt not in day.index:continue
        drow=day.loc[dt]
        rows.append({'date':pd.Timestamp(dt),'open_daily':float(drow.Open),'close_daily':float(drow.Close),
                     'c1400':float(a.iloc[-1].Close),'c0935':float(b.iloc[-1].Close),
                     'vol1400_1500':float(g[(g.hm>=14*60)&(g.hm<15*60)].Volume.sum()),
                     'bars':len(g),'first_hm':int(g.hm.min()),'last_hm':int(g.hm.max())})
    s=pd.DataFrame(rows).sort_values('date')
    s['ticker']=ticker
    # link by observed trading-day sequence, no calendar assumptions
    s['next_date']=s.date.shift(-1);s['next_open']=s.open_daily.shift(-1);s['next_0935']=s.c0935.shift(-1)
    s['gross']=s.next_0935/s.next_open-1
    s['crash']=s.close_daily/s.c1400-1
    s['vol_med20_prev']=s.vol1400_1500.shift(1).rolling(20,min_periods=10).median()
    s['relvol_recon']=s.vol1400_1500/s.vol_med20_prev
    return s


def naver_crosscheck(frames):
    rec=[]
    for ticker,s in frames.items():
        for dt in ['20260813','20260814','20260818','20260819','20260820','20260821']:
            r=requests.get(f'https://api.stock.naver.com/chart/domestic/item/{ticker}/minute5',params={'startDateTime':dt+'0900','endDateTime':dt+'1530'},timeout=20,headers={'User-Agent':'Mozilla/5.0'})
            arr=r.json() if r.ok else []
            if not isinstance(arr,list) or not arr:continue
            nv=pd.DataFrame(arr);nv['ts']=pd.to_datetime(nv.localDateTime);nv['hm']=nv.ts.dt.hour*60+nv.ts.dt.minute
            y=s[s.date==pd.Timestamp(pd.to_datetime(dt).date())]
            if y.empty:continue
            y=y.iloc[0]
            def nclose(hm):
                q=nv[nv.hm<=hm];return float(q.iloc[-1].currentPrice) if len(q) else None
            rec.append({'ticker':ticker,'date':dt,'y_c1400':y.c1400,'n_c1400':nclose(840),'y_c0935':y.c0935,'n_c0935':nclose(575),'y_daily_close':y.close_daily,'n_close':float(nv.iloc[-1].currentPrice)})
    c=pd.DataFrame(rec)
    if len(c):
        for x in ['c1400','c0935']:
            c[x+'_relerr']=(c['y_'+x]/c['n_'+x]-1).abs()
        c['daily_close_relerr']=(c.y_daily_close/c.n_close-1).abs()
        c.to_csv(OUT/'naver_yahoo_crosscheck.csv',index=False)
        summ={'rows':len(c),'median_c1400_relerr':float(c.c1400_relerr.median()),'median_c0935_relerr':float(c.c0935_relerr.median()),'median_daily_close_relerr':float(c.daily_close_relerr.median())}
        (OUT/'crosscheck.json').write_text(json.dumps(summ,indent=2),encoding='utf-8')


def main():
    frames={}; results=[]; trades=[]
    for t,ys in SYMS.items():
        s=build(t,ys);frames[t]=s;s.to_csv(OUT/f'{t}_sessions.csv',index=False)
        print(t,s.date.min(),s.date.max(),len(s),'last_hm counts',s.last_hm.value_counts().head().to_dict())
        for thr in [-0.01,-0.015,-0.02,-0.025,-0.03]:
            for volmode in ['NO_VOLUME','RELVOL_LT_1_5']:
                mask=(s.crash<=thr)&s.gross.notna()
                if volmode!='NO_VOLUME':mask &= (s.relvol_recon<1.5)
                tr=s[mask].copy();tr['threshold']=thr;tr['volmode']=volmode;trades.append(tr)
                for cost in [20,30,40,50]:
                    net=tr.gross-cost/10000 if len(tr) else pd.Series(dtype=float)
                    results.append({'ticker':t,'threshold':thr,'volmode':volmode,'cost_bp':cost,'trades':len(tr),
                                    'mean_gross':float(tr.gross.mean()) if len(tr) else None,'mean_net':float(net.mean()) if len(net) else None,
                                    'median_net':float(net.median()) if len(net) else None,'win_rate':float((net>0).mean()) if len(net) else None,
                                    'profit_factor':pf(net) if len(net) else None,'t_stat':tstat(net),'ci95':ci(net),
                                    'start':str(s.date.min().date()),'end':str(s.date.max().date())})
    pd.DataFrame(results).to_csv(OUT/'event_study.csv',index=False)
    pd.concat(trades,ignore_index=True).to_csv(OUT/'trades.csv',index=False)
    naver_crosscheck(frames)
    r=pd.DataFrame(results)
    print('\n=== PRE-SPECIFIED -2% SIGNAL ===')
    print(r[(r.threshold==-0.02)&(r.cost_bp.isin([20,30,40]))].to_string(index=False))
    print('\n=== SENSITIVITY 30bp NO_VOLUME ===')
    print(r[(r.cost_bp==30)&(r.volmode=='NO_VOLUME')].to_string(index=False))
    if (OUT/'crosscheck.json').exists():print('\nCROSSCHECK', (OUT/'crosscheck.json').read_text())

if __name__=='__main__':main()
