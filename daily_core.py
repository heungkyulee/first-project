from __future__ import annotations

import gzip, io, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import requests

OUT=Path('daily_results'); OUT.mkdir(exist_ok=True)
REF='105a5140b3b329a2e06e7a8af5daf2f64703bc42'
BASE=f'https://raw.githubusercontent.com/wonithink-a11y/stock/{REF}/data/backfill'
TICKERS={'005930':'Samsung Electronics','000660':'SK hynix'}
YEARS=list(range(2016,2027))


def stream_gz_jsonl(url):
    with requests.get(url,stream=True,timeout=90,headers={'User-Agent':'Mozilla/5.0'}) as r:
        r.raise_for_status()
        r.raw.decode_content=False
        with gzip.GzipFile(fileobj=r.raw,mode='rb') as gz:
            for line in gz:
                if line.strip(): yield json.loads(line)


def load_price():
    rows=[]; logs=[]
    for y in YEARS:
        url=f'{BASE}/price/a2a/{y}.jsonl.gz'; n=0
        for rec in stream_gz_jsonl(url):
            if rec.get('ticker') in TICKERS:
                rows.append({k:rec.get(k) for k in ['ticker','date','open','high','low','close','volume']}); n+=1
        logs.append({'year':y,'target_rows':n})
    (OUT/'price_download.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    d=pd.DataFrame(rows); d['date']=pd.to_datetime(d.date); d=d.sort_values(['ticker','date'])
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d

INST=['금융투자','보험','투신','사모','은행','기타금융','연기금','기타법인']

def load_flow():
    rows=[]; logs=[]
    for y in YEARS:
        url=f'{BASE}/supplyDemand/a4/{y}.jsonl.gz'; n=0
        for rec in stream_gz_jsonl(url):
            if rec.get('ticker') not in TICKERS: continue
            ba=rec.get('buyAmount',{}); sa=rec.get('sellAmount',{})
            foreign=(ba.get('외국인',0)+ba.get('기타외국인',0))-(sa.get('외국인',0)+sa.get('기타외국인',0))
            inst=sum(ba.get(c,0)-sa.get(c,0) for c in INST)
            indiv=ba.get('개인',0)-sa.get('개인',0)
            rows.append({'ticker':rec['ticker'],'date':rec['date'],'foreign_net':foreign,'inst_net':inst,'indiv_net':indiv,'total_amount':ba.get('전체',0)}); n+=1
        logs.append({'year':y,'target_rows':n})
    (OUT/'flow_download.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    d=pd.DataFrame(rows); d['date']=pd.to_datetime(d.date); return d.sort_values(['ticker','date'])


def load_short():
    rows=[]; logs=[]
    for y in YEARS:
        url=f'{BASE}/shortSelling/a8/{y}.jsonl.gz'; n=0
        try:
            for rec in stream_gz_jsonl(url):
                if rec.get('ticker') in TICKERS:
                    rows.append({k:rec.get(k) for k in ['ticker','date','shortVolume','shortBalanceShares','shortValue','shortBalanceValue']}); n+=1
        except requests.HTTPError as e:
            logs.append({'year':y,'error':str(e)}); continue
        logs.append({'year':y,'target_rows':n})
    (OUT/'short_download.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    d=pd.DataFrame(rows)
    if len(d):
        d['date']=pd.to_datetime(d.date)
        for c in ['shortVolume','shortBalanceShares','shortValue','shortBalanceValue']:d[c]=pd.to_numeric(d[c],errors='coerce')
    return d.sort_values(['ticker','date']) if len(d) else d


def features(price,flow,short):
    d=price.merge(flow,on=['ticker','date'],how='left')
    if len(short): d=d.merge(short,on=['ticker','date'],how='left')
    else:
        for c in ['shortVolume','shortBalanceShares','shortValue','shortBalanceValue']:d[c]=np.nan
    out=[]
    for ticker,g in d.groupby('ticker'):
        g=g.sort_values('date').copy()
        g['ma20']=g.close.rolling(20).mean();g['ma60']=g.close.rolling(60).mean();g['ma120']=g.close.rolling(120).mean()
        g['mom20']=g.close/g.close.shift(20)-1;g['mom60']=g.close/g.close.shift(60)-1
        g['f5']=g.foreign_net.rolling(5,min_periods=3).sum();g['f20']=g.foreign_net.rolling(20,min_periods=10).sum()
        g['i5']=g.inst_net.rolling(5,min_periods=3).sum();g['i20']=g.inst_net.rolling(20,min_periods=10).sum()
        g['short_ratio']=g.shortValue/g.total_amount.replace(0,np.nan)
        g['sr5']=g.short_ratio.rolling(5,min_periods=3).mean();g['sr20']=g.short_ratio.rolling(20,min_periods=10).mean()
        g['short_bal_chg5']=g.shortBalanceShares/g.shortBalanceShares.shift(5)-1
        # TECH: fixed 50 points
        g['tech_score']=15*(g.close>g.ma120)+10*(g.ma20>g.ma60)+10*(g.ma60>g.ma120)+7.5*(g.mom20>0)+7.5*(g.mom60>0)
        # FLOW: fixed 30 points
        g['flow_score']=10*(g.f20>0)+10*(g.i20>0)+5*(g.f5>0)+5*(g.i5>0)
        # SHORT: fixed 20 points, missing early short data is neutral 10/20 rather than bullish/bearish
        has_sr=g.sr5.notna()&g.sr20.notna(); has_bal=g.short_bal_chg5.notna()
        g['short_score']=5.0
        g.loc[has_sr,'short_score']=10*(g.loc[has_sr,'sr5']<=g.loc[has_sr,'sr20'])
        g.loc[has_bal,'short_score']+=10*(g.loc[has_bal,'short_bal_chg5']<=0)
        # Three pre-specified variants, rescaled to 0..100
        g['score_price']=g.tech_score*2
        g['score_price_flow']=(g.tech_score+g.flow_score)*(100/80)
        g['score_full']=g.tech_score+g.flow_score+g.short_score
        # execution-return interval: position established at NEXT open based only on today's close info
        g['next_open']=g.open.shift(-1);g['next2_open']=g.open.shift(-2)
        out.append(g)
    return pd.concat(out,ignore_index=True)


def pos_from_score(s):
    # Mirrors existing decision bands as exposure levels: >=80 active buy, 65-79 partial buy,
    # 45-64 hold, 30-44 reduce, <=29 sell/cash.
    return np.select([s>=80,s>=65,s>=45,s>=30],[1.0,.75,.50,.25],default=0.0)


def mdd(ret):
    eq=(1+pd.Series(ret).fillna(0)).cumprod(); peak=eq.cummax(); return float((eq/peak-1).min())

def sharpe(ret):
    r=pd.Series(ret).dropna(); sd=r.std(ddof=1); return float(r.mean()/sd*np.sqrt(252)) if len(r)>2 and sd>0 else None

def ann_return(ret):
    r=pd.Series(ret).dropna(); total=float((1+r).prod()); return float(total**(252/len(r))-1) if len(r)>0 and total>0 else None

def total_return(ret):return float((1+pd.Series(ret).fillna(0)).prod()-1)


def simulate(g,score_col,roundtrip_bp=30):
    x=g.sort_values('date').copy()
    raw_pos=pos_from_score(x[score_col].fillna(50))
    # score observed at close t -> target position at open t+1. The return earned is open(t+1)->open(t+2).
    x['position']=raw_pos
    x['fwd_open_ret']=x.next2_open/x.next_open-1
    # cost at open t+1 from changing exposure vs previous target. Split round-trip equally for stress simplicity.
    delta=pd.Series(raw_pos,index=x.index).diff().fillna(raw_pos[0])
    x['turnover']=delta.abs()
    x['strategy_ret']=x.position*x.fwd_open_ret - x.turnover*(roundtrip_bp/2)/10000
    return x


def metrics(x,start,end,label,ticker,variant,cost):
    q=x[(x.date>=pd.Timestamp(start))&(x.date<=pd.Timestamp(end))&x.fwd_open_ret.notna()].copy()
    if q.empty:return None
    bh=q.fwd_open_ret
    return {'ticker':ticker,'name':TICKERS[ticker],'period':label,'variant':variant,'roundtrip_bp':cost,
            'start':str(q.date.min().date()),'end':str(q.date.max().date()),'days':len(q),
            'strategy_total':total_return(q.strategy_ret),'strategy_cagr':ann_return(q.strategy_ret),'strategy_sharpe':sharpe(q.strategy_ret),'strategy_mdd':mdd(q.strategy_ret),
            'buyhold_total':total_return(bh),'buyhold_cagr':ann_return(bh),'buyhold_sharpe':sharpe(bh),'buyhold_mdd':mdd(bh),
            'excess_total_pp':(total_return(q.strategy_ret)-total_return(bh))*100,
            'avg_exposure':float(q.position.mean()),'turnover_units':float(q.turnover.sum()),'position_changes':int((q.turnover>0).sum())}


def run():
    print('Downloading adjusted daily prices...');p=load_price();print(p.groupby('ticker').date.agg(['min','max','count']))
    print('Downloading KRX supply/demand...');f=load_flow();print(f.groupby('ticker').date.agg(['min','max','count']))
    print('Downloading short-selling...');s=load_short();print(s.groupby('ticker').date.agg(['min','max','count']) if len(s) else 'none')
    d=features(p,f,s);d.to_parquet(OUT/'features.parquet',index=False)
    periods=[('TRAIN','2016-01-01','2021-12-31'),('VALID','2022-01-01','2023-12-31'),('OOS','2024-01-01','2026-12-31'),('FULL','2016-01-01','2026-12-31')]
    variants=[('PRICE_ONLY','score_price'),('PRICE_FLOW','score_price_flow'),('PRICE_FLOW_SHORT','score_full')]
    rec=[]; curves=[]
    for ticker,g in d.groupby('ticker'):
        g=g[g.ma120.notna()].copy()
        for v,sc in variants:
            for cost in [20,30,40]:
                sim=simulate(g,sc,cost); sim['ticker']=ticker;sim['variant']=v;sim['cost']=cost;curves.append(sim[['ticker','date','variant','cost','position','fwd_open_ret','strategy_ret','turnover',sc]].rename(columns={sc:'score'}))
                for label,a,b in periods:
                    m=metrics(sim,a,b,label,ticker,v,cost)
                    if m:rec.append(m)
    md=pd.DataFrame(rec);md.to_csv(OUT/'metrics.csv',index=False)
    cv=pd.concat(curves,ignore_index=True);cv.to_parquet(OUT/'curves.parquet',index=False)
    # Equal-weight 2-stock portfolio for each variant/cost/period
    port=[]
    for (v,cost),gg in cv.groupby(['variant','cost']):
        piv=gg.pivot_table(index='date',columns='ticker',values='strategy_ret',aggfunc='first').dropna(how='all')
        bp=gg.pivot_table(index='date',columns='ticker',values='fwd_open_ret',aggfunc='first').dropna(how='all')
        r=piv.mean(axis=1,skipna=True); br=bp.mean(axis=1,skipna=True)
        for label,a,b in periods:
            mask=(r.index>=pd.Timestamp(a))&(r.index<=pd.Timestamp(b)); rr=r[mask]; bb=br.reindex(rr.index)
            if len(rr):port.append({'ticker':'COMBINED_EW','name':'50/50 Samsung+Hynix','period':label,'variant':v,'roundtrip_bp':cost,'days':len(rr),
                'strategy_total':total_return(rr),'strategy_cagr':ann_return(rr),'strategy_sharpe':sharpe(rr),'strategy_mdd':mdd(rr),
                'buyhold_total':total_return(bb),'buyhold_cagr':ann_return(bb),'buyhold_sharpe':sharpe(bb),'buyhold_mdd':mdd(bb),
                'excess_total_pp':(total_return(rr)-total_return(bb))*100})
    pd.DataFrame(port).to_csv(OUT/'portfolio_metrics.csv',index=False)
    # Compact print: OOS at realistic 30bp + full
    print('\n=== OOS 2024-2026, 30bp ===')
    print(md[(md.period=='OOS')&(md.roundtrip_bp==30)].to_string(index=False))
    print('\n=== PORTFOLIO OOS, 30bp ===')
    pm=pd.DataFrame(port);print(pm[(pm.period=='OOS')&(pm.roundtrip_bp==30)].to_string(index=False))
    # Save reproducibility/definitions
    defs={'source_ref':REF,'execution':'score at close t; rebalance at open t+1; earn open t+1 to open t+2','bands':'>=80:1.0, >=65:.75, >=45:.5, >=30:.25, else 0',
          'variants':{'PRICE_ONLY':'technical 50 rescaled x2','PRICE_FLOW':'technical50+flow30 rescaled to 100','PRICE_FLOW_SHORT':'technical50+flow30+short20'},
          'technical':'15 close>MA120 +10 MA20>MA60 +10 MA60>MA120 +7.5 mom20>0 +7.5 mom60>0',
          'flow':'10 foreign20>0 +10 institution20>0 +5 foreign5>0 +5 institution5>0',
          'short':'10 sr5<=sr20 +10 shortBalanceShares 5d change<=0; missing part neutral',
          'costs':'20/30/40bp round-trip stress, allocated half per unit exposure change',
          'no_parameter_optimization':True,'oos':'2024-01-01 onward, never used to fit weights/thresholds'}
    (OUT/'DEFINITIONS.json').write_text(json.dumps(defs,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':run()
