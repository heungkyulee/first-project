from __future__ import annotations
import io, json, zipfile
from pathlib import Path
import requests

OUT=Path('micro_results'); OUT.mkdir(exist_ok=True)
ASSETS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT']
DATES=['2026-08-20','2026-08-21','2026-08-22']
KINDS=['bookTicker','aggTrades']
BASE='https://data.binance.vision/data/futures/um/daily'


def url(asset,kind,date):
    return f'{BASE}/{kind}/{asset}/{asset}-{kind}-{date}.zip'

rows=[]
for a in ASSETS:
    for d in DATES:
        for k in KINDS:
            u=url(a,k,d)
            try:
                r=requests.get(u,stream=True,timeout=30,headers={'User-Agent':'Mozilla/5.0'})
                rows.append({'asset':a,'date':d,'kind':k,'status':r.status_code,'content_length':r.headers.get('content-length'),'content_type':r.headers.get('content-type'),'url':u})
                r.close()
            except Exception as e:
                rows.append({'asset':a,'date':d,'kind':k,'error':repr(e),'url':u})

# Download one representative small-ish pair to inspect exact schema/timestamps.
inspect=[]
for a,k in [('XRPUSDT','bookTicker'),('XRPUSDT','aggTrades')]:
    u=url(a,k,'2026-08-21')
    try:
        r=requests.get(u,timeout=120,headers={'User-Agent':'Mozilla/5.0'})
        rec={'asset':a,'kind':k,'status':r.status_code,'bytes':len(r.content)}
        if r.ok:
            z=zipfile.ZipFile(io.BytesIO(r.content))
            rec['names']=z.namelist()
            name=z.namelist()[0]
            with z.open(name) as f:
                lines=[]
                for _ in range(5):
                    b=f.readline()
                    if not b: break
                    lines.append(b.decode('utf-8',errors='replace').strip())
            rec['head']=lines
        inspect.append(rec)
    except Exception as e:
        inspect.append({'asset':a,'kind':k,'error':repr(e)})

(OUT/'probe.json').write_text(json.dumps({'files':rows,'inspect':inspect},indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'files':rows,'inspect':inspect},indent=2,ensure_ascii=False))
