from __future__ import annotations
import io, json, zipfile
from pathlib import Path
import requests

OUT=Path('micro_results'); OUT.mkdir(exist_ok=True)
ASSETS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT']
BASE='https://data.binance.vision/data/futures/um'
rows=[]

# Completed-month bookTicker is the high-resolution BBO source.
for a in ASSETS:
    for month in ['2026-06','2026-07']:
        for kind in ['bookTicker','aggTrades']:
            u=f'{BASE}/monthly/{kind}/{a}/{a}-{kind}-{month}.zip'
            try:
                r=requests.get(u,stream=True,timeout=30,headers={'User-Agent':'Mozilla/5.0'})
                rows.append({'asset':a,'period':month,'scope':'monthly','kind':kind,'status':r.status_code,'content_length':r.headers.get('content-length'),'content_type':r.headers.get('content-type'),'url':u})
                r.close()
            except Exception as e:
                rows.append({'asset':a,'period':month,'scope':'monthly','kind':kind,'error':repr(e),'url':u})

# Daily bookDepth is much lower frequency, but inspect availability as a fallback.
for a in ASSETS:
    d='2026-08-21'; kind='bookDepth'
    u=f'{BASE}/daily/{kind}/{a}/{a}-{kind}-{d}.zip'
    try:
        r=requests.get(u,stream=True,timeout=30,headers={'User-Agent':'Mozilla/5.0'})
        rows.append({'asset':a,'period':d,'scope':'daily','kind':kind,'status':r.status_code,'content_length':r.headers.get('content-length'),'content_type':r.headers.get('content-type'),'url':u})
        r.close()
    except Exception as e:
        rows.append({'asset':a,'period':d,'scope':'daily','kind':kind,'error':repr(e),'url':u})

inspect=[]
# Download the smaller XRP monthly BBO if reasonable and inspect the exact schema/timestamps.
u=f'{BASE}/monthly/bookTicker/XRPUSDT/XRPUSDT-bookTicker-2026-07.zip'
try:
    r=requests.get(u,timeout=180,headers={'User-Agent':'Mozilla/5.0'})
    rec={'asset':'XRPUSDT','kind':'bookTicker','period':'2026-07','status':r.status_code,'bytes':len(r.content)}
    if r.ok:
        z=zipfile.ZipFile(io.BytesIO(r.content)); rec['names']=z.namelist(); name=z.namelist()[0]
        with z.open(name) as f:
            lines=[]
            for _ in range(6):
                b=f.readline()
                if not b: break
                lines.append(b.decode('utf-8',errors='replace').strip())
        rec['head']=lines
    inspect.append(rec)
except Exception as e:
    inspect.append({'asset':'XRPUSDT','kind':'bookTicker','period':'2026-07','error':repr(e)})

payload={'files':rows,'inspect':inspect}
(OUT/'probe.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(payload,indent=2,ensure_ascii=False))
