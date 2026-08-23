from __future__ import annotations
import gzip, json
from pathlib import Path
import requests

OUT=Path('micro_results'); OUT.mkdir(exist_ok=True)
ASSETS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT']
DATES=['2026/03/01','2026/04/01','2026/05/01','2026/06/01','2026/07/01','2026/08/01']
TYPES=['quotes','trades','book_ticker','incremental_book_L2']
BASE='https://datasets.tardis.dev/v1/binance-futures'

rows=[]
for a in ASSETS:
    for d in DATES:
        for typ in TYPES:
            u=f'{BASE}/{typ}/{d}/{a}.csv.gz'
            rec={'asset':a,'date':d,'type':typ,'url':u}
            try:
                r=requests.get(u,stream=True,timeout=45,headers={'User-Agent':'Mozilla/5.0'})
                rec.update(status=r.status_code,content_length=r.headers.get('content-length'),content_type=r.headers.get('content-type'))
                if r.ok:
                    try:
                        r.raw.decode_content=False
                        with gzip.GzipFile(fileobj=r.raw,mode='rb') as gz:
                            head=[]
                            for _ in range(4):
                                b=gz.readline()
                                if not b: break
                                head.append(b.decode('utf-8',errors='replace').strip())
                        rec['head']=head
                    except Exception as e:
                        rec['head_error']=repr(e)
                r.close()
            except Exception as e:
                rec['error']=repr(e)
            rows.append(rec)

payload={'files':rows}
(OUT/'probe.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(payload,indent=2,ensure_ascii=False))
