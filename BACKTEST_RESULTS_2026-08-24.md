# 삼성전자·SK하이닉스 무료데이터 백테스트 결과 — 2026-08-24

## 결론

현재 HYDRA형 점수/비중조절 엔진은 **절대수익은 크게 냈지만, 삼성전자·SK하이닉스 Buy & Hold 대비 일관된 알파를 증명하지 못했다.** 2024-01-02~2026-07-30 OOS에서 SK하이닉스는 모든 변형이 Buy & Hold를 크게 하회했다. 삼성전자는 PRICE_ONLY가 20bp 비용에서 Buy & Hold와 거의 같고 MDD/Sharpe가 개선되었지만 30bp 이상에서는 절대수익 기준 소폭 하회했다. 수급·공매도를 더한 모델은 MDD를 더 줄이지만 수익도 더 많이 희생했다.

분봉 PMCRASH(14:00→종가 급락 후 다음날 시가 매수, 09:35 매도)는 최근 60거래일에서 삼성전자에는 양(+) 후보 신호가 있었지만 표본 9건으로 통계적 증거가 부족했고, SK하이닉스에서는 30bp 비용 후 음(-)의 평균수익이었다. 따라서 같은 장중 규칙을 두 종목에 공통 적용하면 안 된다.

## 1. 데이터와 실행환경

비밀키·증권사 계정 없이 GitHub Actions 무료 러너에서 공개 데이터만 사용했다.

- 일봉 가격: `wonithink-a11y/stock` 공개 backfill의 adjusted A2a, 005930/000660 각각 2016-01-04~2026-08-03, 2,595행.
- 투자자 수급: 공개 KRX backfill A4, 2016-01-04~2026-08-14.
- 공매도: 공개 KRX backfill A8, 2016-06-30~2026-08-14.
- 최근 5분봉: Yahoo Finance, 005930.KS/000660.KS 각각 2026-05-27~2026-08-21, 60거래일·4,285 bars.
- 최근 네이버 공개 5분봉과 Yahoo를 2026-08-13/14/18/19/20/21, 두 종목 12 stock-days에서 대조. 14:00, 09:35, 일종가의 median relative error가 모두 0이었다.
- 네이버 공개 `minute5`는 실측상 현재 약 6거래일만 반환했다.
- Hugging Face SCOTS의 580,136,960-byte KOSPI200 5분봉 DuckDB도 받았으나 005930/000660의 과거 2025-01~2026-01 데이터가 장 후반부 위주로 불완전하여 next-open/09:35 전략에는 사용하지 않았다.

## 2. HYDRA Quant Core 정의

첫 실행 전에 아래 규칙을 고정했고 OOS 결과를 보고 가중치를 최적화하지 않았다.

Technical 50점:
- +15: close > MA120
- +10: MA20 > MA60
- +10: MA60 > MA120
- +7.5: 20일 모멘텀 > 0
- +7.5: 60일 모멘텀 > 0

Flow 30점:
- +10: 외국인 20일 누적 순매수 > 0
- +10: 기관 20일 누적 순매수 > 0
- +5: 외국인 5일 누적 순매수 > 0
- +5: 기관 5일 누적 순매수 > 0

Short 20점:
- 최근 5일 공매도 비율 <= 20일 평균: +10
- 공매도 잔고주식수 5일 변화 <= 0: +10

세 변형:
- PRICE_ONLY: Technical 50을 0~100으로 재조정
- PRICE_FLOW: Technical 50 + Flow 30을 0~100으로 재조정
- PRICE_FLOW_SHORT: Technical 50 + Flow 30 + Short 20

점수→목표 익스포저:
- >=80: 100%
- 65~79: 75%
- 45~64: 50%
- 30~44: 25%
- <=29: 0%

실행은 close(t)에 점수를 만들고 open(t+1)에 비중을 바꾼 뒤 open(t+1)→open(t+2) 수익률을 먹는다. 미래정보 사용을 피했다. 왕복 비용은 20/30/40bp 스트레스를 적용했다.

검증 구간:
- TRAIN: 2016~2021
- VALID: 2022~2023
- OOS: 2024-01-02~2026-07-30, 627 intervals

주의: 삼성전자 2016~2021 구간에는 과거 adjusted open 축에 corporate-action 비정상치가 있어 TRAIN/FULL 누적수익의 inf/-inf는 폐기했다. 아래 핵심 판정은 해당 문제가 없는 2024~2026 OOS만 사용한다.

## 3. OOS 2024~2026, 왕복비용 30bp

### SK하이닉스 000660

| 모델 | 총수익 | CAGR | Sharpe | MDD | 평균 익스포저 |
|---|---:|---:|---:|---:|---:|
| Buy & Hold | +1,072.86% | +169.00% | 1.763 | -53.04% | 100% |
| PRICE_ONLY | +515.50% | +107.59% | 1.465 | -52.60% | 81.78% |
| PRICE_FLOW | +528.04% | +109.28% | 1.580 | -45.57% | 73.96% |
| PRICE_FLOW_SHORT | +503.53% | +105.96% | 1.611 | -42.59% | 68.30% |

판정: **알파 실패.** 수급/공매도는 MDD를 약 10.4%p 줄였지만 초강세의 convex upside를 놓쳐 Buy & Hold 대비 약 545~569%p 누적수익을 희생했다. 비용 20~40bp를 바꿔도 결론은 변하지 않았다.

### 삼성전자 005930

| 모델 | 총수익 | CAGR | Sharpe | MDD | 평균 익스포저 |
|---|---:|---:|---:|---:|---:|
| Buy & Hold | +215.92% | +58.78% | 1.134 | -43.28% | 100% |
| PRICE_ONLY | +212.61% | +58.11% | 1.207 | -36.56% | 73.60% |
| PRICE_FLOW | +175.37% | +50.25% | 1.161 | -34.81% | 65.59% |
| PRICE_FLOW_SHORT | +182.82% | +51.87% | 1.299 | -32.30% | 62.08% |

PRICE_ONLY는 20bp 비용에서는 +216.15%로 Buy & Hold +215.92%보다 +0.22%p 높았으나, 30bp에서 -3.32%p, 40bp에서 -6.82%p로 다시 하회했다.

판정: **절대수익 알파는 미증명, 리스크 조절은 유효.** 특히 PRICE_FLOW_SHORT는 Sharpe 1.299 vs 1.134, MDD -32.30% vs -43.28%로 위험조정 특성이 좋아졌지만 총수익은 33.10%p 낮았다.

### 50/50 삼성전자+SK하이닉스

| 모델 | 총수익 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Buy & Hold | +550.25% | +112.22% | 1.594 | -47.19% |
| PRICE_ONLY | +362.93% | +85.13% | 1.446 | -43.44% |
| PRICE_FLOW | +334.67% | +80.50% | 1.498 | -39.20% |
| PRICE_FLOW_SHORT | +329.42% | +79.62% | 1.581 | -35.62% |

판정: full 모델은 MDD를 약 11.6%p 줄이고 Sharpe는 Buy & Hold에 거의 근접했지만, 절대수익은 약 220.8%p 낮았다. **현재 엔진은 alpha engine이 아니라 risk/exposure controller에 가깝다.**

참고로 2022~2023 VALID, 30bp에서 50/50 Buy & Hold는 +6.03%, PRICE_ONLY +8.41%, PRICE_FLOW +6.91%, PRICE_FLOW_SHORT -0.41%였다. 따라서 수급·공매도 축의 추가가 시기 전반에서 일관되게 개선을 만들지는 않았다.

## 4. 최근 60일 PMCRASH 장중 이벤트 스터디

원형 핵심 규칙: 당일 14:00→종가 <= -2%, 다음 거래일 시가 매수, 09:35 매도. 공개 원 연구의 거래량 feature 생성기는 repo에 없어서 두 버전을 같이 봤다.

- NO_VOLUME: 급락 조건만 사용, 재구성 오류 없음.
- RELVOL_LT_1_5: 직전 20세션 동일 시간대 거래량 median 대비 상대거래량 <1.5로 재구성. 원 feature의 정확한 복제품이라고 주장하지 않는다.

### 삼성전자, -2%, 30bp

NO_VOLUME:
- 9 trades
- gross mean +0.9726%
- net mean +0.6726%
- median net -0.3929%
- win rate 44.4%
- Profit Factor 2.03
- t = 0.70
- bootstrap 95% CI of mean net = [-0.879%, +2.608%]

RELVOL_LT_1_5 reconstruction:
- 5 trades
- gross mean +1.3010%
- net mean +1.0010%
- PF 2.49
- t = 0.62
- 95% CI = [-1.265%, +4.192%]

판정: **흥미로운 후보지만 증거 부족.** 평균은 양수이고 20~50bp 비용 스트레스에서도 mean은 양수였으나 표본이 9건, 5건이고 CI가 0을 넓게 포함한다. 실전 규칙으로 승격 금지.

### SK하이닉스, -2%, 30bp

NO_VOLUME:
- 13 trades
- gross mean +0.1896%
- net mean -0.1104%
- median net -1.0658%
- win rate 38.5%
- PF 0.905
- t = -0.14
- 95% CI = [-1.515%, +1.477%]

RELVOL_LT_1_5 reconstruction:
- 8 trades
- gross mean -0.6358%
- net mean -0.9358%
- median net -1.0868%
- win rate 25%
- PF 0.334
- t = -1.27

민감도에서도 Hynix NO_VOLUME 30bp 평균은 threshold -1%, -1.5%, -2%, -2.5%, -3%에서 모두 음수였다.

판정: **최근 구간 Hynix PMCRASH는 기각.** 동일 규칙을 두 종목에 공유하면 안 된다.

## 5. 과학적 판정

1. 현재 HYDRA식 공통 점수 엔진은 “돈을 버는가?”라는 절대 질문에는 강세장 덕에 yes지만, “Buy & Hold보다 더 좋은 alpha가 있는가?”에는 **no evidence**다.
2. Hynix는 frequent de-risking이 upside를 크게 훼손했다. 강한 추세에서는 기본 full-long을 유지하고 명확한 thesis-break 때만 hedge/reduce하는 구조가 더 적합하다는 가설이 생겼다. 이것은 이번 OOS를 본 뒤 생긴 post-hoc 가설이므로 같은 OOS에 다시 맞춰 ‘증명’하면 안 된다.
3. Samsung은 tactical risk timing이 Hynix보다 잘 맞았다. PRICE_ONLY가 B&H와 거의 같은 수익에 MDD를 줄였고, PMCRASH도 최근 60일에서 양의 후보 신호가 있었다. 다만 PMCRASH n=9라 forward validation이 필요하다.
4. 수급/공매도는 독립 alpha라기보다는 risk-control feature로 보는 편이 현재 증거와 맞다.
5. 여기서 OOS를 계속 보고 가중치를 손으로 바꿔 수익이 나올 때까지 반복하면 p-hacking이다. 다음 전략 버전은 walk-forward/expanding-window updater로만 제안하고, 현재 2024~2026은 이미 researcher OOS가 소비되었다고 표시해야 한다.

## 6. 다음 버전 후보, 아직 미승격

- Hynix: full-long base + high-conviction invalidation overlay. 점수 45~79에서 자동 50~75%로 줄이는 현재 band를 폐기하는 후보.
- Samsung: PRICE_ONLY risk scaler를 baseline으로 두고, PMCRASH를 ‘추가매수 후보’로만 붙이는 구조.
- 공통: 수급/공매도는 매수 alpha 점수보다 리스크 페널티/신뢰도에 사용.
- 어떤 후보도 이번 OOS를 재최적화해 바로 activation하지 않는다. 월별 expanding-window walk-forward와 이후 실제 forward signal ledger를 통과해야 activation한다.
