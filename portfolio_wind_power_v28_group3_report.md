# 기상예보 기반 풍력발전량 예측 모델링 연구 보고서

## 초록

본 프로젝트는 DACON 풍력발전량 예측 경진대회 데이터를 활용하여, LDAPS/GFS 수치예보 데이터와 터빈 SCADA 관측 데이터를 기반으로 KPX 그룹별 시간 단위 발전량을 예측하는 것을 목표로 하였다. 최종 제출 기준 모델은 `v28 ExtraTrees`로 고정하였다. 이후 LightGBM, CatBoost, XGBoost, feature slimming, 정격권 터빈 수 proxy, 후류 feature, 눈/적설 risk feature 등을 검토했으나, 리더보드와 2024 holdout 검증에서 v28을 안정적으로 대체하지 못하였다.

핵심 분석 대상은 `group3`였다. group3는 group1/2와 달리 2023~2024년 label만 존재하며, 설비용량 대비 10% 미만 발전 구간이 전체 label의 약 46.3%를 차지하였다. 또한 80~100% 고발전 구간은 약 7.8%에 불과하였다. 이로 인해 모델은 고발전 구간에서 평균 회귀(regression to the mean) 성향을 보였고, 여러 모델에서도 고발전 과소예측이 반복되었다. 본 보고서는 이러한 현상을 데이터 분포, 예보 feature 분리력, SCADA 기반 물리 해석, 모델 비교 실험의 관점에서 정리한다.

## 1. 문제 정의

본 대회의 목적은 특정 예보 생성 시점에 실제로 사용 가능한 기상예보 데이터만을 이용하여, 미래 12~35시간의 KPX 그룹별 시간 단위 풍력발전량을 예측하는 것이다.

평가 지표는 다음 두 가지를 함께 고려한다.

- `1-NMAE`: 설비용량으로 정규화한 평균 절대 오차. 값이 클수록 좋다.
- `FICR`: 시간별 오차율이 6% 이하 또는 8% 이하 구간에 들어가는 정도를 정산금 관점에서 평가한 지표. 값이 클수록 좋다.

공식 평가에서는 실제 발전량이 설비용량의 10% 이상인 시간만 평가 대상이 된다. 따라서 단순히 전체 MAE를 낮추는 것보다, 평가 대상 구간에서 6%/8% 오차 경계 안에 들어오도록 예측을 안정화하는 것이 중요하다.

## 2. 데이터 구성

사용 데이터는 다음과 같다.

- `train_labels.csv`: KPX group1, group2, group3의 시간별 실제 발전량 label
- `ldaps_train/test.csv`: LDAPS 지역 수치예보 데이터
- `gfs_train/test.csv`: GFS 전지구 수치예보 데이터
- `scada_vestas_train.csv`: VESTAS 터빈별 10분 단위 SCADA 관측 데이터
- `scada_unison_train.csv`: UNISON 터빈별 10분 단위 SCADA 관측 데이터
- `info.xlsx`: 터빈 위치, 제조사, 설비용량, rotor diameter, hub height 등 메타 정보

group별 설비용량은 다음과 같이 적용하였다.

| 그룹 | 설비용량 |
|---|---:|
| group1 | 21,600 kWh |
| group2 | 21,600 kWh |
| group3 | 21,000 kWh |

group3는 UNISON U136 터빈 5기로 구성되며, rotor diameter는 136m, hub height는 117m로 확인하였다.

## 3. 기준 모델 v28

최종 기준 모델은 `baseline_ontology_v28_gfs_grid5.py`이다.

v28의 핵심 구조는 다음과 같다.

- 모델: group별 `ExtraTreesRegressor`
- target: 발전량을 설비용량으로 나눈 capacity factor 형태로 학습
- feature 수: 598개
- 입력 feature:
  - calendar feature
  - LDAPS 전체 격자 평균/최대/최소/표준편차
  - group별 거리 가중 LDAPS feature
  - GFS 전체 격자 및 group별 거리 가중 feature
  - GFS grid5 중심 상층 풍속 feature
  - 풍속 크기, 풍속 제곱, 풍속 세제곱
  - GFS 10m/100m 기반 wind shear alpha 및 hub 117m proxy

v28은 여러 파생 feature와 실험 모델 중 리더보드 기준 가장 안정적인 성능을 보였으므로, 이후 모든 분석의 기준점으로 고정하였다.

## 4. Group3 타깃 분포 EDA

group3는 group1/2와 다르게 2023~2024년 label만 존재한다. 분석 대상 group3 유효 label 수는 17,537개였다.

group3 capacity factor 분포는 다음과 같다.

| CF 구간 | 행 수 | 비율 |
|---|---:|---:|
| 0 발전 | 2,846 | 16.23% |
| 0~10% | 5,278 | 30.09% |
| 10~20% | 1,944 | 11.08% |
| 20~40% | 2,365 | 13.49% |
| 40~60% | 1,891 | 10.78% |
| 60~80% | 1,800 | 10.26% |
| 80~100% | 1,376 | 7.85% |
| 100% 초과 | 38 | 0.22% |

따라서 group3는 설비용량의 10% 미만 발전 구간이 약 46.3%로 매우 크며, 반대로 80~100% 고발전 구간은 약 7.8%에 불과하다. 이 분포는 회귀 모델이 고발전 tail을 과소예측하기 쉬운 구조를 만든다.

## 5. SCADA 기반 물리 해석

SCADA는 예측 시점의 test에는 제공되지 않으므로 직접 feature로 사용할 수 없다. 하지만 train 구간에서는 예보 feature와 실제 터빈 상태의 관계를 해석하는 데 사용할 수 있다.

UNISON group3 SCADA는 10분 단위로 기록되며, 주요 컬럼은 다음과 같다.

- 터빈별 `power_kw10m`
- 터빈별 nacelle wind speed `ws`
- 터빈별 wind direction `wd`

분석 과정에서 `power_kw10m`은 순간 출력이라기보다 10분 단위 발전량 성격을 가지는 것으로 해석하였다. UNISON 4.2MW 터빈의 10분 정격 발전량은 약 700kWh이며, SCADA 값도 이 근처에서 포화되는 패턴을 보였다.

### 5.1 정격 풍속과 실제 출력

UNISON의 정격 풍속을 약 11.3m/s로 두고, raw 10분 SCADA에서 `ws >= 11.3m/s`인 경우를 분석하였다.

주요 결과:

- 전체 해당 row: 53,955개
- power CF median: 1.0
- power CF mean: 약 0.832
- CF >= 100% 비율: 약 52.3%
- CF >= 90% 비율: 약 67.1%
- CF < 50% 비율: 약 13.3%
- 0 또는 음수 출력 비율: 약 10.7%

즉, 정격 풍속 이상이라고 항상 100% 출력이 되는 것은 아니다. 같은 고풍속에서도 제어, 정지, 난류, 후류, 착빙, 계통 제한 등의 영향으로 출력이 낮을 수 있다.

### 5.2 정격권 터빈 개수와 group3 발전량

시간 단위로 각 터빈의 평균 풍속을 계산한 뒤, 11.3m/s 이상인 터빈 개수를 세어 실제 group3 label과 비교하였다.

정격권 터빈 수가 많아질수록 고발전 확률은 증가했다. 특히 SCADA 기준 강한 gate 내부에서 60~80% 발전과 80~100% 발전을 비교하면 다음 차이가 나타났다.

| 변수 | 60~80% 평균 | 80~100% 평균 | 차이 |
|---|---:|---:|---:|
| SCADA 정격권 터빈 수 | 2.72 | 3.78 | +1.06 |
| SCADA 평균 풍속 | 12.02 | 13.35 | +1.33 |
| SCADA 터빈 최소 풍속 | 9.84 | 11.04 | +1.21 |
| SCADA 터빈 최대 풍속 | 13.89 | 15.22 | +1.33 |

80~100% 구간에서는 정격권 터빈 5개인 경우가 약 40.9%였고, 60~80% 구간에서는 약 21.8%였다. 따라서 group3 고발전은 “한 지점의 풍속이 높다”보다 “여러 터빈이 동시에 정격권에 들어간다”에 더 가깝다.

## 6. 예보 feature가 고발전 구간을 구분하는가

핵심 질문은 다음과 같이 정리하였다.

> LDAPS/GFS 예보 feature만으로 group3의 CF 60~80%와 80~100%를 안정적으로 구분할 수 있는가?

이를 위해 group3 label 기준으로 60~80%와 80~100% 구간만 분리하고, v28의 598개 feature를 사용하여 분리 가능성을 평가하였다.

2024 holdout 기준 이진 분류 AUC는 다음과 같다.

| feature set | 모델 | AUC |
|---|---|---:|
| 전체 598개 | Logistic Regression | 0.594 |
| 전체 598개 | ExtraTrees Classifier | 0.693 |
| 해석 가능한 날씨 feature 105개 | ExtraTrees Classifier | 0.663 |
| 단변량 상위 40개 | Logistic Regression | 0.679 |
| 단변량 상위 40개 | ExtraTrees Classifier | 0.672 |
| 수작업 풍속/calendar 62개 | ExtraTrees Classifier | 0.635 |

결론적으로, 예보 feature에는 고발전 구분 신호가 존재하지만 충분히 강하지 않다. AUC 0.69 수준은 완전한 무작위보다는 낫지만, 80~100% 고발전을 안정적으로 구분할 정도로 강한 신호는 아니다.

상위 분리 feature는 다음 계열에 집중되었다.

- LDAPS 50m max allgrid max
- LDAPS 10m allgrid max
- LDAPS group3 nearest/grid distance weighted wind
- GFS group3 850hPa wind speed
- GFS grid5 100m/hub117/850hPa 계열
- GFS/LDAPS의 v 성분, 즉 남북 방향 바람 성분

따라서 LDAPS 50m 계열은 고발전 설명력이 있으나, 단독으로 80~100% 발전을 확정하지는 못한다.

## 7. 고풍속 gate와 한계

고발전 sample의 중앙값을 기준으로 LDAPS/GFS wind gate를 만들었을 때, 가장 강한 조합은 다음과 같았다.

- `LDAPS grid12 50m high`
- `GFS grid5 850hPa high`

이 gate 내부에서 CF >= 80% 비율은 약 42.3%였다. SCADA stop/curtailment 의심 구간을 제거해도 약 42.9%로만 증가했다.

따라서 단순한 “정지/제어 이상치 때문에 고발전을 못 맞힌다”는 설명은 충분하지 않다. 고풍속 gate 안에서도 60~80%와 80~100%가 섞이는 것이 본질적인 문제다.

## 8. 눈·적설·영하·고습도 조건 분석

눈 조건은 다음과 같이 정의하였다.

- LDAPS 강설률 `lssrate > 0`
- LDAPS 적설 `snol > 0`
- 기온 `temp < 273.15K`
- 습도 `RH >= 90%`

EDA 결과, 이 조건에서도 group3 발전량은 다음 모든 구간에 존재하였다.

- 0 발전
- 0~10%
- 10~20%
- 20~40%
- 40~60%
- 60~80%
- 80~100%

특히 눈 조건에서도 group3의 10% 이상 발전 sample이 상당수 존재하였다. 따라서 “눈이면 발전량이 0”이라는 hard rule은 성립하지 않는다. 눈 조건은 발전량을 0으로 만드는 결정 규칙이 아니라, 발전량 감소 위험을 높일 수 있는 risk feature 후보로 해석해야 한다.

이 분석은 모델링에서 매우 중요하다. rule-based로 `if snow then output=0`을 적용하면 실제 고발전 sample을 강제로 망가뜨릴 위험이 크다.

## 9. 후류 가설 검토

SCADA wind rose와 터빈 위치를 이용하여 후류 후보를 검토하였다. 후류 판단은 단순히 특정 방향에서 발전량이 낮다는 상관관계만으로 하지 않고, 바람 진행 방향과 터빈 위치 벡터를 이용한 물리적 alignment를 함께 고려하였다.

핵심 개념은 다음과 같다.

- downwind distance: 바람 진행 방향으로 뒤에 있는 정도
- crosswind distance: 중심 바람길에서 수직으로 벗어난 정도
- 후류 후보: downwind distance가 양수이고, crosswind distance가 rotor diameter의 몇 배 이내인 경우

SCADA에서는 일부 방향에서 후류 후보가 관찰되었으나, LDAPS/GFS 예보 풍향이 SCADA의 후류 위험 풍향 분포를 안정적으로 재현하지 못하였다. 따라서 후류 feature를 test 예측용 feature로 직접 이식하기에는 위험하다고 판단하였다.

결론적으로 후류는 물리적으로 존재할 수 있지만, 본 대회의 예보 feature만으로 안정적으로 활용하기 어려워 최종 v28에는 적극 반영하지 않았다.

## 10. 모델 실험 결과

### 10.1 v28 기반 알고리즘 비교

group3, 2024 holdout 기준으로 v28 feature 598개를 동일하게 사용하여 ExtraTrees, CatBoost, XGBoost를 비교하였다.

| 모델 | alpha | 1-NMAE | FICR | Local total |
|---|---:|---:|---:|---:|
| ExtraTrees v28 | 1.10 | 0.86208 | 0.29732 | 0.57970 |
| CatBoost v28 | 1.10 | 0.85992 | 0.29904 | 0.57948 |
| XGBoost v28 | 1.09 | 0.85810 | 0.28213 | 0.57011 |

CatBoost는 고발전 구간 과소예측을 일부 줄였지만, 전체 점수에서는 ExtraTrees를 안정적으로 이기지 못했다. 이후 group3만 CatBoost로 교체한 제출 후보도 리더보드에서 하락하였으므로 폐기하였다.

### 10.2 Feature slimming 실험

v28의 598개 feature 중 CF 60~80과 80~100 분리에 유리한 상위 40개 feature만 선택하여 group3 전용 모델을 만들었다.

결과적으로 top40 모델은 고발전 tail의 bias를 약간 줄였으나, 전체 NMAE와 FICR이 악화되었다.

| 모델 | feature 수 | alpha | 1-NMAE | FICR | Local total |
|---|---:|---:|---:|---:|---:|
| v28 ExtraTrees | 598 | 1.15 | 0.86205 | 0.30683 | 0.58444 |
| top40 ExtraTrees smoother | 40 | 1.15 | 0.85103 | 0.27834 | 0.56468 |
| top40 ExtraTrees | 40 | 1.15 | 0.85101 | 0.27343 | 0.56222 |

이는 v28의 넓은 feature 공간이 단순한 노이즈가 아니라, 계절·공간·상층풍·기상조건의 보조 신호를 일정 부분 제공하고 있음을 시사한다.

### 10.3 정격권 터빈 수 proxy 실험

SCADA 분석에서 “정격권 터빈 개수”가 group3 고발전과 관련이 있음을 확인했기 때문에, LDAPS 50m 풍속으로 각 터빈 위치의 정격권 proxy를 만들었다.

그러나 v28에 이 feature를 추가한 결과:

- 1-NMAE는 극히 미세하게 개선
- FICR은 하락
- 최종 local total은 v28보다 낮음

따라서 이 feature는 물리적으로 타당한 가설이지만, 예보 자료만으로는 실제 정격권 터빈 수를 충분히 복원하지 못한다고 판단하였다.

### 10.4 Classifier 기반 선택적 보정 실험

ExtraTrees leaf 분석에서 group3 고발전 sample이 leaf 내부 평균으로 수축되는 문제가 확인되었다. 이에 따라 단일 회귀 모델을 그대로 사용하는 대신, 먼저 “이 시각이 CF 80% 이상 고발전 regime인가”를 분류한 뒤, 고발전 가능성이 높은 sample에만 v28 예측값을 선택적으로 상향 보정하는 전략을 검토하였다.

분류 target은 다음과 같이 정의하였다.

- 양성 class: group3 actual CF >= 80%
- 음성 class: group3 actual CF < 80%

v28의 598개 feature를 그대로 사용하여 2024 holdout에서 classifier를 학습 및 검증하였다.

| 모델 | ROC-AUC | Average Precision | best F1 | best precision | best recall |
|---|---:|---:|---:|---:|---:|
| ExtraTrees Classifier | 0.9385 | 0.4791 | 0.5506 | 0.4326 | 0.7570 |
| RandomForest Classifier | 0.9309 | 0.4232 | 0.5300 | 0.3906 | 0.8244 |
| Logistic Regression | 0.9158 | 0.4329 | 0.4905 | 0.3714 | 0.7219 |
| CatBoost Classifier | 0.9152 | 0.4022 | 0.4809 | 0.3466 | 0.7851 |

ExtraTrees classifier는 baseline 고발전 비율 8.11% 대비 높은 lift를 보였다.

| 기준 | 선택 row 수 | 실제 CF>=80 비율 | recall | lift |
|---|---:|---:|---:|---:|
| Top 5% | 439 | 55.13% | 33.99% | 6.80배 |
| Top 10% | 878 | 46.92% | 57.87% | 5.79배 |
| Top 20% | 1,756 | 36.62% | 90.31% | 4.51배 |

2024 holdout에서는 classifier 확률 상위 15%이면서 v28 예측 CF가 0.55 이상인 sample에 alpha 1.12를 곱하는 선택적 보정이 가장 좋은 local score를 보였다.

| case | alpha | 선택 row 수 | 선택 sample의 실제 CF>=80 비율 | NMAE | FICR | Local total |
|---|---:|---:|---:|---:|---:|---:|
| v28 base | 1.00 | 0 | - | 0.13792 | 0.29732 | 0.57970 |
| Top 15%, pred CF>=0.55 | 1.12 | 1,309 | 41.94% | 0.13732 | 0.31898 | 0.59083 |

그러나 이 전략을 test 제출 후보로 변환한 v38은 leaderboard에서 성능이 하락하였다. 최종적으로 해당 전략은 폐기하였다.

이 실패는 중요한 실무적 결론을 제공한다. holdout에서 classifier 기반 보정이 좋아 보이더라도, public/private split의 고발전 regime 분포가 다르면 post-hoc calibration은 쉽게 일반화에 실패한다. 특히 선택 sample의 실제 고발전 비율이 약 40~50% 수준이면, 나머지 false positive sample의 과대예측이 FICR을 악화시킬 수 있다. 따라서 본 프로젝트에서는 classifier 기반 선택적 보정을 최종 모델에 포함하지 않고, v28 원본을 유지하였다.

## 11. ExtraTrees leaf 기반 평균 수축 분석

앞선 분석에서 v28 ExtraTrees는 group3 고발전 구간을 체계적으로 과소예측하였다. 특히 2024 holdout에서 실제 CF가 90~100%인 411개 sample 중, 예측 CF가 90~100%로 나온 경우는 0개였다. 이 중 304개는 실제 90~100%였음에도 예측은 60~80%에 머물렀다.

이 현상을 단순히 “모델이 평균으로 간다”고 설명하는 것은 부족하다. 회귀 tree에서 예측값은 전체 평균이 아니라, 입력 sample이 도달한 leaf 내부 train target의 평균이다. 따라서 실제로 평균 수축이 일어났는지 확인하기 위해, 다음 분석을 수행하였다.

분석 대상:

- 모델: v28 ExtraTrees
- group: group3
- 검증 기간: 2024 holdout
- 조건: 실제 CF 90~100%이면서 예측 CF 60~80%
- sample 수: 304개
- 분석 방식: 각 sample이 300개 tree에서 도달한 leaf를 찾고, 해당 leaf에 함께 들어간 train target CF 분포를 집계

결과는 다음과 같다.

| 항목 | 값 |
|---|---:|
| 분석 sample 수 | 304 |
| tree 수 | 300 |
| leaf 내 train target 가중 관측치 수 | 475,600 |
| leaf target CF 평균 | 0.653 |
| leaf target CF 중앙값 | 0.704 |
| leaf target CF 25% 분위수 | 0.501 |
| leaf target CF 75% 분위수 | 0.845 |
| leaf target CF 90% 분위수 | 0.930 |
| leaf target CF < 60% 비율 | 36.96% |
| leaf target CF 60~80% 비율 | 30.74% |
| leaf target CF >= 80% 비율 | 32.30% |
| leaf target CF >= 90% 비율 | 15.71% |

leaf 내부 train target 구간 분포는 다음과 같다.

| leaf 내부 train target CF 구간 | 가중 비율 |
|---|---:|
| 0~10% | 2.02% |
| 10~20% | 4.02% |
| 20~40% | 11.87% |
| 40~60% | 19.05% |
| 60~80% | 30.74% |
| 80~90% | 16.60% |
| 90~100% | 15.24% |
| 100% 초과 | 0.47% |

이 결과는 평균 수축 가설을 지지한다. 실제 90~100% 고발전 sample이 도달한 leaf는 고발전 sample만으로 구성되어 있지 않았다. 오히려 leaf 내부에는 40~80% 중·고발전 sample이 크게 섞여 있었고, 90~100% train sample의 비율은 약 15.2%에 불과했다. 따라서 leaf 평균은 0.65~0.70 수준으로 눌리고, 최종 예측도 60~80% 구간에 머물게 된다.

중요한 결론은 다음과 같다.

> v28 ExtraTrees가 group3 고발전을 과소예측한 이유는 전체 label 평균을 단순히 따라갔기 때문이 아니라, 예보 feature 공간에서 60~80% regime과 90~100% regime이 같은 leaf에 섞였기 때문이다.

이는 모델 개선 방향을 바꾼다. 단순히 고발전 예측값을 일괄적으로 올리는 alpha 보정은 저·중발전 구간의 FICR을 망가뜨릴 수 있다. 더 타당한 방향은 고발전 regime을 먼저 분리할 수 있는 feature 또는 classifier를 만들고, 그 조건에서만 회귀 출력을 선택적으로 보정하는 것이다.

## 12. 최종 채택 및 폐기 결정

최종적으로 다음 판단을 내렸다.

채택:

- v28 ExtraTrees 기반 제출 구조
- LDAPS/GFS 풍속, 풍속 세제곱, 거리 가중 feature
- GFS grid5 상층풍 feature
- group별 독립 모델

폐기:

- group3 CatBoost 교체 제출 후보
- group3 CF>=80 classifier 기반 선택적 보정 제출 후보
- top40 feature slimming 모델
- 정격권 터빈 수 proxy 추가 모델
- hard snow rule
- 예보 기반 wake rule
- 과도한 후처리 calibration

## 13. 연구적 결론

본 프로젝트의 핵심 결론은 다음과 같다.

1. group3는 group1/2보다 label 기간이 짧고 저발전 sample 비중이 크다.
2. group3의 80~100% 고발전 sample은 전체의 약 7.8%로 희소하다.
3. SCADA 기준으로는 정격권 터빈 수가 고발전과 강하게 관련된다.
4. 그러나 LDAPS/GFS 예보 feature만으로 60~80%와 80~100%를 안정적으로 분리하기는 어렵다.
5. 모델을 CatBoost/XGBoost로 바꿔도 고발전 과소예측은 완전히 해결되지 않는다.
6. 눈/적설/영하/고습도 조건은 발전량 0의 rule이 아니라 risk feature로 해석해야 한다.
7. 후류는 SCADA에서 탐지 가능성이 있으나, 예보 feature로 안정적으로 이전하기 어렵다.
8. ExtraTrees leaf 분석 결과, 실제 고발전 sample이 도달한 leaf 내부에 40~80% train sample이 크게 섞여 있어 평균 수축이 발생하였다.
9. classifier 기반 선택적 보정은 2024 holdout에서는 개선되었지만 leaderboard에서는 하락하여, post-hoc calibration의 일반화 위험을 확인하였다.
10. 따라서 현재 성능 병목은 단순 모델 선택 문제가 아니라, 예보 feature의 regime 분리력 한계와 group3 label 분포 불균형, 그리고 public/private 분포 차이가 결합된 문제다.

## 14. 포트폴리오용 해석

본 프로젝트는 단순히 리더보드 점수를 올리는 시도가 아니라, 다음 흐름을 갖는 분석 프로젝트로 정리할 수 있다.

1. Baseline 모델 실행 및 제출 가능성 확인
2. 풍력 물리식 기반 feature 설계
3. 터빈 위치와 grid 거리 기반 ontology feature 생성
4. LDAPS/GFS 예보와 SCADA 관측의 관계 검토
5. group3의 label 기간·분포 차이 발견
6. 고발전 과소예측 문제 정의
7. SCADA 기반 정격권 터빈 수 분석
8. 예보 feature의 고발전 regime 분리력 검증
9. 대체 모델 및 feature slimming 실험
10. ExtraTrees leaf 분석을 통한 평균 수축 원인 확인
11. classifier 기반 선택적 보정 실험과 leaderboard 일반화 실패 확인
12. 실패한 feature와 모델을 근거 기반으로 폐기
13. 최종적으로 v28을 안정 기준 모델로 고정

이 흐름은 취업 포트폴리오에서 “모델을 많이 돌렸다”가 아니라, “도메인 가설을 세우고, 반박 가능한 방식으로 검증하고, 실패한 가설을 폐기했다”는 점을 보여줄 수 있다.

## 15. 향후 과제

향후 추가 연구는 다음 방향이 적절하다.

- group3 고발전 regime을 직접 분류하는 별도 classifier와 회귀 모델의 결합
- 예보 풍속의 공간장 전체를 활용하는 compact spatial representation
- 월별/계절별 regime model
- 예보 오차 보정 모델
- SCADA 기반 사후 설명 모델과 제출용 예보 모델의 명확한 분리
- 2025 public/private split에 강건한 validation 체계 재설계

단, 새로운 feature를 추가하기 전에 반드시 “해당 feature가 60~80과 80~100을 분리하는가”를 먼저 검증해야 한다.
