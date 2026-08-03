# Wind Power Forecasting Portfolio

DACON 풍력발전량 예측 경진대회를 기반으로 수행한 풍력발전량 예측 모델링 프로젝트입니다.  
기상예보 데이터(LDAPS/GFS), 터빈 SCADA 관측 데이터, 터빈 위치 메타정보를 활용하여 KPX 그룹별 시간 단위 발전량을 예측하고, 모델의 실패 원인을 도메인 관점에서 분석했습니다.

> 최종 기준 모델은 `v28 ExtraTrees`입니다. 이후 여러 고도화 실험을 수행했지만, 리더보드 일반화 성능을 기준으로 v28을 최종 기준선으로 고정했습니다.

## Project Goal

예측 기준 시점에 실제로 사용 가능한 기상예보 데이터만을 이용하여 미래 12~35시간의 풍력발전량을 예측합니다.

평가 지표는 다음 두 가지입니다.

- `1-NMAE`: 설비용량으로 정규화한 평균 절대 오차
- `FICR`: 시간별 예측 오차율이 6%/8% 이내에 들어오는 정도를 정산금 관점에서 평가한 지표

공식 평가에서는 실제 발전량이 설비용량의 10% 이상인 시간만 평가 대상이 됩니다. 따라서 단순히 평균 오차를 낮추는 것뿐 아니라, FICR 경계 안에 안정적으로 들어오는 예측이 중요합니다.

## Data Used

이 저장소에는 대회 원본 데이터가 포함되어 있지 않습니다.  
`train/`, `test/`, `info.xlsx`, 제출 CSV 파일은 `.gitignore`로 제외했습니다.

사용한 데이터 구조는 다음과 같습니다.

- `train_labels.csv`: KPX group1, group2, group3 시간별 실제 발전량
- `ldaps_train/test.csv`: LDAPS 지역 수치예보 데이터
- `gfs_train/test.csv`: GFS 전지구 수치예보 데이터
- `scada_vestas_train.csv`: VESTAS 터빈 10분 단위 SCADA
- `scada_unison_train.csv`: UNISON 터빈 10분 단위 SCADA
- `info.xlsx`: 터빈 위치, 제조사, rotor diameter, hub height, 설비용량 메타정보

## Modeling Baseline

최종 기준 모델은 `baseline_ontology_v28_gfs_grid5.py`입니다.

핵심 구성:

- 그룹별 독립 `ExtraTreesRegressor`
- target을 발전량이 아닌 capacity factor로 변환하여 학습
- LDAPS/GFS 풍속 벡터에서 풍속 크기, 풍속 제곱, 풍속 세제곱 생성
- 터빈 위치 기반 거리 가중 grid feature 생성
- GFS grid5 상층풍 feature 추가
- GFS 10m/100m wind shear를 이용한 hub 117m proxy 생성

## Analysis Workflow

프로젝트는 다음 흐름으로 진행했습니다.

1. DACON baseline 실행 및 제출 가능 파일 검증
2. 풍력 물리식 기반 feature 설계
3. 터빈 위치와 grid 거리 기반 ontology feature 생성
4. LDAPS/GFS 예보 풍속과 SCADA 터빈 풍속 비교
5. group3의 label 기간 및 target 분포 차이 확인
6. group3 고발전 과소예측 문제 정의
7. SCADA 기반 정격권 터빈 수 분석
8. v28 ExtraTrees leaf 분석으로 평균 수축 원인 확인
9. CF>=80 classifier 기반 선택적 보정 실험
10. leaderboard에서 일반화되지 않는 실험 폐기
11. 최종적으로 v28 기준 모델 유지

## Key Insights

### 1. Group3는 group1/2와 다른 문제였다

group1/2는 2022~2024년 label이 존재하지만, group3는 2023~2024년 label만 존재합니다.  
group3의 유효 label 17,537개 중 설비용량 10% 미만 발전 구간은 약 46.3%였고, 80~100% 고발전 구간은 약 7.8%에 불과했습니다.

이 때문에 group3는 고발전 sample이 희소하고, 회귀 모델이 고발전 구간을 평균 쪽으로 끌어내리기 쉬운 구조였습니다.

### 2. 고발전 과소예측은 단순 모델 선택 문제가 아니었다

ExtraTrees, CatBoost, XGBoost를 같은 v28 feature로 비교했지만, group3 고발전 과소예측은 완전히 해결되지 않았습니다.

2024 holdout 기준:

- CatBoost는 고발전 bias를 일부 줄였지만 리더보드에서는 하락했습니다.
- XGBoost도 전체 성능에서 v28 ExtraTrees를 넘지 못했습니다.
- group3만 CatBoost로 교체한 제출 후보도 폐기했습니다.

### 3. ExtraTrees leaf 내부에서 평균 수축이 실제로 확인되었다

실제 group3 CF가 90~100%인데 v28 예측 CF가 60~80%에 머문 sample 304개를 추출했습니다.  
각 sample이 ExtraTrees의 300개 tree에서 도달한 leaf를 추적하고, 그 leaf에 들어간 train target CF 분포를 분석했습니다.

핵심 결과:

- leaf target CF 평균: 0.653
- leaf target CF 중앙값: 0.704
- leaf target CF < 60% 비율: 36.96%
- leaf target CF 60~80% 비율: 30.74%
- leaf target CF >= 90% 비율: 15.71%

즉, 실제 고발전 sample이 도달한 leaf 내부에는 90~100% sample만 모여 있지 않았고, 40~80% sample이 크게 섞여 있었습니다.  
따라서 v28의 고발전 과소예측은 전체 평균 때문이 아니라, feature 공간에서 60~80%와 90~100% regime이 같은 leaf에 섞인 결과로 해석했습니다.

### 4. Classifier 기반 선택적 보정은 holdout에서는 성공했지만 leaderboard에서는 실패했다

v28 feature로 group3 `CF>=80%` classifier를 학습했을 때, 2024 holdout에서 ExtraTrees classifier는 다음 성능을 보였습니다.

- ROC-AUC: 0.9385
- Average Precision: 0.4791
- Top 10% precision: 46.92%
- Top 10% recall: 57.87%

이를 바탕으로 classifier가 고발전 가능성이 높다고 판단한 sample에만 v28 예측값을 선택적으로 상향 보정했습니다.  
2024 holdout에서는 FICR이 개선되었지만, 실제 leaderboard에서는 점수가 하락했습니다.

이 실험을 통해 post-hoc calibration은 public/private distribution shift에 취약하다는 점을 확인했고, 최종 모델에서는 선택적 보정을 사용하지 않았습니다.

### 5. 눈, 적설, 후류 feature는 hard rule로 사용하지 않았다

눈/적설/영하/고습도 조건에서도 group3 발전량은 0부터 80~100%까지 모든 구간에 존재했습니다.  
따라서 `snow then output=0` 같은 hard rule은 성립하지 않았습니다.

후류 역시 SCADA에서는 일부 후보가 관찰되었지만, LDAPS/GFS 예보 풍향이 SCADA 후류 위험 방향을 안정적으로 재현하지 못했습니다.  
따라서 후류 feature도 최종 제출 모델에는 포함하지 않았습니다.

## Experiments Kept and Rejected

채택한 방향:

- v28 ExtraTrees 기준 모델
- LDAPS/GFS 풍속 ontology feature
- 거리 가중 grid feature
- GFS grid5 상층풍 feature
- 풍속 세제곱 feature

폐기한 방향:

- group3 CatBoost 교체 제출 후보
- group3 CF>=80 classifier 기반 선택적 보정 제출 후보
- top40 feature slimming 모델
- 정격권 터빈 수 proxy 추가 모델
- snow hard rule
- forecast wake rule
- 과도한 alpha calibration

## Portfolio Report

최종 논문형 보고서는 아래 PDF에 정리했습니다.

- [`output/pdf/wind_power_v28_group3_portfolio_report_v3_final.pdf`](output/pdf/wind_power_v28_group3_portfolio_report_v3_final.pdf)

보고서에는 다음 내용이 포함되어 있습니다.

- 문제 정의와 평가 지표
- v28 기준 모델 구조
- group3 target 분포 EDA
- SCADA 기반 정격 풍속/정격권 터빈 수 분석
- 고발전 regime 분리 가능성 분석
- ExtraTrees leaf 평균 수축 분석
- classifier 기반 선택적 보정 실험
- 실패 실험의 폐기 근거

## Repository Notes

대회 원본 데이터와 제출 파일은 포함하지 않았습니다.  
재현을 위해서는 DACON에서 제공한 원본 데이터를 아래 구조로 배치해야 합니다.

```text
.
├── train/
├── test/
├── info.xlsx
├── sample_submission.csv
└── baseline_ontology_v28_gfs_grid5.py
```

## Main Files

- `portfolio_wind_power_v28_group3_report.md`: 포트폴리오 보고서 원문
- `make_wind_portfolio_pdf.py`: Markdown 보고서를 PDF로 변환하는 스크립트
- `analyze_group3_v28_leaf_shrinkage.py`: ExtraTrees leaf 평균 수축 분석
- `diagnose_group3_cf80_classifier.py`: group3 CF>=80 classifier 진단
- `test_group3_selective_calibration.py`: classifier 기반 선택적 보정 holdout 실험

## Final Conclusion

본 프로젝트의 최종 결론은 다음과 같습니다.

> v28 ExtraTrees는 group3 고발전 구간을 과소예측하지만, 단순 모델 교체나 후처리 보정은 leaderboard에서 안정적으로 일반화되지 않았다. 고발전 과소예측의 핵심 원인은 예보 feature 공간에서 60~80%와 90~100% 발전 regime이 충분히 분리되지 않아, ExtraTrees leaf 내부 평균으로 예측이 수축되는 현상이다.

따라서 이 프로젝트는 단순히 점수를 올리는 모델 실험이 아니라, 풍력 도메인 지식과 모델 내부 진단을 연결하여 실패 원인을 추적한 분석 프로젝트로 정리했습니다.
