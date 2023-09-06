# 딥러닝 네트워크를 활용한 주식 가격 결정 모델 (Asset pricing models using deep learning networks)

![Alt text](ppt/%EB%A0%88%ED%8D%BC%EB%9F%B0%EC%8A%A4.png)

## 개요

딥러닝 네트워크 기반의 알고리즘이 발전됨에 따라 인공지능은 전 세계적으로 빠른 성장세를 보인다. 인공지능은 다양한 분야에서 적용되고 있으며, 빅데이터와 클라우드 컴퓨팅의 발전 등으로 인해 더욱 빠른 성장세를 보이고 있다. 그 중, 금융은 인공지능이 가장 많이 활용될 분야로 예상되어 최근 많은 연구가 되고 있다.

금융에서, 잠재 팩터를 알아낼 수 있다면 알파라고 여겨왔던 비정상 초과 수익률을 해석할 수 있게 된다. 본 연구는 리스크 팩터에 자산이 얼마나 동적으로 노출되었는지를 딥러닝 네트워크를 통해 검증한 연구(Gu, Kelly, and Xiu, 2021)를 바탕으로, 한국의 시장 요인과 외부 요인을 딥러닝 네트워크에 입력으로 제공해 팩터의 비선형적 관계를 해석하고 초과 수익률을 해석하고자 한다.

## 현황

Gu, Kelly, and Xiu(GKX, 2021)은 기존의 재무 개념과 머신러닝 기술인 오토인코더를 결합하여 비선형 관계를 더 잘 파악할 수 있는 잠재 요인 모델링 방식으로 자산 가격 매커니즘을 더 잘 이해하고 예측 정확성을 향상시키는 방법을 제안했다.

Kelly, Pruitt, and Su(2019)에서 사용되는 전통적인 instrumental beta에 신경망 임베드를 적용해 보다 유연한 베타간 유연한 비선형성 및 상호작용으로 금융 시장의 리스크-수익률 분석에 머신 러닝 기술을 적용하는 데 기여했다. 해당 논문에서 제안된 오토인코더 모델은 Fama-French 모델, PCA 방법 및 IPCA와 같은 선형 조건화 방법과 경쟁 모델 보다 우수한 성능을 보인다.

추가적으로 이 논문은 복잡한 금융 데이터를 모델링하는 데 있어 딥러닝 기법의 효과를 입증하고, 모델 성능을 개선하는 데 있어 Earlystopping, Ensemble, Batch Normalization등의 기법이 얼마나 중요한지 강조한다. 이러한 딥러닝 기법과 전통적인 재무 개념의 결합은 자산 특성/요소 간의 비선형 관계를 효과적으로 파악할 수 있음을 보여준다.

## 요약

본 연구는 데이터 준비와 피처 엔지니어링, 딥러닝 모델의 아키텍처 결정과 하이퍼파라미터 설정, 그리고 제너레이터를 사용하여 미니배치 데이터를 모델에 공급하는 단계로 구성된다. 이어서 AlphaLens 라이브러리를 사용하여 모델의 성능을 평가하고 그 결과를 시각화한다. 즉, 데이터 준비부터 모델 생성, 성능 평가까지 전 과정을 아우르는 종합적인 딥러닝 분석을 진행한다.

## 데이터 상세

### 데이터 수집

먼저 FinanceDataReader를 이용하여 KRX 거래소에 상장된 데이터를 모두 수집하였다. 이 중 Yahoo Finance에서 검색이 가능한 KOSPI, 로 구성하고 KOSDAQ과 KONEX는 제거하였다. Yahoo Finance에서 검색이 가능하도록 “종목코드.KS”로 레이블을 지정하였으며, 상장폐지 되지 않은 882개의 데이터 중 메타 데이터가 존재하지 않는 24개의 데이터를 제외한 858개의 종목으로 데이터를 수집할 수 있었다. 최종적으로 사용할 데이터의 종목 수는 858개, 수집 기간은 2000-01-04 ~ 2023-05-08이고, OHLCV 데이터 주기는 일별 데이터, 수익률은 주간으로 계산하였다.

### Firm characteristic 데이터의 생성

GKX(2021)는 총 94개의 자산 특성을 사용하였으나, 이 중 가장 영향력 있는 20개의 지표를 확인할 수 있었다. 이러한 지표는 모멘텀, 장단기 반전, 최근 최대 수익률, 회전율, 거래량, 시가총액 등이며, 다른 특성들은 모델에 큰 영향을 주지 못한다는 것을 보여준다. 따라서 우리는 20개의 지표를 생성하는 것을 목적으로 하되, 본 연구는 한국 시장을 분석하는 것이므로 OHLCV Data와 Metadata를 활용하여 생성할 수 있는 16개의 특성을 사용한다.

## Firm Characteristics

|                                 |                                                                                                                                                 |   |
|:-------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------:|---|
|       Short-Term Reversal       |                                                            1-month cumulative return                                                            |   |
|          Stock Momentum         |                                           11-month cumulative returns ending 1-month before month end                                           |   |
|         Momentum Change         |                                        Cumulative return from months t-6 to t-1 minus months t-12 to t-7                                        |   |
|        Industry Momentum        |                                                  Equal-weighted avg. industry 12-month returns                                                  |   |
|        Recent Max Return        |                                                    Max daily returns from calendar month t-1                                                    |   |
|        Long-Term Reversal       |                                                     Cumulative returns months t-36 to t-13.                                                     |   |
|             Turnover            |                               Avg. monthly trading volume for most recent three months scaled by number of shares                               |   |
|       Turnover Volatility       |                                                     Monthly std dev of daily share turnover                                                     |   |
|        Log Market Equity        |                                                  Natural log of market cap at end of month t-1                                                  |   |
|            KRW Volume           |                                        Natural log of trading volume time price per share from month t-2                                        |   |
|        Amihud Illiquidity       |                                                 Average of daily (absolute return / KRW volume)                                                 |   |
|          Risk Measures          |                                                  Standard dev of daily returns from month t-1.                                                  |   |
|           Market Beta           |   Estimated market beta from weekly returns and equal weighted market returns for 3 years ending month t-1 with at least 52 weeks of returns.   |   |
|           Beta Squared          |                                                               Market beta squared                                                               |   |
| Idiosyncratic return volatility | Standard dev of a regression of residuals of weekly returns on the returns of an equal weighted market index returns for the prior three years. |   |

### Price Trend

+ **Short-Term** Reversal 1-month cumulative return
+ **Stock Momentum** 11-month cumulative returns ending 1-month before month end
+ **Momentum** Change Cumulative return from months t-6 to t-1 +minus months t-12 to t-7
+ I**ndustry Momentum Equal-weighted avg.** industry 12-month returns
+ **Recent Max Return** Max daily returns from calendar month t-1
+ **Long-Term Reversal** Cumulative returns months t-36 to t-13

### Liquidity Metrics

+ **Turnover Avg.** monthly trading volume for most recent three months scaled by number of shares
+ **Turnover Volatility** Monthly std dev of daily share turnover
+ **Log Market Equity** Natural log of market cap at end of month t-1
+ **Volume** Natural log of trading volume time price per share from month t-2
+ **Amihud Illiquidity** Average of daily (absolute return / dollar volume)

### Risk Measures

+ **Return Volatility** Standard dev of daily returns from month t-1
+ **Market Beta** Estimated market beta from weekly returns and equal weighted market returns for 3 years ending month t-1 with at least 52 weeks of returns
+ **Beta Squared** Market beta squared
+ **Idiosyncratic return volatility** Standard deviation of weekly return residuals regressed on equal weighted market index returns for the prior three years.

### Meta data

+ **Return** weekly returns

## Modeling

### input_beta

각 자산은 15개의 베타를 가지고 있고, 이들은 [8, 16, 32]개의 유닛을 가진 히든 레이어를 통과하여 베타 간의 비선형성을 분석한다. 그 다음, ReLU 활성화 함수를 거친 후 배치 정규화를 수행하여 스무딩 효과를 통해 글로벌 최적점에 더 쉽게 도달할 수 있도록 조절한다. 마지막으로, K개의 팩터로 레이어 가중치를 전달할 때 K는 [2, 3, 4, 5, 6] 중 하나로 설정되어 실험이 진행된다. 오른쪽에는 유닛이 8개이고 K가 3인 경우의 모델이 그려져 있다. 이 K개의 요인은 베타의 비선형성을 잘 나타내는 방향으로 학습되며, input_factor와 내적한 값을 최종적으로 출력한다.

### Input_factor

개별 주식의 수익률을 입력으로 넣고, 베타의 개수 레이어, K개의 팩터 레이어를 거쳐 내적한다. GKX(2021)는 베타의 개수와 대응하는 레이어를 추가하는 것을 추천한다. 이는 각 특성에 따라 구성된 포트폴리오의 수익률을 뜻하게 되며, 모델을 더 경량화할 수 있고 Panel Imbalance 등을 해결할 수 있게 된다.

## Data Generator

### Generator

신경망 모델을 학습하기 위한 입력 데이터의 전처리는 중요하다. 대규모 데이터를 정확한 형식으로 정리하는 것은 쉽지 않으며 전체 데이터를 메모리에 올릴 수 없는 상황이 발생할 수도 있다. 반면 제너레이터는 데이터에 접근할 때마다 메모리에 적재하는 방식을 사용하여 메모리를 효율적으로 할당할 수 있다. 이를 통해 Lazy evaluation이 가능해지며, 실제로 필요한 값만 계산되어 메모리 사용량을 줄일 수 있다.

### 주식 시장 데이터

주식 시장 데이터는 과거부터 현재까지 약 20년 이상의 데이터가 각 종목마다 존재하여 요구하는 메모리 사용량이 많으므로, 모델을 학습하는 도중 Out-Of-Memory가 발생하지 않도록 코드 최적화가 필수적이다.

## Quant & Statistical Analysis

Period Wise Return은 [5, 10, 21]일 단위로 계산한 종목 수익률의 변동폭을 나타낸 값이고, 왼쪽 그림은 해당 종목에 부여된 Factor 값의 분위로 구분하여 나타낸 그림이다. Factor의 분위는 각 종목마다 계산된 값으로 결정되며, 약 -1.14 ~ 0.24까지의 분포를 띈다. 예를 들어, Factor의 영향이 제일 적은 집합인 3분위의 경우 수익률의 변동폭이 매우 적은 것을 알 수 있다. 오른쪽 그림은 테스트 기간인 2019년부터 2023년까지 Top minus Bottom Quantile의 평균 수익률을 나타낸 것인데, 이동평균선이 대체로 양수에 존재하며 Factor가 대체로 시장을 잘 설명하는 것을 보여준다.

![Alt text](ppt/%EA%B7%B8%EB%A6%BC2.png)

정보 계수(IC)는 예측된 주식 수익률과 실제 주식 수익률 간의 상관관계를 나타내며, 애널리스트의 기여도를 측정하는 데 사용되기도 한다. 정보 계수가 +1이면 예측 수익률과 실제 수익률 사이에 완벽한 선형 관계가 있음을 나타내고, 0이면 선형 관계가 없음을 나타낸다. 아래 그림은 2019년부터 2023년까지 테스트한 성과이다. 왼쪽부터 차례대로 IC의 이동평균선, IC의 분포, IC의 Q-Q Plot, 월별 IC의 평균을 의미한다. IC의 평균이 양수이고, 정규분포를 따르며, 월별 평균 IC도 지속적으로 양수인 구간이 많아, Factor가 시장을 잘 설명하고 있음을 확인할 수 있다.

![Alt text](ppt/%EA%B7%B8%EB%A6%BC3.png)

## 결론

본 연구는 GKX(2021)에 착안하여 한국 시장의 Firm Characteristics을 반영한 딥러닝 모델을 구현하는 것이 목적이었다. 이에 데이터 준비부터 모델링, 학습, 평가의 일련의 절차를 통해 한국 시장을 잘 설명하는 Factor를 찾을 수 있었다. 해당 Factor로 구한 정보 계수(IC) 값의 평균이 0보다 크고 정규 분포에 가까우므로 KOSPI 시장을 설명하는데 기여한다고 할 수 있을 것이다.

GKX(2021)는 미국 주식시장 분석을 위해 CRSP 데이터를 이용, [NYES, AMEX, NASDAQ]의 모든 종목에 대해 1957년부터 2016년까지 총 60년치의 주식 데이터 및 94개의 종목 특성을 사용했다. 총 3만 여개의 종목, 한 달에 약 6200개의 종목으로 모델을 만들었으며, 1957년부터 1974년까지 학습, 1975년부터 1986년까지 검증, 1987년부터 2016년까지 30년을 테스트했다. 한국 시장은 미국에 비해 역사가 오래되지 않아 절대적인 데이터의 수가 부족하다. 같은 방법으로 나스닥 시장에 적용했을 경우 약 2500개의 주식 종목으로 학습을 진행할 수 있었던 반면, KOSPI는 약 800개에 그쳤으며, 자산 특성을 구할 수 있는 메타 데이터도 얻기 어려웠다.

추후 다양한 백테스트 기법 및 통계적 검정 방법으로 모델이 강건성을 띄는지도 추가적으로 확인할 계획이다. IC의 값이 유의미하게 도출되었으나, 그 평균이 0.1을 넘지 못하므로 시장을 더 잘 설명할 수 있는 모델을 학습하는 것이 중요하다. 또한, BatchNorm이나 Ensemble 같은 딥러닝 최적화 기법을 통해서도 모델의 성능을 효과적으로 높일 수 있을 것으로 보인다.

##### References
Gu, S., Kelly, B., & Xiu, D. (2021). Autoencoder asset pricing models. Journal of Econometrics, 222(1), 429-450.
Kelly, B. T., Pruitt, S., & Su, Y. (2019). Characteristics are covariances: A unified model of risk and return. Journal of Financial Economics, 134(3), 501-524.

##### Code references
[https://github.com/Stefan-jansen/machine-learning-for-trading]
[https://github.com/syasini/Python_Generators]
