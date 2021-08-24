
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
#scikit-learn: Python 언어를 통한 머신러닝의 대부분을 손쉽게 사용가능하도록, numpy와 scipy를 기반을 작성된 라이브러리
import scipy.stats as stats
import itertools
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pylab

import warnings
warnings.filterwarnings('ignore')

#선형 회귀분석 실습
#다중 선형 회귀 모형 : Multivariate Linear Regression
#1번 ram price dataset

ram_prices = pd.read_csv('../Practice/Data/회귀_data/ram_price.csv')
print(ram_prices)
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Year")
plt.ylabel("Price ($/Mbyte)")
plt.show()

plt.plot(ram_prices.date,ram_prices.price,'--')
plt.xlabel('Year')
plt.ylabel('Raw Price')
plt.show()

##sklearn.linear_model.LinearRegression( )함수를 이용해서 선형회귀를 수행
from sklearn.linear_model import LinearRegression
## 2000년 이전을 훈련 데이터, 2000년 이후를 테스트 데이터
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# 가격 예측을 위해 날짜 특성만을 이용
X_train = data_train.date[:, np.newaxis]

# 데이터와 타깃 사이의 관계를 간단하게 만들기 위해 로그 스케일로 변환
y_train = np.log(data_train.price)

#실제로 다중선형회귀모델의 함수를 정의하고 학습하는 구문
linear_reg = LinearRegression().fit(X_train, y_train)

# 예측은 전체 기간에 대해서 수행합니다
X_all = ram_prices.date[:, np.newaxis]

pred_lr = linear_reg.predict(X_all)

# 예측한 값의 로그 스케일을 되돌립니다.
price_lr = np.exp(pred_lr)
print(price_lr)

#실제로 예측된 값을 시각화하여 확인
plt.semilogy(data_train.date, data_train.price, label="Training Data")
plt.semilogy(data_test.date, data_test.price, label="Test Data")
plt.semilogy(ram_prices.date, price_lr, label="Linear Regression Prediction")
plt.legend()
plt.show()

#회귀계수 확인해보기
linear_reg.coef_
print('연도에 따른 회귀계수(로그변환):', linear_reg.coef_)

regressiomn_model = LinearRegression()
#fit을 통해 모델이 통계계수들을 추정
regressiomn_model.fit(X_train,y_train)
#추정이 어떻게 되었는지를 반환
regressiomn_model.coef_
regressiomn_model.predict(X_all)


#**********************************************************#
#2번 California Housing Data
'''
데이터 구조
데이터: 1990년 캘리포니아의 각 행정 구역 내 주택 가격
관측치 개수: 20640개
변수 개수: 설명변수 8개 / 반응변수 1개

설명 변수(예측값을 설명할 수 있는 변수)
MedInc : 행정 구역 내 소득의 중앙값
HouseAge : 행정 구역 내 주택 연식의 중앙값
AveRooms : 평균 방 갯수
AveBedrms : 평균 침실 갯수
Population : 행정 구역 내 인구 수
AveOccup : 평균 자가 비율
Latitude : 해당 행정 구역의 위도
Longitude : 해당 행정 구역의 경도

반응 변수(예측하고자 하는 값)
House Value: 주택가격'''
from IPython.display import display, HTML
# 데이터 전처리 패키지
import numpy as np
import pandas as pd

# 기계학습 모델 구축 및 성능 평가 패키지
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy as sp
import scipy.stats as stats
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# 데이터 시각화 패키지
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')

california = fetch_california_housing()
print(california.DESCR)
print(type(california))

X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.DataFrame(california.target, columns=['House Value'])
print(X.head())
print(y.head())

data = pd.concat([X, y], axis=1)
print(type(data),data.shape, data.head(), sep='\n')
##모델을 학습 & 테스트 데이터로 분리
#아래 함수는 전체 데이터를 무작위로 학습과 테스트 데이터로 분리해줌
#random_state 난수 생성 시드
train_data, test_data = train_test_split(data, test_size=0.3, random_state=1234)
print(type(train_data),train_data.shape, train_data.head(), sep='\n')
print(type(test_data),test_data.shape,test_data.head(), sep='\n')

#모델링
#OLS: 가장 기본적인 결정론적 선형 회귀 방법으로 잔차제곱합(RSS: Residual Sum of Squares)를 최소화하는 가중치(β1, β2 ...) 구하는 방법
#모델 선언: model = sm.OLS(Y데이터, X데이터)
#모델 학습: model_trained = model.fit()

lm = sm.OLS(train_data['House Value'], train_data.drop(['House Value'], axis=1))
lm_trained = lm.fit()

display(lm_trained.summary())

#학습 데이터 (Training Data)에 대한 예측 및 성능 평가
train_pred = lm_trained.predict(train_data.drop(['House Value'], axis=1))
print(train_pred)


plt.figure(figsize=(6, 6))
plt.title('실제값 vs. 예측값')
plt.scatter(train_data['House Value'], train_pred)
plt.xlabel('실제값', size=16)
plt.ylabel('예측값', size=16)
plt.xlim(-2, 8)
plt.ylim(-2, 8)
plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score   #결정계수
#실제값과 예측된 값을 기준으로 True값 대비 예측오차가 몇 %인지에 대한 평균값
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100


print('Training MSE: {:.3f}'.format(mean_squared_error(train_data['House Value'], train_pred)))
print('Training RMSE: {:.3f}'.format(np.sqrt(mean_squared_error(train_data['House Value'], train_pred))))
print('Training MAE: {:.3f}'.format(mean_absolute_error(train_data['House Value'], train_pred)))
print('Training MAPE: {:.3f}'.format(mean_absolute_percentage_error(train_data['House Value'], train_pred)))
print('Training R2: {:.3f}'.format(r2_score(train_data['House Value'], train_pred)))

print('-'*50)
#테스트 데이터 (Testing Data)에 대한 예측 성능 평가
test_pred = lm_trained.predict(test_data.drop(['House Value'], axis=1))
display(test_pred)

plt.figure(figsize=(6, 6))
plt.title('실제값 vs. 예측값')
plt.scatter(test_data['House Value'], test_pred)
plt.xlabel('실제값', size=16)
plt.ylabel('예측값', size=16)
plt.xlim(-2, 8)
plt.ylim(-2, 8)
plt.show()

print('Testing MSE: {:.3f}'.format(mean_squared_error(test_data['House Value'], test_pred)))
print('Testing RMSE: {:.3f}'.format(np.sqrt(mean_squared_error(test_data['House Value'], test_pred))))
print('Testing MAE: {:.3f}'.format(mean_absolute_error(test_data['House Value'], test_pred)))
print('Testing MAPE: {:.3f}'.format(mean_absolute_percentage_error(test_data['House Value'], test_pred)))
print('Testing R2: {:.3f}'.format(r2_score(test_data['House Value'], test_pred)))

print('-'*50)

