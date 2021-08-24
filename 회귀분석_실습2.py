
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

