
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
# 현재경로 확인
print(os.getcwd())

# 데이터 불러오기
boston = pd.read_csv("D:/Practice/Data/part2_data/Boston_house.csv")
print(boston.head())

# target 제외한 데이터만 뽑기
boston_data = boston.drop(['Target'],axis=1)

# data 통계 뽑아보기
boston_data.describe()

'''
타겟 데이터
1978 보스턴 주택 가격
506개 타운의 주택 가격 중앙값 (단위 1,000 달러)

특징 데이터
CRIM: 범죄율
INDUS: 비소매상업지역 면적 비율
NOX: 일산화질소 농도
RM: 주택당 방 수
LSTAT: 인구 중 하위 계층 비율
B: 인구 중 흑인 비율
PTRATIO: 학생/교사 비율
ZN: 25,000 평방피트를 초과 거주지역 비율
CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
AGE: 1940년 이전에 건축된 주택의 비율
RAD: 방사형 고속도로까지의 거리
DIS: 직업센터의 거리
TAX: 재산세율'''

##crim/rm/lstat 세게의 변수로 각각 단순 선형 회귀 분석하기
# 변수 설정 target/crim/rm/lstat
target = boston[['Target']]
crim = boston[['CRIM']]
rm = boston[['RM']]
lstat = boston[['LSTAT']]

##target ~ crim 선형회귀분석
# crim변수에 상수항추가하기
crim1=sm.add_constant(crim,has_constant='add')
print(crim1)

# sm.OLS 적합시키기
model1 = sm.OLS(target,crim1)
fitted_model1 = model1.fit()

# summary함수통해 결과출력
print(fitted_model1.summary())
##해석 - R-squared : 0.151 -> 범죄율(X)이 설명하는 Y의 총 변동성은 약 15% 정도
##범죄율(X)에 해당하는 회귀계수값(coef)은 -0.4152, 그때 P-Value는 매우 유의미하다.

## 회귀 계수 출력
fitted_model1.params

##y_hat=beta0 + beta1 * X 계산해보기
#회귀 계수 x 데이터(X)
np.dot(crim1,fitted_model1.params)

## predict함수를 통해 yhat구하기
pred1=fitted_model1.predict(crim1)

## 직접구한 yhat과 predict함수를 통해 구한 yhat차이  --> 0이 나와야 정상 !!!
np.dot(crim1,fitted_model1.params) - pred1

#적합시킨 직선 시각화 해보기
import matplotlib.pyplot as plt
plt.yticks(fontname = "Arial") #
plt.scatter(crim,target,label="data")#범죄와 타겟에 대한 산점도
plt.plot(crim,pred1,label="result")
plt.legend()
plt.show()

plt.scatter(target,pred1)
plt.xlabel("real_value")
plt.ylabel("pred_value")
plt.show()

## residual 시각화 -> 잔차
fitted_model1.resid.plot()
plt.xlabel("residual_number")
plt.show()

#잔차의 합계산해보기
sum(fitted_model1.resid)
#----------------------------------------------------------------#
#다른 두변수에도 동일하게 진행
rm1 = sm.add_constant(rm, has_constant='add')
lstat1 = sm.add_constant(lstat, has_constant='add')

model2 = sm.OLS(target,rm1)
fitted_model2=model2.fit()
model3 = sm.OLS(target,lstat1)
fitted_model3=model3.fit()

fitted_model2.summary()
fitted_model3.summary()

pred2=fitted_model2.predict(rm1)
pred3=fitted_model3.predict(lstat1)

plt.scatter(rm,target,label="data")
plt.plot(rm,pred2,label="result")
plt.legend()
plt.show()

plt.scatter(lstat,target,label="data")
plt.plot(lstat,pred3,label="result")
plt.legend()
plt.show()

fitted_model2.resid.plot()
plt.xlabel("residual_number")
plt.show()

fitted_model3.resid.plot()
plt.xlabel("residual_number")
plt.show()
#-----------------------------------------------------#

fitted_model1.resid.plot(label="crim")
fitted_model2.resid.plot(label="rm")
fitted_model3.resid.plot(label="lstat")
plt.legend()


###다중선형회귀분석 실습
#crim, rm, lstat 세개의 변수를 통해 다중회귀적합

x_data=boston[['CRIM','RM','LSTAT']] ##변수 여러개
print(x_data.head())

#상수항 추가
x_data1 = sm.add_constant(x_data, has_constant='add')
#회귀모델 적합
multi_model = sm.OLS(target,x_data1)
fitted_multi_model=multi_model.fit()

print(fitted_multi_model.summary())


##단순선형회귀모델의 회귀계수와 비교
print(fitted_model1.params)
print(fitted_model2.params)
print(fitted_model3.params) ##단순선형회귀분석의 회귀 계수(R2)와 비교

#다중선형회귀모델의 회귀 계수
print(fitted_multi_model.params)

#행렬연산을 통해 beta구하기
from numpy import linalg ##행렬연산을 통해 beta구하기
ba=linalg.inv((np.dot(x_data1.T,x_data1))) ## (X'X)-1
np.dot(np.dot(ba,x_data1.T),target) ##(X'X)-1X'y

pred4=fitted_multi_model.predict(x_data1)

#residual plot
fitted_multi_model.resid.plot()
plt.xlabel("residual_number")
plt.show()

#앞에 그렸던 잔차모델과 같이 그려보기

fitted_model1.resid.plot(label="crim")
fitted_model2.resid.plot(label="rm")
fitted_model3.resid.plot(label="lstat")
fitted_multi_model.resid.plot(label="full")
plt.legend()
plt.show()

#---------------------------------------------------#
#crim, rm, lstat, b, tax, age, zn, nox, indus 변수를 통한 다중선형회귀분석

x_data2=boston[['CRIM','RM','LSTAT','B','TAX','AGE','ZN','NOX','INDUS']]  ##변수 추가
print(x_data2.head())

#상수항 추가
x_data2_ = sm.add_constant(x_data2, has_constant='add')
#회귀모델 적합
multi_model2 = sm.OLS(target,x_data2_)
fitted_multi_model2 = multi_model2.fit()

print(fitted_multi_model2.summary())
#변수 3개만 추가한 모델의 회귀 계수
print(fitted_multi_model.params)
#Full변수
print(fitted_multi_model2.params)

#dase모델과 Full모델의 잔차비교
fitted_multi_model.resid.plot(label="full")
fitted_multi_model2.resid.plot(label="full_add")
plt.legend()
plt.show()

#---------------------------------------------------------------#
##상관계수/산점도를 통해 다중공선성 확인
#상관행렬
x_data2.corr()
#상관행렬 시각화 해서 보기

cmap = sns.light_palette("darkgray", as_cmap=True)
sns.heatmap(x_data2.corr(), annot=True, cmap=cmap)
plt.show()

#변수별 산점도 시각화
sns.pairplot(x_data2)
#plt.show()

##VIF를 통한 다중공선성 확인

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(
    x_data2.values, i) for i in range(x_data2.shape[1])]
vif["features"] = x_data2.columns
print(vif)
print('-------------------------------------')
#nox 변수 제거후(X_data3) VIF 확인
vif = pd.DataFrame()
x_data3= x_data2.drop('NOX',axis=1)
vif["VIF Factor"] = [variance_inflation_factor(
    x_data3.values, i) for i in range(x_data3.shape[1])]
vif["features"] = x_data3.columns
print(vif)

print('-------------------------------------')
#rm 변수 제거후(X_data4) VIF 확인
vif = pd.DataFrame()
x_data4= x_data3.drop('RM',axis=1)
vif["VIF Factor"] = [variance_inflation_factor(
    x_data4.values, i) for i in range(x_data4.shape[1])]
vif["features"] = x_data4.columns
print(vif)