from IPython.display import display, HTML

import warnings
warnings.filterwarnings('ignore')

# 데이터 전처리
import numpy as np
import pandas as pd

# 기계학습 모델 및 평가
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# 시각화
import seaborn as sns
import matplotlib.pyplot as plt


#실습 1 : Breast Cancer
#반응변수 : 양성여부 (malignant : 악성-0, Benign : 양성-1)

breast_cancer = load_breast_cancer()
print(breast_cancer.DESCR)
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = pd.DataFrame(breast_cancer.target, columns=['target'])

data = pd.concat((X,y),axis=1)
display(data)

#데이터 shape 확인
print(f'관측지수 : {data.shape[0]} \n변수수 : {data.shape[1]}')

#구분선
print('-'*50)
#print(data.info())  # 30번에 존재하는 타겟값 예측

#타겟의 타입 변경
data.target = data.target.astype('category')
print(data.info())

#간단한 요약 통계량 확인
print('-'*25+'통계량'+'-'*25)
print(data.describe())
print(data.target.value_counts())

print('-'*50)

#그래프로 시각화 하여 인사이트 얻기
plt.figure(figsize= (8,8))
sns.pairplot(data[['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                   'mean smoothness', 'mean compactness', 'mean concavity',
                   'mean concave points', 'mean symmetry', 'mean fractal dimension','target']],
             hue='target', palette = 'husl')
plt.show()

#상관관계확인
plt.figure(figsize=(10,10))
heat_map = sns.heatmap(data.corr(),
                       cmap='bwr',#https://matplotlib.org/examples/color/colormaps_reference.html
                       linewidths=.1)
heat_map.set_xticklabels(heat_map.get_xticklabels(),
                         rotation=90)
heat_map.xaxis.set_ticks_position('top')
plt.show()


#타겟 변수의 클래스 비율을 유지하며 train/test 데이터 분리
#설명변수(X), 반응변수(y) 나누기

X = data.drop('target', axis=1)
display(X.head(3))
y = data['target']

#Training 데이터 70% / Testing 데이터 30% 나누기
#클래스 비율 유지 -> train_test_split 함수 내 stratify 옵션
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123, stratify=y)

print('-'*50)
#정규화
preprocessor = preprocessing.Normalizer()
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#sklearn.linear_model의 logistic 회귀모델 이용
model = LogisticRegression(penalty='none',
                           fit_intercept=True,
                           solver='newton-cg',
                           verbose=1,
                           random_state=20210827)
model.fit(X_train,y_train)
pred_train=model.predict(X_train)
pred_test=model.predict(X_test)
print(pred_train)
print(pred_test)


#모델 결과 확인
print(f'학습 정오행렬 \n{confusion_matrix(y_train, pred_train)}')
print(f'테스트 정오행렬 \n{confusion_matrix(y_test, pred_test)}\n')
print(f'학습 정확도 : {accuracy_score(y_train, pred_train):.4f}')
print(f'테스트 정확도 : {accuracy_score(y_test, pred_test):.4f}')

print('-'*50)

def perf_eval(cm):
    # True positive rate: TPR
    TPR = cm[1, 1] / sum(cm[1]) # recall
    # True negative rate: TNR
    TNR = cm[0, 0] / sum(cm[0])
    # Simple Accuracy
    ACC = (cm[0, 0] + cm[1, 1]) / sum(cm.reshape(-1,))
    # F1-measure
    Precision = cm[1,1] /sum(cm[:,1])
    F1 = 2*TPR*Precision/(TPR+Precision)
    return ([TPR, TNR, ACC, F1])

cm_test = confusion_matrix(y_test, pred_test)
print('TPR:',perf_eval(cm_test)[0])
print('TNR:',perf_eval(cm_test)[1])
print('ACC:',perf_eval(cm_test)[2])
print('F1:',perf_eval(cm_test)[3])

#모델 결과물 확인
#사용된 hyperparameter (사용자 지정 변수)
#계수 : 1단위 증가할 때 로그아드의 변화량
#odd ratio 확인 : 1단위 증가할 때 변화하는 성공확률의 비율
model.get_params

pd.DataFrame(np.concatenate((model.coef_.T, np.exp(model.coef_).T),axis=1),
             index=breast_cancer.feature_names,
             columns=['coefficient','odd ratio'])
model.intercept_