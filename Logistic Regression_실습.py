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
breast_cancer = pd.read_csv(../)
breast_cancer = load_breast_cancer()
print(breast_cancer.DESCR)
