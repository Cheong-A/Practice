import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

train = pd.read_csv("C:/Users/blue_/Downloads/fast/데이터 분석 라이브러리/data/train.csv", encoding ='utf-8' )
test = pd.read_csv("C:/Users/blue_/Downloads/fast/데이터 분석 라이브러리/data/test.csv", encoding ='utf-8')

print(train.PassengerId.nunique())

#필요 없는 칼럼 drop
train1 = train.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"]) #PassengerId, Name, Cabin, Ticket
print(train1)

grouped = pd.pivot_table(train1, index='Pclass')
grouped.plot(kind="barh")
plt.show()

#성별/등급에 따른 피벗 테이블
grouped = pd.pivot_table(train1, index=['Sex', "Pclass"])
print(grouped)

# 성별/등급을 기준으로 만든 pivot table에서 age=평균값, survived=sum값
grouped = pd.pivot_table(train1, index=['Sex', "Pclass"],
                        aggfunc = {"Age" : np.mean, "Survived" : np.sum})
print(grouped)
grouped.plot(kind="barh")
plt.show()

#결측치 확인
print(train1[train1.isnull().any(axis=1)])

# Age column의 결측치들을 Age column의 평균값으로 채움.
train1.Age.fillna(train.Age.mean(), inplace=True)
print(train1)

#Embarked
train1.loc[train.isnull().any(axis=1), "Embarked"] = "S"

#결측치 처리 확인
print(train1[train1.isnull().any(axis=1)])


# 범주형 column, one-hot encoding
# "Sex" / "Embarked" feature
train_OHE = pd.get_dummies(data=train1, columns = ["Sex", "Embarked"])
print(train_OHE)