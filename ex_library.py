#라이브러리 호출

import numpy as np
import pandas as pd

list1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
    ]
#----------------------------------------------#
arr1 = np.array(list1) #파이썬 list를 numpy array로 변환

print("arr2의 ndim: ", arr1.ndim) # arr2의 차원
print("arr2의 shape: ", arr1.shape)# arr2의 행, 열의 크기
print("arr2의 size : ", arr1.size ) # arr2의 행 x 열
print("arr2의 dtype: ", arr1.dtype) # arr2의 원소의 타입. # int64 : integer + 64bits
print("arr2의 itemsize: ", arr1.itemsize)# arr2의 원소의 사이즈(bytes) # 64bits = 8B
print("arr2의 nbytes: ", arr1.nbytes)# itemsize * size # numpy array가 차지하는 메모리 공간
#----------------------------------------------#
a = np.zeros(1) #원소가 0인 array를 생성
b = np.ones(1) #원소가 1인 array를 생성
c = np.arange(1) #특정 범위의 원소를 가짐
#----------------------------------------------#
arr1 = np.arange(10)
arr1[:3] # 앞에서부터 원소 3개 slicing


data = pd.read_csv("..data/iris.csv") # 파일 불러오기
print(data)
# 숫자로 하드 코딩
data.loc[data["Species"] == "Iris-setosa","Species"] = 0
data.loc[data["Species"] == "Iris-versicolor","Species"] = 1
data.loc[data["Species"] == "Iris-virginica","Species"] = 2
data

data2 = pd.read_csv("..data/kaggle_survey_2020_responses.csv")
print(data2)
# 박사 학위 소지자이면서, 대한민국 국적을 가진 사람
data2[(data2["Q4"] == "Doctoral degree") & (data2["Q3"]== "South Korea")]