import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt


###########!!!!!!!ubuntu version!!!!!!##############

# 클래스 정의
# img_00 = ".\dataset\class00\TRAIN" # 클래스 0번 (큰곰자리)
# img_01 = ".\dataset\class01" # 클래스 1번 (남십자성)

img_00 = "./dataset/class00/test_stack" # 클래스 0번 (큰곰자리)
img_01 = "./dataset/class01/test_stack" # 클래스 1번 (남십자성)
img_02 = "./dataset/class02/test_stack" # 클래스 2번 (없음)


ver = 240 # (이미지 리사이즈 후) 세로 픽셀수!!!! 240
hor = 320 # (이미지 리사이즈 후) 가로 픽셀수!!!! 320
X_all = [] # 속성 데이터가 들어갈 변수 생성
Y_all = [] # 정답 클래스가 들어갈 변수 생성

#0000000000000000000000000000000000
# 클래스 0번 속성 생성
files_00 = sorted(glob.glob(img_00 + "/*.npy"))
num_00 = len(files_00)

X = [] # 비어있는 배열 생성

for i, filepath in enumerate(files_00):
    
    temp= np.load(filepath) 
    X.append(temp) # 이미지를 하나씩 추가하여 한 클래스의 이미지 셋(집합) 생성
    
X_00 = np.array(X) # 한 클래스의 이미지 셋을 배열 형태로 변환

# 클래스 0번 정답 클래스 생성
Y_00 = 0 * np.ones(num_00)
#0000000000000000000000000000000000

#1111111111111111111111111111111111
# 클래스 1번 속성 생성
files_01 = sorted(glob.glob(img_01 + "/*.npy"))
num_01 = len(files_01)
X = [] # 비어있는 배열 생성

for i, filepath in enumerate(files_01):
    temp= np.load(filepath) 
    X.append(temp) # 이미지를 하나씩 추가하여 한 클래스의 이미지 셋(집합) 생성
    
X_01 = np.array(X) # 한 클래스의 이미지 셋을 배열 형태로 변환

# 클래스 1번 정답 클래스 생성
Y_01 = 1 * np.ones(num_01)
#1111111111111111111111111111111111

#2222222222222222222222222222222222
# 클래스 2번 속성 생성
files_02 = sorted(glob.glob(img_02 + "/*.npy"))
num_02 = len(files_02)
X = [] # 비어있는 배열 생성

for i, filepath in enumerate(files_02):
    temp= np.load(filepath) 
    X.append(temp) # 이미지를 하나씩 추가하여 한 클래스의 이미지 셋(집합) 생성
    
X_02 = np.array(X) # 한 클래스의 이미지 셋을 배열 형태로 변환

# 클래스 2번 정답 클래스 생성
Y_02 = 2 * np.ones(num_02)
#2222222222222222222222222222222222


# 속성 및 정답 클래스 합치기
X_all = np.concatenate((X_00, X_01, X_02), axis=0) # 모든 클래스 속성 합치기
Y_all = np.concatenate((Y_00, Y_01, Y_02), axis=0) # 모든 클래스 정답 합치기

# 학습셋과 테스트셋의 구분
from sklearn.model_selection import train_test_split

# 학습셋과 테스트셋의 구분
x_train, x_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.3, random_state=0)

xy = (x_train, x_test), (y_train, y_test)
np.save("./dataset_constellation.npy", xy)









