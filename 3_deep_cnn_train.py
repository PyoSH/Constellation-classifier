'''
Based by prof.이승호 leesh903@koreatech.ac.kr
Modified by 표승현
'''

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
#============================================
# 딥러닝 모델 생성 관련 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,Dropout

# 정답 데이터 변환(전처리) 관련
from tensorflow.keras import utils

# 데이터 셋 불러오기에 사용
import numpy as np
#============================================

# 데이터셋 불러오기
(x_train, x_test), (y_train, y_test) = np.load('dataset_constellation.npy', allow_pickle=True)


# 학습용 속성 데이터
x_train_encoded = x_train.astype('float32') / 255 # 이미 형태가 갖춰졌으므로 reshape 과정은 필요 없음


# 테스트용 속성 데이터
x_test_encoded = x_test.astype('float32') / 255 # 이미 형태가 갖춰졌으므로 reshape 과정은 필요 없음

# 정답 클래스에 원 핫 인코딩 적용(해당 클래스 자리만 1, 나머진 0)
y_train_encoded = utils.to_categorical(y_train, num_classes=3)
y_test_encoded = utils.to_categorical(y_test, num_classes=3)

## 딥러닝 모델의 구조 설정
# 컨볼루션 층(convolution layer) - 풀링 포함
# model = Sequential()
# model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(240, 320, 7)))
# model.add(MaxPooling2D((6, 6)))
# model.add(Dropout(0.5))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((6, 6)))
# model.add(Dropout(0.5))
# model.add(Conv2D(64, (3, 3), activation='relu'))

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(240, 320, 7)))
model.add(MaxPooling2D((6, 6)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 일반적인 신경망 층(fully connected layer)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax')) # 클래스의 개수가 바뀌니까 출력층의 개수도 바뀌어야!!!!


model.summary() # 딥러닝 구조 확인

# 딥러닝 학습 설정
# (loss : 오차함수 종류, optimizer : 최적화 알고리즘, metrics : 평가 기준)
model.compile(optimizer='adam', # or 'adam'
              loss='mean_squared_error', # or 'categorical_crossentropy'
              metrics=['acc'])

## 딥러닝 학습 실행
# ▶▶▶학습과정 에포크 별 시각화를 위해 변수 history에 에포크 별 결과를 저장
history = model.fit(x_train_encoded, y_train_encoded, # 가중치 갱신에 학습 셋 사용
                    epochs=10, batch_size=100,
                    verbose=1, # 에포크 별 성능을 표시(1로 설정) 또는 미표시(0로 설정)
                    validation_data=(x_test_encoded, y_test_encoded)) # 에포크 별 테스트 셋으로 성능 검증

# 딥러닝 모델 학습 후 성능 측정(정확도)
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test_encoded, y_test_encoded)[1]))

# 딥러닝 모델 저장
model.save('my_cnn_model_2018120133.h5')   # 모델을 컴퓨터에 저장

import matplotlib.pyplot as plt # 학습과정 시각화를 위해 라이브러리 불러오기
training_accuracy = history.history['acc'] # 학습 셋에 대한 성능 데이터(에포크 별)
test_accuracy = history.history['val_acc'] # 테스트 셋에 대한 성능 데이터(에포크 별)

epoch_count = range(1, len(training_accuracy)+1) # 에포크를 1부터 설정값까지 카운팅하는 변수

plt.plot(epoch_count, training_accuracy, 'r--') # 학습 셋에 대한 정확도는 빨간색 점선('r--') 표시
plt.plot(epoch_count, test_accuracy, 'b-') # 테스트 셋에 대한 정확도는 파란색 실선('b-') 표시

plt.legend(["Training Accuracy", "Test Accuracy"])
plt.xlabel("Epoch") # x축(가로축) 이름
plt.ylabel("Accuracy") # y축(세로축) 이름
plt.show()

