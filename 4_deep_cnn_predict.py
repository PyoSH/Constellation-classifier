#============================================
# 데이터 셋 준비
import numpy as np
from tensorflow.keras.preprocessing import image
(x_train, x_test), (y_train, y_test) = np.load('dataset_constellation.npy', allow_pickle=True) 

# 테스트용 속성 데이터
x_test_encoded = x_test.astype('float32') / 255 # 이미 형태가 갖춰졌으므로 reshape 과정은 필요 없음
#============================================


#===============================
# 저장했던 딥러닝 모델 불러오기
from tensorflow.keras.models import load_model  # 모델 불러오기 라이브러리
model = load_model('my_cnn_model_2018120133.h5') # 모델을 새로 불러옴
#===============================

#===============================
# 속성(이미지)을 불러온 딥러닝 모델에 입력하여 예측값 획득
import numpy as np



index = 11 # 샘플 이미지 선택
x_input = x_test_encoded[index,:] # 테스트 셋에서 샘플 이미지 하나를 선택
x_input = x_input.reshape(1, 240, 320, 7) 
# 1 : 1장의 이미지를 입력
# 240, 320 : 240x320 크기의 이미지를 의미
# 7  : 채널 수
prediction = model.predict(x_input) # 예측 수행
#softmax_sum = np.sum(prediction) # 소프트맥스 출력의 총 합은 1이어야 함
#===============================

# 클래스 이름
constellation_class = {
      0: 'Big Dipper'
    , 1: 'Crux'
    , 2: 'Nothing'
}

#===============================
# 예측값이 맞는지 확인
class_predicted = np.argmax(prediction) # 클래스(예측)
class_actual = y_test[index]            # 클래스(정답)

class_predicted_name = constellation_class[class_predicted] # 클래스 이름(예측)
class_actual_name = constellation_class[class_actual] # 클래스 이름(정답)

model.summary() # 딥러닝 구조 확인

print("클래스(예측) : %s" %class_predicted_name) # 클래스 이름(예측) 출력
print("클래스(정답) : %s" %class_actual_name) # 클래스 이름(정답) 출력

import matplotlib.pyplot as plt
# print(x_test[index,:,:,:])

test_img = x_test[index,:,:,0].astype(np.uint8) # predict -> use for classification
plt.imshow(test_img) # 샘플 이미지 표시
plt.show()

#=============================== 




