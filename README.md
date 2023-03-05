# Constellation-classifier

한국기술교육대학교 2022년 1학기 [딥러닝 응용(prof.이승호)] 강좌의 텀 프로젝트입니다. <br>
클래스당 100장으로 구성된 데이터셋은 https://drive.google.com/file/d/1pMNI5WedppeDYJg70S4Fmd2ctCG-6nOP/view?usp=sharing 으로 받을 수 있습니다 :)  <br>

## 천문항법을 위한 별자리 검출 및 분류의 아이디어
 별자리는 지구의 자전과 공전에 따라 다양하게 관측될 수 있습니다. 이 때문에 인류는 고대부터 별자리를 관찰하는 것으로 위치를 알 수 있었습니다. <br>
다양한 별자리를 구분하기 위해서는 다양한 수학적 방법이 필요합니다. 또한, 관측 데이터인 영상은 촬영하는 장비와 환경에 따라 같은 관측대상임에도 불구하고 다양한 관측 영상이 나오기 때문에 각기 다른 영상처리를 해야 하는 문제가 생깁니다. 이 문제(패턴 인식, 다양하게 관측되는 대상)을 해결하기 위해 합성곱 신경망을 사용해보았습니다. <br>
<br>
 별자리를 우선적으로 인식해 위치정보를 얻어내는 방식은 별 하나 하나를 통해 항법하는 것 보다는 정확도가 떨어지고, 단일 항성을 분류하는 것보다 정보처리량이 많아지기 때문에 인공위성 등 우주환경의 플랫폼 보다는 행성 표면 혹은 대기권에서 활동하는 플랫폼에 우선적으로 적용할 수 있을 것이라고 생각합니다. <br>
성층권에서 작동하는 무인 비행선 혹은 드론의 경우에는 태양광의 영향을 그나마 덜 받겠지만, 행성 표면에서 작동하는 무인 선박 또는 탐사 로버는 낮 시간에 별자리를 관찰할 수 없어 행성 표면의 플랫폼은 태양 등의 주된 천체물을 사용해 위치정보를 얻어야 할 것입니다. 
(시간대, 지구상의 위치, 방향 도출)

### 목표
입력 : 하늘 사진 <br>
출력 : 특정 별자리 분류(북두칠성, 남십자성, 별자리 없음)

### 개발 환경

```
플랫폼 = Linux-5.4.0-110-generic-x86_64-with-glibc2.17, ubuntu 18.04
가상환경 = Anaconda3
Python = 3.8.13
Tensorflow - 2.3.0
Numpy – 1.18.5
Matplotlib – 3.5.2
spyder – 4.1.5
pandas – 1.4.2
seaborn – 0.11.2
sckit-learn – 1.1.1
h5py – 2.10.0
scipy – 1.4.1
```

# 0. 관측장소, 시간 설정
2022-05-22 00시 ~ 2022-05-23-00시

북반구 (원본 154장) <br>
검출→동향 <br>
→ 한국, 천안(북반구 중위권) <br>
→ Akureyri, 아이슬란드(북반구 최북단)<br>

남반구 (원본 154장)<br>
검출→남향<br>
→ Antofagasta, 칠레(남반구 중위권)<br>
→ Punta Arenas, 칠레(남반구 최남단)<br>

# 1. 별자리 사진 얻기
관측 지역과 시간대를 통제하기 위해 시뮬레이션(Stellarium)을 사용했습니다.

1) 별자리 선을 표시
2) 특정 별자리를 추적하는 화면을 동영상으로 저장
2) 저장한 동영상을 py 코드를 사용해 영상들로 나누어 line_1에 저장

# 2. 전처리
**시뮬레이션 영상** → **전처리**(이진화, 코너/직선검출) → **Detection** → **분류** <br><br>

1. **선 유무** <br>
학습의 난이도를 낮추기 위해 stellarium 프로그램 상에서 별자리 선을 표시(개선사항)
2. **이진화 종류** <br>
별을 제외한 다른 노이즈를 제거하기 위함
→ threshold ⇒ (binary_0)
→ adaptive threshold(mean)+morph_close ⇒ (binary_1)
→ adaptive threshold(mean, gausian, otsu)+morph_close ⇒ (binary_1)
3. **코너검출** <br>
 특징점 추출을 미리 해 줌으로써 합성곱 신경망의 성능이 향상되는지 알기 위함
**→ FAST 사용**
4. **직선검출** <br>
단순 점만으로는 별자리 검출 시 다른 별자리의 일부분을 보고도 특정 별자리라고 인식할 수 있으므로 별자리의 패턴을 강조하기 위함
**→Hough Transform** 

![Screenshot from 2023-03-05 17-29-25](https://user-images.githubusercontent.com/42665051/222950139-81a31cda-3a93-44a5-afb1-7a6b1a558ec7.png)


### 3. 이미지 다층화처리 및 데이터셋 구성
0번 클래스 → 북두칠성
1번 클래스 → 남십자성
2번 클래스 → 없음

rgb가 3레이어이듯, 위의 전처리 결과물들을 각기 저장해서 여러 영상들로 만들면 레이어가 여럿인 하나의 대상으로 묶을 수 있고, 이 프로젝트에서는(height,width,layer)입니다.

Layer: line_1, binary_0, binary_1, binary_2, FAST_0, FAST_1, Hough 
![Screenshot from 2023-03-05 17-28-28](https://user-images.githubusercontent.com/42665051/222950108-483c4be8-0244-4dc2-bd18-29533be35a5b.png)


### 4. CNN모델 예측 및 성능평가
epoch = 10 <br>
batch_size = 100 <br>

```python
ver_final.Layer (type)       Output Shape              Param #   
=================================================================
conv2d_2 (Conv2D)            (None, 236, 316, 32)      5632      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 39, 52, 32)        0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 39, 52, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 37, 50, 64)        18496     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 18, 25, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 28800)             0         
_________________________________________________________________
dense_3 (Dense)              (None, 512)               14746112  
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 200)               102600    
_________________________________________________________________
dropout_5 (Dropout)          (None, 200)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 3)                 603       
=================================================================
```
![Screenshot from 2023-03-05 17-26-40](https://user-images.githubusercontent.com/42665051/222950160-c951781b-debb-4a5d-aaec-c11dc08c1050.png)

# 추후 연구방향 


### 참조
1. Lindblad, Thomas and Clark S. Lindsey. “Star Identification using Neural Networks.” (2007). 
2. Rijlaarsdam D, Yous H, Byrne J, Oddenino D, Furano G, Moloney D. Efficient Star Identification Using a Neural Network. *Sensors (Basel)* . 2020;20(13):3684. Published 2020 Jun 30. doi:10.3390/s20133684
3. Rijlaarsdam, David et al. “A Survey of Lost-in-Space Star Identification Algorithms since 2009.” *Sensors (Basel, Switzerland)* vol. 20,9 2579. 1 May. 2020, doi:10.3390/s20092579 
4. Zhan, Yinhu & Chen, Shaojie & Zhang, Xu. (2021). Adaptive celestial positioning for the stationary Mars rover based on a self-calibration model for the star sensor. Journal of Navigation. 1-16. 10.1017/S0373463321000680.
5. Dachev, Yuri & Panov, Avgust. (2017). 21 st century Celestial navigation systems.
6. Liheng Ma, Dongshan Zhu, Chunsheng Sun, Dongkai Dai, Xingshu Wang, and ShiQiao Qin, "Three-axis attitude accuracy of better than 5 arcseconds obtained for the star sensor in a long-term on-ship dynamic experiment," Appl. Opt. 57, 9589-9595 (2018)
7. **Liu, Xiaoge et al. “Constellation Detection.” (2015).**
8. Constellation Queries over Big Data
