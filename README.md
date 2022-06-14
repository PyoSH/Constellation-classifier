# Constellation-classifier

한국기술교육대학교 2022년 1학기 딥러닝 응용 강좌의 텀 프로젝트로 시작했다. 
한국기술교육대학교 융합학과 이승호 교수님(leesh903@koreatech.ac.kr)의 템플릿 코드(2,3,4번 코드)를 기반으로 진행되었다. 

Notion Page : https://difficult-algebra-4c9.notion.site/ef12e08709d1455e871795ba6d40f77c


# 딥러닝 기반 별자리 분류를 사용한 별자리 항법

별자리 탐지 및 구분을 통해 우주공간에 있는 물체의 위치 예측하기

⇒ **무인 선박 혹은 성층권 무인항공기를 위한** 별자리 항법 프로그램 제안  

- 서론 & 해낼 것
    
    망망대해에서, GPS를 사용하지 않고 위치를 파악할 수 있는 방법은 별자리의 위치를 통해 위도, 경도를 계산하는 거이다. 사람이 있는 선박은 GPS가 없어도 운행이 가능하지만, 무인 선박은 INS와 GPS가 필수적이다. 
    앞으로 기후위기 연구, 운송업 등 자율항해 선박이 늘어나게 된다면 다양한 방법으로 위치정보를 알 수 있어야 할 것이다. 
    
    인공위성과 달리 병진, 회전가속이 연속적으로 변하는 선박은 단일 영상이 아닌 연속된 영상으로 위치예상값이 나와야 정확한 정보를 얻을 수 있다. (또한 화각이 넓어야 한다)
    
    → 별자리를 이용한 항법은 기존 스타 트래커와는 다르게, 아주 넓은 화각과 연속된 영상을 기반으로 위치를 추정해야 한다. 
    
    기존 및 인공신경망을 이용한 스타 트래커
    
    ### 1. 스타 트래커란 무엇인가?
    
    ![Untitled](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%92%20896361ef8efb4cbeb5e60765653b174e/Untitled.png)
    
    Star Tracker는 위의 사진과 같은 카메라 장치를 통해 별을 인식하고, 이것을 데이터베이스에 등록된 별과 비교해 우주공간에서 위치와 자세를 알아낼 수 있는 장치이다. 
    
    우주공간에서 작동해야 하고 높은 정확도가 요구되기 때문에 고가이다. 
    
    작동원리는 다음과 같다. 
    
    중심으로 하는 별을 잡고, 이 별이 어떤 별인지 데이터베이스와 비교해 알아낸다. 
    이 별의 주변에 있는 별이 어느 별인지 알아내고, 이를 토대로 구면 좌표계에서 우주선이 어느 곳을 향하고 있는지, 어느 곳에 있는지 도출한다. 
    
    우주선은 (~LEO 등등 궤도 따라서 다른 초속 근거로 나타내기~) 매우 빠르기 때문에 별이 슉슉 지나간다. 카메라에 잡히는 별이 진짜 별인지 노이즈인지 확인하기 위해 모션 블러를 방지하는 화면 전처리를 실행해 주고, 움직임이 예측과 일치하는지 확인해주는 수학적 과정(아래의 논문들에서 정확한 용어 찾아서 써주기)
    
    별을 찾고 대조하는 알고리즘을 대체하기 위해, 딥러닝 등의 방법이 제안되었지만**[1],[2]**
    이는 기존의 별을 찾는 과정을 대체할 뿐이다.  별 하나하나를 알아내는 것은 기존의 방법에 바로 사용할 수 있다는 장점이 있지만, ~~~~한 단점이 있다. 
    
    때문에 정확성은 비교적 낮을지라도 더 범용적이며 직관적으로 사용할 수 있는 방법을 제시하고 구현하고자 한다. ***(뭔가 서순이 바뀐 느낌인데???)***
    
    ## 별자리 검출 및 분류를 사용한 별자리 항법
    
    밤 하늘에 별자리는 무수히 많다. 이 별자리는 지구의 자전과 공전에 따라 다양하게 관측될 수 있다. 
    이 성질을 이용해, 인류는 고대부터 별자리를 관찰하는 것으로 배를 타며 이동할 수 있었다. 
    
     카메라를 이용해서 별을 검출하고 어느 것인지 구분하는 것은, 기존의 컴퓨터 비전으로도 충분하다.(근거는?) 단순한 예시로, 관측 사진에서 잡티 등 노이즈는 제거해 별만 보이게 하고, 그 별의 색깔과 밝기, 근처 별의 정보를 통해 어느 별인지 알아내는 과정이 있는데 여기에는 큰 연산요소가 있지 않다. 그러나, 화각을 높여서 이 별자리가 어느 별자리인지 알아낸다면 어떨까?(기존에 시도한 방법들에 대한 논문이 필요하다!!! 지금부턴 뇌피셜이니까 근거가 필요해)
    
    별자리는 패턴이라고 볼 수 있다. 그러나 다양한 패턴이 한 데에 얽힌다면, 이것들을 구분하기 위해서는 다양한 수학적 방법이 필요하다. 또한, 관측 데이터인 영상은 촬영하는 장비와 환경에 따라 같은 관측대상임에도 불구하고 다양한 관측 영상이 나오기 때문에 각기 다른 영상처리를 해야 하는 문제가 생긴다. 
    
    이 문제(패턴 인식, 다양하게 관측되는 대상)을 해결하기 위해 합성곱 신경망을 도입한다. 
    합성곱 신경망을 비롯한 딥 러닝은 다양하게 나타나는 관측값에서 특징을 통해 검출 및 분류를 실행한다는 점에서 본 문제를 해결하는 것에 알맞다. 
    
    그러나 별자리를 우선적으로 인식해 위치정보를 얻어내는 방식은 별 하나 하나를 통해 항법하는 것 보다는 정확도가 떨어지고, 단일 항성을 분류하는 것보다 정보처리량이 많아지기 때문에 인공위성 등 우주환경의 플랫폼 보다는 행성 표면 혹은 대기권에서 활동하는 플랫폼에 우선적으로 적용할 수 있을 것이다. 
     성층권에서 작동하는 무인 비행선 혹은 드론의 경우에는 태양광의 영향을 그나마 덜 받겠지만, 행성 표면에서 작동하는 무인 선박 또는 탐사 로버는 낮 시간에 별자리를 관찰할 수 없다. 때문에, 행성 표면의 플랫폼은 태양 등의 주된 천체물을 사용해 위치정보를 얻어야 한다. 
    
    2022-04-10 오늘은 여기까지!!! 
    
    ### 지구 표면의 응용
    
    - 시간대
    - 지구상의 위치
    - 방향 도출
    
    천문항법을 기반으로 진행
    
    ### 우주공간의 응용
    
    - 지구 혹은 행성 위에서 위치
    - 
    

---

나침판 보정 이상으로 해낼 수 있을까?

해내고 싶은 것은?

### 이번 프로젝트의 목표

1. 하늘 사진을 주었을 때 특정 별자리 검출 및 분류
2. 카메라가 비추는 방향을 안다고 가정할 때 별자리가 회전한 각도와 위치를 통해 북반구 혹은 남반구를 분별한다. 

우수 프로젝트 됐다~~~~!!! 깃에도 올릴 수 있으면 올리자. 이승호 교수님한테 여쭤봤으니까 기다리자~~~(2022.06.11) 

[2018120133_표승현.pdf](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%92%20896361ef8efb4cbeb5e60765653b174e/2018120133_%ED%91%9C%EC%8A%B9%ED%98%84.pdf)

### 개발 환경 만들기

이름 : **Deep_APP**

```jsx

**플랫폼** = Linux-5.4.0-110-generic-x86_64-with-glibc2.17, ubuntu 18.04
**가상환경** = Anaconda3
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

### 0. 관측장소, 시간 설정

<aside>
💡 2022-05-22 00시 ~ 2022-05-23-00시

북반구 (원본 154장)
검출→동향
→ 한국, 천안(북반구 중위권)
→ Akureyri, 아이슬란드(북반구 최북단)

남반구 (원본 154장)
검출→남향
→ Antofagasta, 칠레(남반구 중위권)
→ Punta Arenas, 칠레(남반구 최남단)

</aside>

### 1. 별자리 사진 얻기

<aside>
💡 Stellarium 밤하늘 시뮬레이터를 이용

1) 별자리 선을 표시한다
2) 특정 별자리를 추적하는 화면을 동영상으로 저장한다
2) 저장한 동영상을 py 코드를 사용해 영상들로 나누어 line_1에 저장한다.

- 캡쳐한 영상.
    
    ![sim_origin.png](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%92%20896361ef8efb4cbeb5e60765653b174e/sim_origin.png)
    
    ![0.jpeg](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%92%20896361ef8efb4cbeb5e60765653b174e/0.jpeg)
    
</aside>

### 2. 전처리

<aside>
💡 **시뮬레이션 영상** → **전처리**(이진화, 코너/직선검출) → **Detection** → **분류**

1. **선 유무**
stellarium 프로그램에서 선을 해줄 수 있다. 

2. **이진화 종류**
별을 제외한 다른 노이즈를 제거하기 위함
→ threshold ⇒ (*binary_0*)
→ adaptive threshold(mean)+morph_close ⇒ (*binary_1*)
→ adaptive threshold(mean, gausian, otsu)+morph_close ⇒ (*binary_1*)
3. **코너검출**
 특징점 추출을 미리 해 줌으로써 합성곱 신경망의 성능이 향상되는지 알기 위함
****→ FAST
4. **직선검출**
단순 점만으로는 별자리 검출 시 다른 별자리의 일부분을 보고도 특정 별자리라고 인식할 수 있으므로 별자리의 패턴을 강조하기 위함
**→** Hough Transform

[이미지 전처리(완)](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%92%20896361ef8efb4cbeb5e60765653b174e/%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%8E%E1%85%A5%E1%84%85%E1%85%B5(%E1%84%8B%E1%85%AA%E1%86%AB)%20d82c8ae02b1446d7822370fb1997d6a2.md)

</aside>

### 3. 이미지 다층화처리 및 데이터셋 구성

<aside>
💡 0번 클래스 → 북두칠성
1번 클래스 → 남십자성
2번 클래스 → 없음

rgb가 3레이어이듯, 위의 전처리 결과물들을 각기 저장해서 여러 영상들로 만들면 레이어가 여럿인 하나의 대상으로 묶을 수 있다고 한다. 그러면 (height?, width?, N layer)로 코드 상에서 처리할 수 있다고 한다!

→ line_1, binary_0, binary_1, binary_2, FAST_0, FAST_1, Hough를 하나로 묶는다? 됐다! 

[데이터셋 만들기(완)](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%92%20896361ef8efb4cbeb5e60765653b174e/%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%89%E1%85%A6%E1%86%BA%20%E1%84%86%E1%85%A1%E1%86%AB%E1%84%83%E1%85%B3%E1%86%AF%E1%84%80%E1%85%B5(%E1%84%8B%E1%85%AA%E1%86%AB)%2098e0a2a79ee547059436d6c05dcbd46c.md)

</aside>

### 4. CNN모델 예측 및 성능평가

<aside>
💡 일단은 그냥 템플릿대로 돌렸다. 
epoch = 10
batch_size = 100

</aside>

```python
ver_final.Layer (type)                 Output Shape              Param #   
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

### 4. 필터, 특징맵 확인

<aside>
💡 합성곱 층이 세 층.

[필터, 특징 맵 확인 & 전처리의 효과?](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%92%20896361ef8efb4cbeb5e60765653b174e/%E1%84%91%E1%85%B5%E1%86%AF%E1%84%90%E1%85%A5,%20%E1%84%90%E1%85%B3%E1%86%A8%E1%84%8C%E1%85%B5%E1%86%BC%20%E1%84%86%E1%85%A2%E1%86%B8%20%E1%84%92%E1%85%AA%E1%86%A8%E1%84%8B%E1%85%B5%E1%86%AB%20&%20%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%8E%E1%85%A5%E1%84%85%E1%85%B5%E1%84%8B%E1%85%B4%20%E1%84%92%E1%85%AD%E1%84%80%E1%85%AA%2029c2e41bb09a4e1e9d901d3fba288914.md)

</aside>

![Screenshot from 2022-05-31 00-54-49.png](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%87%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A1%E1%84%85%E1%85%B5%20%E1%84%92%20896361ef8efb4cbeb5e60765653b174e/Screenshot_from_2022-05-31_00-54-49.png)

[Searching dataset for Constellation detection](https://astronomy.stackexchange.com/questions/38483/searching-dataset-for-constellation-detection)

---

### 검출!!!! 내가 할 수 있을까…?

[NAVER D2](https://d2.naver.com/helloworld/8344782)

[R-CNN object detection with Keras, TensorFlow, and Deep Learning - PyImageSearch](https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/)

[Faster R-CNN step by step, Part II](https://dongjk.github.io/code/object+detection/keras/2018/06/10/Faster_R-CNN_step_by_step,_Part_II.html)

[Keras documentation: Object Detection with RetinaNet](https://keras.io/examples/vision/retinanet/)

[종이 시험지 자동 채점 프로그램 | Tensorflow Object Detection API | Faster RCNN | Ch3. 문제 분류 모델 학습하기](https://velog.io/@nayeon_p00/%EC%A2%85%EC%9D%B4-%EC%8B%9C%ED%97%98%EC%A7%80-%EC%9E%90%EB%8F%99-%EC%B1%84%EC%A0%90-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8-Tensorflow-Object-Detection-API-Faster-RCNN-Ch3.-%EB%AC%B8%EC%A0%9C-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0)

[Step-by-Step R-CNN Implementation From Scratch In Python](https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55)

[MaskRCNN Custom Training 드디어 종결!! 아.. 고생많았다.(Colab 사용)](https://hansonminlearning.tistory.com/41?category=935564)

- **후속연구!용!**
    
    [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
    
    [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
    
    [models/research/object_detection at master · tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection)
    

---

### 참조!!!!!!!

- 참조 논문들!!
    1. Lindblad, Thomas and Clark S. Lindsey. “Star Identification using Neural Networks.” (2007).
        
        [[PDF] Star Identification using Neural Networks | Semantic Scholar](https://www.semanticscholar.org/paper/Star-Identification-using-Neural-Networks-Lindblad-Lindsey/6698ddad6a975c4e22159fdf027a4c4582cd95dc)
        
    
    1. Rijlaarsdam D, Yous H, Byrne J, Oddenino D, Furano G, Moloney D. Efficient Star Identification Using a Neural Network. *Sensors (Basel)* . 2020;20(13):3684. Published 2020 Jun 30. doi:10.3390/s20133684
        
        [Efficient Star Identification Using a Neural Network](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7374481/)
        
    
    1. Rijlaarsdam, David et al. “A Survey of Lost-in-Space Star Identification Algorithms since 2009.” *Sensors (Basel, Switzerland)* vol. 20,9 2579. 1 May. 2020, doi:10.3390/s20092579
        
        [](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7248786/)
        
    
    1. Zhan, Yinhu & Chen, Shaojie & Zhang, Xu. (2021). Adaptive celestial positioning for the stationary Mars rover based on a self-calibration model for the star sensor. Journal of Navigation. 1-16. 10.1017/S0373463321000680.
        
        [](https://www.researchgate.net/publication/353956401_Adaptive_celestial_positioning_for_the_stationary_Mars_rover_based_on_a_self-calibration_model_for_the_star_sensor)
        
    
    1. Dachev, Yuri & Panov, Avgust. (2017). 21 st century Celestial navigation systems.
        
        [21st century Celestial navigation systems](https://www.academia.edu/35098589/21st_century_Celestial_navigation_systems)
        
    
    1. Liheng Ma, Dongshan Zhu, Chunsheng Sun, Dongkai Dai, Xingshu Wang, and ShiQiao Qin, "Three-axis attitude accuracy of better than 5 arcseconds obtained for the star sensor in a long-term on-ship dynamic experiment," Appl. Opt. 57, 9589-9595 (2018)
        
        [Three-axis attitude accuracy of better than 5 arcseconds obtained for the star sensor in a long-term on-ship dynamic experiment](https://opg.optica.org/ao/abstract.cfm?uri=ao-57-32-9589)
        
    
    1. **Liu, Xiaoge et al. “Constellation Detection.” (2015).**
        
        [[PDF] Constellation Detection | Semantic Scholar](https://www.semanticscholar.org/paper/Constellation-Detection-Liu-Ji/f1d7792bf6796b9286bcb732ea7cd94eae2ad90e)
        
    2. ****Constellation Queries over Big Data****
        
        [Constellation Queries over Big Data](https://doi.org/10.48550/arXiv.1703.02638)
        
    3.
