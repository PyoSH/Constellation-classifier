#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:13:38 2022

@author: 표승현

검출기능 구현을 가정한 목표설명용 데모 코드
GET_ROI함수 상단에서 ROI를 바꾸고 main함수 상단의 im경로를 바꿔야 정상적으로 돌아간다.

"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # 모델 불러오기 라이브러리
      
def GET_ROI(im):
    #################1. 입력 이미지에 맞게 ROI 바꾸기!!!!###############
    # (x,y),(w,h) =(92,382),(194,190)     ## ROI -> DemoImg_crux.jpeg
    # (x,y),(w,h) =(192,582),(194,190)     ## ROI -> nothing
    # (x,y),(w,h) =(20,238),(658,412)  ## ROI -> DemoImg_BigDipper.jpeg
    (x,y),(w,h) =(530,200),(400,430)  ## ROI -> Demo_BigDipper_2.jpeg
    roi_img = im[y:y+h, x:x+w]
    
    cv2.rectangle(im, (x,y,w,h), 255, 1)
    return (x,y,w,h)

def LINEDETECT(img):
    img_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    minLineLength = 100
    maxLineGap = 0
    
    lines = cv2.HoughLinesP(img,1,np.pi/360,133,minLineLength,maxLineGap)
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(img_color,(x1,y1),(x2,y2),(0,0,255),15)
    
    return img_color
    
def binarize_grpAll(gray_img):
    width = 320             # 320
    height = 240 
    img_origin = gray_img.copy()
    
    ## 2. 임계값을 기준으로 이진화하기
    ret, th1 = cv2.threshold(gray_img,150,255, cv2.THRESH_BINARY)
    th3 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,15)
    t, t_otsu = cv2.threshold(gray_img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print('otsu threshold:', t)
    
    kernel = np.ones((13, 13), np.uint8)
    th3_close = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    
    
    ## 2.1 FAST전처리
    fast = cv2.FastFeatureDetector_create()
            
    kp_2 = fast.detect(th3_close,None)
    FAST_2 =  cv2.cvtColor(cv2.drawKeypoints(th3_close, kp_2, None),cv2.COLOR_BGR2GRAY)
    
    kp_3 = fast.detect(t_otsu,None)
    FAST_3 =cv2.cvtColor(cv2.drawKeypoints(t_otsu, kp_3, None),cv2.COLOR_BGR2GRAY)
    Hough_0 = cv2.cvtColor(LINEDETECT(th3_close),cv2.COLOR_BGR2GRAY)
    
    dst_0 = cv2.resize(img_origin, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    dst_1 = cv2.resize(th1, dsize=(width, height), interpolation=cv2.INTER_LINEAR) 
    dst_2 = cv2.resize(th3_close, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    dst_3 = cv2.resize(t_otsu, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    dst_4 = cv2.resize(FAST_2, dsize=(320, 240), interpolation=cv2.INTER_LINEAR)
    dst_5 = cv2.resize(FAST_3, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    dst_6 = cv2.resize(Hough_0, dsize=(width, height), interpolation=cv2.INTER_LINEAR) 
    
    img_result = np.stack((dst_0,dst_1,dst_2,dst_3,dst_4,dst_5,dst_6),axis=-1)
    print("detect_shape= {}".format(img_result.shape))
    
    return img_result
           

if __name__ == '__main__':
    
    #################0. 입력 이미지 바꾸기!!!!###############
    # im = cv2.imread("DemoImg_BigDipper.jpeg",cv2.IMREAD_GRAYSCALE)
    im = cv2.imread("Demo_BigDipper_2.jpeg",cv2.IMREAD_GRAYSCALE)
    # im = cv2.imread("DemoImg_crux.jpeg",cv2.IMREAD_GRAYSCALE)
    
# ================테스트용 npy파일 만들기================================================
    (x,y,w,h) = GET_ROI(im) # ROI = 578,240 ~ 578,618 ~ 940,618 ~940,240
    roi_img = im[y:y+h, x:x+w]
#     
    binarize_grpAll(im)
    img_detected = binarize_grpAll(roi_img)
    # np.save(("./demo_array.npy"), img_detected)
#     
# ======================================================================================
    model = load_model('my_cnn_model_2018120133.h5') # 모델을 새로 불러옴
    
    img_input = img_detected
    # img_input = np.load("demo_array.npy")
    x_input = img_input.reshape(1, 240, 320, 7)
    
    prediction = model.predict(x_input) # 예측 수행
    
    constellation_class = {
      0: 'Big Dipper'
    , 1: 'Crux'
    , 2: 'Nothing'}
    
    class_predicted = np.argmax(prediction) # 클래스(예측)
    
    #################2. 입력 이미지에 맞게 정답 클래스 바꾸기!!!!###############
    class_actual = 0          # 클래스(정답) - 수동설정
    
    class_predicted_name = constellation_class[class_predicted] # 클래스 이름(예측)
    class_actual_name = constellation_class[class_actual] # 클래스 이름(정답)
    
    
    print("클래스(예측) : %s" %class_predicted_name) # 클래스 이름(예측) 출력
    print("클래스(정답) : %s" %class_actual_name) # 클래스 이름(정답) 출력
    
    import matplotlib.pyplot as plt
    # print(x_test[index,:,:,:])
    
    test_img = x_input[0,:,:,0].astype(np.uint8) # predict -> use for classification
    #################3. 입력 이미지에 맞게 Wide scene 이미지 바꾸기!!!!###############
    # img_wide =cv2.imread("DemoImg_crux.jpeg")
    # img_wide =cv2.imread("DemoImg_BigDipper.jpeg") #입력영상 바꾸면 바꿔줘야함!
    img_wide =cv2.imread("Demo_BigDipper_2.jpeg") #입력영상 바꾸면 바꿔줘야함!
    
    titles = ['Wide scene','ROI']
    images = [img_wide, test_img]

    for i in range(2):
            plt.subplot(1,2,i+1);plt.imshow(images[i])            
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
    plt.savefig("detect_table.png",dpi=400)
    