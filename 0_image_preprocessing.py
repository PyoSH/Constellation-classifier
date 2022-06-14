# -*- coding: utf-8 -*-
"""
Created on Sat May 14 01:12:18 2022

@author: 표승현

코드의 역할
1. 캡쳐된 영상을 여러 전처리 방법을 통해 총 6개의 추가된 영상을 만들기
2. 보고서에 사용될 요약자료 만들기

주의사항: 클래스02(=없음)에는 직선이 없기 때문에 허프 변환을 하면 안된다.

"""
import sys
import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import platform
import glob



def hangulFilePathImageRead ( filePath ) :
    img_array = np.fromfile(filePath, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    img = np.ascontiguousarray(img)

    return img


class prepro:
    def __init__(self,class_index):
        
        self.ClassDir="./dataset/class"+class_index
        print("classdir: " +self.ClassDir)
        
        
        self.preprocess_dir=[self.ClassDir+"/binary_0", self.ClassDir+"/binary_1", self.ClassDir+"/binary_2", 
                             self.ClassDir+"/FAST_0",self.ClassDir+"/FAST_1",self.ClassDir+"/Hough",]
        
        
        files_00 = sorted(glob.glob(self.ClassDir + "/line_1/*.jpeg"))
        
        self.len_ = len(files_00)
        self.gray_img = np.zeros((640,480),np.uint8)
        
        print(self.len_)
        
    def Table(self, imgS,Tablename):
        titles = ['Original','Global','GaussClose','OTSU']
        images = imgS
        
        for i in range(4):
            plt.subplot(2,2,i+1);plt.imshow(images[i])            
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
            
        # plt.show()
        
        if Tablename=="binary":
            plt.savefig("./gray_table.png",dpi=300)
        elif Tablename=="table_fast":
            plt.savefig("./fast_table.png",dpi=300)
        
    def Table_howto(self, imgS,Tablename):
        titles = ['Original','binary','Morph','FAST','LineDetect']
        images = imgS
        
        for i in range(5):
            plt.subplot(1,5,i+1);plt.imshow(images[i])
            # plt.imshow(images[i],Tablename)
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
            
        plt.savefig("./preProcessing.png",dpi=300)    
    
    def LINEDETECT(img):
        img_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        minLineLength = 100
        maxLineGap = 0
        
        lines = cv2.HoughLinesP(img,1,np.pi/360,133,minLineLength,maxLineGap)
        for i in range(len(lines)):
            for x1,y1,x2,y2 in lines[i]:
                cv2.line(img_color,(x1,y1),(x2,y2),(0,0,255),15)
        
        return img_color
    
    def ROI_detect(img_color):
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        ret, img_binary = cv2.threshold(img_gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        
        for cnt in contours:
            cv2.drawContours(img_color, [cnt], 0, (255, 0, 0), 3)  # blue
        
        cv2.imshow("result0", img_color)
        
        cv2.waitKey(0)
        
        
        for cnt in contours:
        
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        cv2.imshow("result1", img_color)
        
        cv2.waitKey(0)
        
        
        
        for cnt in contours:
        
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color,[box],0,(0,0,255),2)
        
        
        cv2.imshow("result2", img_color)
        
        cv2.waitKey(0)
        
        return box
        
        
    def binarize(self, gray_img,n):
        
        file_path_ = self.ClassDir+"/line_1"+"/"+str(n)+".jpeg"
        
        ## 2. 임계값을 기준으로 이진화하기
        ret, th1 = cv2.threshold(gray_img,150,255, cv2.THRESH_BINARY)
        th3 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,15)
        t, t_otsu = cv2.threshold(gray_img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # print('otsu threshold:', t)
        
        kernel = np.ones((13, 13), np.uint8)
        th3_close = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
        
        ## 2.1 FAST전처리
        fast = cv2.FastFeatureDetector_create()
        kp_0 = fast.detect(gray_img,None)
        FAST_0=cv2.drawKeypoints(gray_img, kp_0, None)
        
        kp_1 = fast.detect(th1,None)
        FAST_1 =cv2.drawKeypoints(th1, kp_1, None)
        
        kp_2 = fast.detect(th3_close,None)
        FAST_2 =cv2.drawKeypoints(th3_close, kp_2, None)
        
        kp_3 = fast.detect(t_otsu,None)
        FAST_3 =cv2.drawKeypoints(t_otsu, kp_3, None)
        
        # print("Threshold: ", fast.getThreshold())
        # print("nonmaxSuppression: ", fast.getNonmaxSuppression())
        # print("neighborhood: ", fast.getType())
        # print("Total Keypoints with nonmaxSuppression: ", len(kp_0))
        
        
        ## 2.2 Hough전처리
        # Hough_0 = prepro.LINEDETECT(t_otsu)
        if self.ClassDir == "./dataset/class00" or self.ClassDir == "./dataset/class01" :
            Hough_0 = prepro.LINEDETECT(th3_close)
        else:
            Hough_0 = th3
            
        
        # prepro.ROI_detect(Hough_0)
        
        ## 3. 전처리된 이미지 저장하기
        #city_date = "cheonan_20220522"
        cv2.imwrite((self.preprocess_dir[0]+"/"+str(n)+".jpeg"), th1)
        cv2.imwrite((self.preprocess_dir[1]+"/"+str(n)+".jpeg"), th3_close)
        cv2.imwrite((self.preprocess_dir[2]+"/"+str(n)+".jpeg"), t_otsu)
        cv2.imwrite((self.preprocess_dir[3]+"/"+str(n)+".jpeg"), FAST_2)
        cv2.imwrite((self.preprocess_dir[4]+"/"+str(n)+".jpeg"), FAST_3)
        cv2.imwrite((self.preprocess_dir[5]+"/"+str(n)+".jpeg"), Hough_0)## for class00, class01       
        print("{} 번째 전처리 저장완료" .format(n))
        

# =============================================================================
#         ## 4. 보고서용 표 만들기
#         imgS = [gray_img,th1,th3_close,t_otsu]
#         prepro.Table(self, imgS, "binary")
#         print("이진화 비교 표 생성완료")
#         
#         imgS_1 = [FAST_0,FAST_1,FAST_2,FAST_3]
#         prepro.Table(self, imgS_1, "table_fast")
#         print("FAST 비교 표 생성완료")
#         
#         imgS_howto=[gray_img,th3,th3_close,FAST_2,Hough_0]
#         prepro.Table_howto(self, imgS_howto, "PreProcessing")
#         print("전처리 표 생성완료")
# =============================================================================

        
        return t_otsu
    
    
    
    def run(self):
        for i in range(self.len_):
        # for i in range(1):
            
            num = i
            ## 1. 사진 불러오기 &8UC1으로 만들기
            file_path_ = self.ClassDir+"/line_1"+"/"+str(num)+".jpeg"
            src = hangulFilePathImageRead(file_path_)
            if src is None:
                print('Image load failed')
                sys.exit()
            
            ## 2~3번 
            self.gray_img = prepro.binarize(self,src,num)
             

        cv2.waitKey(0)
        
            
if __name__ == '__main__':
    print(platform.platform())
    
    GO_0 = prepro("00")
    GO_0.run()
    GO_1 = prepro("01")
    GO_1.run()
    GO_2 = prepro("02")
    GO_2.run()
    # GO_test = prepro("detect")
    # GO_test.run()
    
    
   

