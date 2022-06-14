#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 04:38:54 2022

@author: Pyo SeungHyun

코드의 역할
1. 여러 폴더에 나뉘어 저장된 사진들을 3차원배열으로 만들어 npy 저장형식으로 저장하기
2. 영상의 크기를 조정하기(width, height)

"""

import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt 
from PIL import Image


class stkImg:
    def __init__(self,class_index):
        
        self.ClassDir="./dataset/class"+class_index
        print("classdir: " +self.ClassDir)
        
        
        self.preprocess_dir=[self.ClassDir+"/binary_0", self.ClassDir+"/binary_1", self.ClassDir+"/binary_2", 
                             self.ClassDir+"/FAST_0",self.ClassDir+"/FAST_1",self.ClassDir+"/Hough",]
        
        self.result_path_color = self.ClassDir+"/test_stack_color"
        self.result_path_gray = self.ClassDir+"/test_stack"
        
        files_00 = sorted(glob.glob(self.ClassDir + "/line_1/*.jpeg"))
        
        self.len_ = len(files_00)
        
        
        print(self.len_)
        
    def Table_howto(self, imgS):
        titles = ['Origin[0]','bin_glob[1]','bin_Morph[2]','bin_otsu[3]','FAST_0[4]','FAST_1[5]','LineDetect[6]']
        images = imgS
        
        for i in range(7):
            plt.subplot(2,4,i+1);plt.imshow(images[i])
            
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
            
        plt.savefig("/home/chicken/PSH/project_imgs/stackImg_inside_gray.png",dpi=300)    
    
    
    def run(self,colortype,width,height):
        for i in range(self.len_):
        # for i in range(150):
            n=i
            if colortype == "RGB":
                img_origin = cv2.imread(self.ClassDir+"/line_1"+"/"+str(n)+".jpeg")  
                img_1 = cv2.imread(self.preprocess_dir[0]+"/"+str(n)+".jpeg")
                img_2 = cv2.imread(self.preprocess_dir[1]+"/"+str(n)+".jpeg")
                img_3 = cv2.imread(self.preprocess_dir[2]+"/"+str(n)+".jpeg")
                img_4 = cv2.imread(self.preprocess_dir[3]+"/"+str(n)+".jpeg")
                img_5 = cv2.imread(self.preprocess_dir[4]+"/"+str(n)+".jpeg")
                img_6 = cv2.imread(self.preprocess_dir[5]+"/"+str(n)+".jpeg")
            
            elif colortype == "gray":    
                img_origin = cv2.imread(self.ClassDir+"/line_1"+"/"+str(n)+".jpeg",cv2.IMREAD_GRAYSCALE)
                img_1 = cv2.imread(self.preprocess_dir[0]+"/"+str(n)+".jpeg",cv2.IMREAD_GRAYSCALE)
                img_2 = cv2.imread(self.preprocess_dir[1]+"/"+str(n)+".jpeg",cv2.IMREAD_GRAYSCALE)
                img_3 = cv2.imread(self.preprocess_dir[2]+"/"+str(n)+".jpeg",cv2.IMREAD_GRAYSCALE)
                img_4 = cv2.imread(self.preprocess_dir[3]+"/"+str(n)+".jpeg",cv2.IMREAD_GRAYSCALE)
                img_5 = cv2.imread(self.preprocess_dir[4]+"/"+str(n)+".jpeg",cv2.IMREAD_GRAYSCALE)
                img_6 = cv2.imread(self.preprocess_dir[5]+"/"+str(n)+".jpeg",cv2.IMREAD_GRAYSCALE)
            
            dst_0 = cv2.resize(img_origin, dsize=(width, height), interpolation=cv2.INTER_AREA)
            dst_1 = cv2.resize(img_1, dsize=(width, height), interpolation=cv2.INTER_AREA)
            dst_2 = cv2.resize(img_2, dsize=(width, height), interpolation=cv2.INTER_AREA)
            dst_3 = cv2.resize(img_3, dsize=(width, height), interpolation=cv2.INTER_AREA)
            dst_4 = cv2.resize(img_4, dsize=(width, height), interpolation=cv2.INTER_AREA)
            dst_5 = cv2.resize(img_5, dsize=(width, height), interpolation=cv2.INTER_AREA)
            dst_6 = cv2.resize(img_6, dsize=(width, height), interpolation=cv2.INTER_AREA)
            print("영상 크기 조정(너비: {}, 높이: {}) 완료 ".format(width, height))
            
            img_result = np.stack((dst_0,dst_1,dst_2,dst_3,dst_4,dst_5,dst_6),axis=-1)
            print("영상 3차원 배열화 완료 ")
            
            if colortype == "RGB":
                np.save((self.result_path_color+"/"+str(n)+".npy"), img_result)
            elif colortype=="gray" :
                np.save((self.result_path_gray+"/"+str(n)+".npy"), img_result)
                
            print("{}번째 영상 3차원 배열 저장 완료 ".format(n))
            
# =============================================================================
#             ## 보고서용 테이블
#             # imgS=[img_result[:,:,:,0],img_result[:,:,:,1],img_result[:,:,:,2],
#             #       img_result[:,:,:,3],img_result[:,:,:,4],img_result[:,:,:,5],img_result[:,:,:,6]]
#             imgS=[img_result[:,:,0],img_result[:,:,1],img_result[:,:,2],
#                   img_result[:,:,3],img_result[:,:,4],img_result[:,:,5],img_result[:,:,6]]
#             
#             stkImg.Table_howto(self, imgS)
# =============================================================================
            
            # cv2.imshow("test",img_result[:,:,:,5]) ##test_stack_color
            # cv2.imshow("test",img_result[:,:,5])   ##test_stack_gray
            # cv2.waitKey(0)
            
            
if __name__ == '__main__':
    
    
    
    colortype= "gray"       ##colortype : "RGB" or "gray"
    width = 320             # 320
    height = 240            # 240
    
    GO_0 = stkImg("00") ## Class index
    GO_0.run(colortype,width,height)
    GO_1 = stkImg("01")
    GO_1.run(colortype,width,height)
    GO_2 = stkImg("02")
    GO_2.run(colortype,width,height)
    # GO_test = stkImg("detect")
    # GO_test.run(colortype,width,height)
    
    
    

