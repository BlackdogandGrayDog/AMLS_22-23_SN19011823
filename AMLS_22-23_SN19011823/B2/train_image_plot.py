#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 20:46:56 2022

"""

from matplotlib import pyplot as plt
import os
import cv2

'''
as the module name shows, it is used for plotting image in this cases.
The first function compare the original image and cropped eye image.
The second one gives several different examples of eye images
'''

def train_image_plotting(images_dir, num_image): ## this function compare the original image and cropped eye image used for training
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    img = cv2.imread(image_paths[num_image], flags = 1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(1,2,1)
    plt.title('Orginal Image', fontsize = 10, fontweight='bold')
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.title('Eye Image', fontsize = 10, fontweight='bold')
    img = img[240:285, 180:230]
    img = cv2.resize(img, (30, 30), interpolation = cv2.INTER_AREA)
    plt.imshow(img)
    plt.show()
    
    
def eye_image_plotting(eye_dir, num_image, num_image2, num_image3, num_image4): ## this function gives several different examples of eye images
    image_paths = [os.path.join(eye_dir, l) for l in os.listdir(eye_dir)]
    img = cv2.imread(image_paths[num_image], flags = 1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(2,2,1)
    plt.title('Sunglasses', fontsize = 10, fontweight='bold')
    plt.subplots_adjust(wspace=0.3,hspace=0.5)
    plt.imshow(img)
    
    img = cv2.imread(image_paths[num_image2], flags = 1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(2,2,2)
    plt.title('Fade Sunglasses', fontsize = 10, fontweight='bold')
    plt.imshow(img)

    img = cv2.imread(image_paths[num_image3], flags = 1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(2,2,3)
    plt.title('circle eyeglasses', fontsize = 10, fontweight='bold')
    plt.subplots_adjust(wspace=0.1,hspace=0.5)
    plt.imshow(img)
    
    img = cv2.imread(image_paths[num_image4], flags = 1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(2,2,4)
    plt.title('rectangle eyeglasses', fontsize = 10, fontweight='bold')
    plt.imshow(img)
