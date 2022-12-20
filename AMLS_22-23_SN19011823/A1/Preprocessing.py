#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 02:06:13 2022

@author: ericwei
"""

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
#%%

''' 
This module includes some functions to extract training, validation, test images and labels, and corresponding labels. 
Also, there are preprocessing precedures on processing images before training a cnn model
'''
def __init__(self):
    pass
        
'''
Take dataset folder dir and label csv files as inputs, extract gender_labels in the csv file (-1 is femail, 1 is male)
Convert to one hot encoding (0 and 1) and output labels
'''
def extract_labels(basedir, labels_filename, images_dir): 
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    gender_labels = np.array(list(gender_labels.values()))
    train_labels = []
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    
    for img in image_paths:
        file_name= int(img.split('.')[-2].split('/')[-1])
        train_labels.append(gender_labels[file_name]) ## reorder the labels to the consistent fashion as training image
    
    train_labels = np.array(train_labels)
    train_labels[np.where(train_labels == -1)] = 0  ## One hot encoding, convert -1 to 0
    
    return train_labels

'''
Take dataset folder dir and images_dir as input, use cv2 to read each image to gray scale. 
Resize and preprocess the image to 80 x 80 pixel and normalisation]
'''

def image_processing(basedir, images_dir):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    train_img = []
    
    for img in image_paths:
        img_read = cv2.imread(img, flags = 0)  # Read Grey Scale, channel = 1
        img_read = cv2.resize(img_read, (80, 80), interpolation = cv2.INTER_AREA) ## resize the image to 80 x 80
        train_img.append(img_read)
    
    train_img = np.array(train_img)/ 255.0 # normalisation
    
    return train_img


'''
Take dataset folder dir and images_dir as input, use cv2 to read each image to RGBscale. 
Resize and preprocess the image to 80 x 80 pixel and normalisation]
'''
def image_processing_RGB(basedir, images_dir):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    train_img = []
    
    for img in image_paths:
        img_read = cv2.imread(img, flags = 1)  # Read RGB image, channel = 3
        img_read = cv2.resize(img_read, (80, 80), interpolation = cv2.INTER_AREA) ## resize the image to 80 x 80
        train_img.append(img_read)
    
    train_img = np.array(train_img)/ 255.0 # normalisation
    
    return train_img


'''
As the function name shows, it plots the original image and processed image for trainning the CNN model
'''
def train_image_plotting(images_dir, num_image):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    img = cv2.imread(image_paths[num_image], flags = 1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(1,2,1)
    plt.imshow(img)
    img = cv2.imread(image_paths[num_image], flags = 0)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.figure()
    plt.subplot(1,2,1)
    img = cv2.resize(img, (80, 80))
    plt.imshow(img)
    plt.subplot(1,2,2)
    img = cv2.resize(img, (80, 80), interpolation = cv2.INTER_AREA)
    plt.imshow(img)
    plt.show()

    
