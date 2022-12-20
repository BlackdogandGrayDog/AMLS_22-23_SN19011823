#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 02:55:12 2022

@author: ericwei
"""
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
this module is to extract original cartoon image as in A1, no longer converted to 68 landmark points
'''

def extract_labels(basedir, labels_filename, images_dir): ## Take dataset folder dir and label csv files as inputs
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    gender_labels = np.array(list(gender_labels.values()))
    train_labels = []
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    
    for img in image_paths:
        file_name= int(img.split('.')[-2].split('/')[-1])
        train_labels.append(gender_labels[file_name])
    
    train_labels = np.array(train_labels)
    
    return train_labels


'''
this function use cv2 imread to read RGB scale image for training and resize to (80,80) for training
'''
def image_processing(basedir, images_dir):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    train_img = []
    
    for img in image_paths:
        img_read = cv2.imread(img, flags = 1)  # Read RGB, channel = 3
        img_read = cv2.resize(img_read, (80, 80), interpolation = cv2.INTER_AREA)
        train_img.append(img_read)
    
    train_img = np.array(train_img)/ 255.0 # normalisation
    
    return train_img

'''
as the name shows, this function used for ploting training image as in Task A1
'''
def train_image_plotting(images_dir, num_image):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    img = cv2.imread(image_paths[num_image], flags = 1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(1,2,1)
    plt.title('Orginal Image', fontsize = 10, fontweight='bold')
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.title('Resized Image', fontsize = 10, fontweight='bold')
    img = cv2.resize(img, (80, 80), interpolation = cv2.INTER_AREA)
    plt.imshow(img)
    plt.show()
    