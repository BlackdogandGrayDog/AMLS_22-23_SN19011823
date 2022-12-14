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
import train_and_test
import cnn_model

#%%

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
    train_labels[np.where(train_labels == -1)] = 0
    
    return train_labels



def image_processing(basedir, images_dir):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    train_img = []
    
    for img in image_paths:
        img_read = cv2.imread(img, flags = 0)  # Read Grey Scale, channel = 1
        img_read = cv2.resize(img_read, (80, 80), interpolation = cv2.INTER_AREA)
        train_img.append(img_read)
    
    train_img = np.array(train_img)/ 255.0 # normalisation
    
    return train_img



def image_processing_RGB(basedir, images_dir):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    train_img = []
    
    for img in image_paths:
        img_read = cv2.imread(img, flags = 1)  # Read RGB image, channel = 3
        img_read = cv2.resize(img_read, (80, 80), interpolation = cv2.INTER_AREA)
        train_img.append(img_read)
    
    train_img = np.array(train_img)/ 255.0 # normalisation
    
    return train_img



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
    
    
    
    
#%% dir route

basedir = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/Datasets/celeba'
testdir = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/Datasets/celeba_test'
images_dir = os.path.join(basedir,'img')
test_images_dir = os.path.join(testdir,'img')
labels_filename = 'labels.csv'   
    
#%% Training Labels
y = extract_labels(basedir, labels_filename, images_dir)

## GrayScale
# X_train = image_processing(basedir, images_dir)
# X_train, X_val, y_train, y_val = train_test_split(X_train,y, test_size=0.2)

## RGB Scale
X_train_RGB = image_processing_RGB(basedir, images_dir)
# X_train_RGB, X_val_RGB, y_train_RGB, y_val_RGB = train_test_split(X_train_RGB, y, test_size=0.2)

## Test Image and Label
X_test = image_processing_RGB(testdir, test_images_dir)
y_test = extract_labels(testdir, labels_filename, test_images_dir)

history, result, model = train_and_test.train_and_test(X_train_RGB, y, X_test, y_test)
train_and_test.conf_matrix(model, X_test, y_test)

#%% Each Layer's output image
cnn_model.layer_outputs(images_dir, model)

#%% kernel_number Tuning

# kernel_nums = [4, 8, 16, 32]
# acc_max_kn, loss_min_kn = hyperpara_tuning_training.kernel_num_tuning(kernel_nums, X_train, y_train, X_val, y_val)
#%% kernel_size_tuning

# kernel_size = [(3,3), (5,5), (7,7), (11,11)]
# acc_max_ks, loss_min_ks = hyperpara_tuning_training.kernel_size_tuning(16, kernel_size, X_train, y_train, X_val, y_val)
#%% Maxpooling_size_tuning

# maxpool_size = [(2,2), (3,3), (5,5)]
# acc_max_mp, loss_min_mp = hyperpara_tuning_training.Maxpooling_size_tuning(16, (5,5), maxpool_size, X_train, y_train, X_val, y_val)

#%% fc_size_tuning

# fc_size = [64, 512, 1024, 4096]
# acc_max_fc, loss_min_fc = hyperpara_tuning_training.fc_size_tuning(16, (5,5), (2,2), fc_size, X_train, y_train, X_val, y_val)

#%% Gray_RGB

# val_acc_gr, val_loss_gr = hyperpara_tuning_training.Gray_RGB_tuning(16, (5,5), (2,2), 512, X_train, y_train, X_val, y_val, X_train_RGB, y_train_RGB, X_val_RGB, y_val_RGB)

