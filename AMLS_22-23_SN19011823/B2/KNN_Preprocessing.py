#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 18:35:55 2022

"""

import os
import numpy as np
import cv2

'''
As in this Task we use both KNN and CNN models and compare their performance, this module is used for preprocessing training image for KNN Model training.
Simply get the training labels and images as the one used in Task B1, no need for imagedatagenerator
'''

def extract_labels(basedir, labels_filename, eye_dir): ## Take dataset folder dir and label csv files as inputs, output training labels
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    eye_labels = {line.split('\t')[0] : int(line.split('\t')[1]) for line in lines[1:]}
    eye_labels = np.array(list(eye_labels.values()))
    train_labels = []
    eye_paths = [os.path.join(eye_dir, l) for l in os.listdir(eye_dir)]
    
    for img in eye_paths:
        file_name= int(img.split('.')[-2].split('/')[-1])
        train_labels.append(eye_labels[file_name])  ## reorder the training labels in consistence of training image
    
    train_labels = np.array(train_labels)
    
    return train_labels



def image_processing(basedir, eye_dir): ## this function is  used for generating training image for KNN model training
    image_paths = [os.path.join(eye_dir, l) for l in os.listdir(eye_dir)]
    train_img = []
    
    for img in image_paths:
        img_read = cv2.imread(img, flags = 1)  # Read RGB, channel = 3
        img_read = cv2.resize(img_read, (img_read.shape[1], img_read.shape[0]), interpolation = cv2.INTER_AREA)
        train_img.append(img_read)
    
    train_img = np.array(train_img)/ 255.0 # normalisation
    
    return train_img


#%%
# basedir = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/Datasets/cartoon_set'
# images_dir = os.path.join(basedir,'img')
# labels_filename = 'labels.csv'
# eye_dir = os.path.join(basedir,'eye_img')

# testdir = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/Datasets/cartoon_set_test'
# test_images_dir = os.path.join(testdir,'img')


#%%
# train_labels = extract_labels(basedir, labels_filename, eye_dir)
# train_img = image_processing(basedir, eye_dir)
# train_img = train_img.reshape((train_img.shape[0], train_img.shape[1] * train_img.shape[2] * train_img.shape[3]))
# X_train, X_val, y_train, y_val = train_test_split(train_img, train_labels, test_size = 0.2, shuffle = True)

#%% KNN
# KNN_accuracy = KNN_model.KNN_model(X_train, y_train, X_val, y_val)
#%%
# KNN_model.KNN_model_plot(KNN_accuracy)
