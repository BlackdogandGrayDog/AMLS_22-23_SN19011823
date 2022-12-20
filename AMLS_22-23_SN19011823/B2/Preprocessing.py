#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 21:10:01 2022

@author: ericwei
"""

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator

#%%
'''
This module contains preprocessing functions before training a cnn model.
It will first extract both file names and original cartoon images, and then cropped the eye's area, writing them into a new folder for training.
Then three functions are used for generating different imagedatagenerators for trainin and validation, training, testing and plotting a confusion matrix
'''
def extract_file_names(images_dir):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    file_names = []
    for img in image_paths:
        file_name= img.split('/')[-1] ## obtain the images' names in the form of e.g. 91.png
        file_names.append(file_name)
    
    return image_paths, file_names



def image_processing(basedir, image_paths, file_names, eye_dir):    ## takes destination folder path as input to create a new folder for training images (eye images)
    train_img = []
    eye_dirs = []
    for img in image_paths:
        img_read = cv2.imread(img, flags = 1)  # Read RGB, channel = 3
        img_read = cv2.cvtColor(img_read,cv2.COLOR_BGR2RGB)
        img_read = img_read[240:285, 180:230]   ## crop the eye area, left eye image for training
        img_read = cv2.resize(img_read, (30, 30), interpolation = cv2.INTER_AREA) ## resize into 30 x 30 pixels
        train_img.append(img_read)
    train_img = np.array(train_img)/ 255.0 # normalisation
    
    if os.path.isdir(eye_dir): ## if there is already a folder created, do not need to create again
        pass
    else:
        os.mkdir(eye_dir)   ## ## if there is no folder created, create the folder for cropped eye images
    
    for i, file_name in enumerate(file_names):
        eyedir = os.path.join(eye_dir, file_name)
        eye_dirs.append(eyedir)
        plt.imsave(eyedir, train_img[i])
        
    return eye_dirs




def extract_image_labels_df(basedir, labels_filename): ## Take dataset folder dir and label csv files as inputs
    labels_file = os.path.join(basedir, labels_filename)
    labels_df = pd.read_csv(labels_file, dtype=str, sep = '\t') ## read the csv file as string type to fit flowfromdataframe function 'categorical'
    images = labels_df['file_name']
    eye_colour = labels_df['eye_color']
    df = {'file_name': images, 'eye_color': eye_colour} ## create a dataframe contains file names and eye color
    train_df = pd.DataFrame(data=df)

    return train_df




def train_val_generator(train_df, eye_dir):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split = 0.2) ## normalise the image and using 20% of image as validation data
    
    train_gen = train_datagen.flow_from_dataframe(dataframe = train_df,
                                                    directory = eye_dir,
                                                    x_col = "file_name",
                                                    y_col = "eye_color",
                                                    target_size = (30, 30),
                                                    batch_size = 128,
                                                    shuffle = True,
                                                    class_mode = 'categorical', ## multiclass classification, so using categorical
                                                    subset = 'training')
    
    val_gen = train_datagen.flow_from_dataframe(dataframe = train_df,
                                                    directory = eye_dir,
                                                    x_col = "file_name",
                                                    y_col = "eye_color",
                                                    target_size = (30, 30),
                                                    batch_size = 128,
                                                    shuffle = True,
                                                    class_mode = 'categorical',
                                                    subset = 'validation')
    
    train_step_size = train_gen.samples // train_gen.batch_size
    val_step_size = val_gen.samples // val_gen.batch_size


    return train_gen, val_gen, train_step_size, val_step_size


def train_generator(train_df, eye_dir):
    train_datagen = ImageDataGenerator(rescale=1./255) ## normalise the image and no validation data, for training and testing
    
    train_gen = train_datagen.flow_from_dataframe(dataframe = train_df,
                                                    directory = eye_dir,
                                                    x_col = "file_name",
                                                    y_col = "eye_color",
                                                    target_size = (30, 30),
                                                    batch_size = 128,
                                                    shuffle = True,
                                                    class_mode = 'categorical')
    
    train_step_size = train_gen.samples // train_gen.batch_size



    return train_gen, train_step_size



def test_cm_generator(test_df, test_eye_dir):
    test_datagen = ImageDataGenerator(rescale=1./255) ## normalise the image and no validation data
    
    test_gen = test_datagen.flow_from_dataframe(dataframe = test_df,
                                                    directory = test_eye_dir,
                                                    x_col = "file_name",
                                                    y_col = "eye_color",
                                                    target_size = (30, 30),
                                                    batch_size = 128,
                                                    shuffle = False, ## dataset is not being shuffle for confusion_matrix plotting
                                                    class_mode = 'categorical')
    
    test_step_size = np.math.ceil(test_gen.samples / test_gen.batch_size)



    return test_gen, test_step_size
