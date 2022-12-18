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

import train_image_plot
import hyperpara_tuning_training
import train_and_test
from keras.models import load_model

#%%

def extract_file_names(images_dir):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    file_names = []
    for img in image_paths:
        file_name= img.split('/')[-1]
        file_names.append(file_name)
    
    return image_paths, file_names



def image_processing(basedir, image_paths, file_names, eye_dir):
    train_img = []
    eye_dirs = []
    for img in image_paths:
        img_read = cv2.imread(img, flags = 1)  # Read RGB, channel = 3
        img_read = cv2.cvtColor(img_read,cv2.COLOR_BGR2RGB)
        img_read = img_read[240:285, 180:230]
        img_read = cv2.resize(img_read, (30, 30), interpolation = cv2.INTER_AREA)
        train_img.append(img_read)
    train_img = np.array(train_img)/ 255.0 # normalisation
    
    if os.path.isdir(eye_dir):
        pass
    else:
        os.mkdir(eye_dir)
    
    for i, file_name in enumerate(file_names):
        eyedir = os.path.join(eye_dir, file_name)
        eye_dirs.append(eyedir)
        plt.imsave(eyedir, train_img[i])
        
    return eye_dirs




def extract_image_labels_df(basedir, labels_filename): ## Take dataset folder dir and label csv files as inputs
    labels_file = os.path.join(basedir, labels_filename)
    labels_df = pd.read_csv(labels_file, dtype=str, sep = '\t')
    images = labels_df['file_name']
    eye_colour = labels_df['eye_color']
    df = {'file_name': images, 'eye_color': eye_colour}
    train_df = pd.DataFrame(data=df)

    return train_df




def train_val_generator(train_df, eye_dir):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split = 0.2)
    
    train_gen = train_datagen.flow_from_dataframe(dataframe = train_df,
                                                    directory = eye_dir,
                                                    x_col = "file_name",
                                                    y_col = "eye_color",
                                                    target_size = (30, 30),
                                                    batch_size = 128,
                                                    shuffle = True,
                                                    class_mode = 'categorical',
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
    train_datagen = ImageDataGenerator(rescale=1./255)
    
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
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_gen = test_datagen.flow_from_dataframe(dataframe = test_df,
                                                    directory = test_eye_dir,
                                                    x_col = "file_name",
                                                    y_col = "eye_color",
                                                    target_size = (30, 30),
                                                    batch_size = 128,
                                                    shuffle = False,
                                                    class_mode = 'categorical')
    
    test_step_size = np.math.ceil(test_gen.samples / test_gen.batch_size)



    return test_gen, test_step_size


    




#%%
basedir = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/Datasets/cartoon_set'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'
eye_dir = os.path.join(basedir,'eye_img')

testdir = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/Datasets/cartoon_set_test'
test_images_dir = os.path.join(testdir,'img')
test_eye_dir = os.path.join(testdir,'eye_img')

#%%
image_paths, file_names = extract_file_names(images_dir)
eye_dirs = image_processing(basedir, image_paths, file_names, eye_dir)

#%%
train_image_plot.train_image_plotting(images_dir, 91)
train_image_plot.eye_image_plotting(eye_dir, 97, 107, 85, 99)

#%%
train_df = extract_image_labels_df(basedir, labels_filename)
train_gen, val_gen, train_step_size, val_step_size = train_val_generator(train_df, eye_dir)

#%%
kernel_nums = [4, 8, 16, 32]
acc_max_kn, loss_min_kn = hyperpara_tuning_training.kernel_num_tuning(kernel_nums, train_gen, train_step_size, val_gen, val_step_size)

#%%
kernel_size = [(2,2), (3,3), (5,5), (7,7)]
acc_max_ks, loss_min_ks = hyperpara_tuning_training.kernel_size_tuning(16, kernel_size, train_gen, train_step_size, val_gen, val_step_size)

#%%
maxpool_size = [(2,2), (3,3), (5,5)]
acc_max_mp, loss_min_mp = hyperpara_tuning_training.Maxpooling_size_tuning(16, (7,7), maxpool_size, train_gen, train_step_size, val_gen, val_step_size)

#%%
fc_size = [8, 64, 512, 4096]
acc_max_fc, loss_min_fc = hyperpara_tuning_training.fc_size_tuning(16, (7,7), (2,2), fc_size, train_gen, train_step_size, val_gen, val_step_size)

#%%
test_image_paths, test_file_names = extract_file_names(test_images_dir)
test_eye_dirs = image_processing(testdir, test_image_paths, test_file_names, test_eye_dir)
#%%
test_df = extract_image_labels_df(testdir, labels_filename)
train_gen, train_step_size = train_generator(train_df, eye_dir)
test_gen, test_step_size = train_generator(test_df, test_eye_dir)

#%%
history, result, model = train_and_test.train_and_test(train_gen, train_step_size, test_gen, test_step_size)
#%%
test_gen_cm, test_step_size_cm = test_cm_generator(test_df, test_eye_dir)
#%%
model = load_model('B2_final_model.h5')
train_and_test.conf_matrix(model, test_gen_cm, test_step_size_cm)