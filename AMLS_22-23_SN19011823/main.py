#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 23:53:28 2022


This is main.py file, click 'run file' icon, it will automatically go through all the tasks

"""
import os
import dlib
import A1.A1_main as A1_main
from A1.Preprocessing import train_image_plotting as A1_train_image_plotting
from A1.cnn_model import layer_outputs as A1_layer_outputs
import A2.A2_main as A2_main
from A2.Preprocessing import train_image_plotting as A2_train_image_plotting
from A2.Preprocessing import null_image_plot as A2_null_image_plot
import B1.B1_main as B1_main
import B2.B2_main as B2_main
import B2.train_image_plot as B2_train_image_plot

print('\n')
print('##############################')
print('Task Packages import successful ~')
print('##############################')
print('\n')

### Chapter A
A_basedir = os.path.join('Datasets','celeba')
A_testdir = os.path.join('Datasets','celeba_test')
A_images_dir = os.path.join(A_basedir,'img')
A_test_images_dir = os.path.join(A_testdir,'img')
A_labels_filename = 'labels.csv'
A2_detector = dlib.get_frontal_face_detector()
A2_predictor = dlib.shape_predictor(os.path.join('A2','shape_predictor_68_face_landmarks.dat'))   

# ### Task A1
print('\n')
print('##############################')
print('Welcome to Task A1 ~')
print('##############################')
print('\n')
A1_main.hyperpara_tuning_A1(A_basedir, A_labels_filename, A_images_dir) # task A1 hyperparameter tuning
A1_train_image_plotting(A_images_dir, 90) # task A1 training image plotting
A1_result, A1_model = A1_main.train_and_testing(A_basedir, A_images_dir, A_testdir, A_test_images_dir, A_labels_filename)  # task A1 training and tuning
A1_layer_outputs(A_images_dir, A1_model) # cnn filters output

# ### Task A2
print('\n')
print('##############################')
print('Welcome to Task A2 ~')
print('##############################')
print('\n')
A2_main.model_tuning(A2_detector, A2_predictor, A_basedir, A_labels_filename, A_images_dir)
A2_main.train_and_tesing_A2(A2_detector, A2_predictor, A_basedir, A_labels_filename, A_images_dir, A_testdir, A_test_images_dir)
A2_train_image_plotting(A_images_dir, 90, A2_detector, A2_predictor)  ## task A2 training image plotting
A2_null_image_plot(A2_detector, A2_predictor, A_basedir, A_labels_filename, A_images_dir) ## task A2 null image (no feature) plotting


### Chapter B
B_basedir = A_basedir = os.path.join('Datasets','cartoon_set')
B_testdir = os.path.join('Datasets','cartoon_set_test')
B_images_dir = os.path.join(B_basedir,'img')
B_test_images_dir = os.path.join(B_testdir,'img')
B_labels_filename = 'labels.csv'
B2_eye_dir = os.path.join(B_basedir,'eye_img')
B2_test_eye_dir = os.path.join(B_testdir,'eye_img')

### Task B1
print('\n')
print('##############################')
print('Welcome to Task B1 ~')
print('##############################')
print('\n')
B1_main.feature_training_knn_svm(A2_detector, A2_predictor, B_basedir, B_labels_filename, B_images_dir)
B1_main.model_tuning(B_basedir, B_images_dir, B_labels_filename)
B1_main.train_and_test_knn_svm(B_basedir, B_images_dir, B_labels_filename, B_testdir, B_test_images_dir)

### Task B2
print('\n')
print('##############################')
print('Welcome to Task B2 ~')
print('##############################')
print('\n')
B2_main.image_cropped_preprocessing(B_images_dir, B_basedir, B2_eye_dir, B_test_images_dir, B_testdir, B2_test_eye_dir)
B2_main.KNN_training(B_basedir, B_labels_filename, B2_eye_dir)
B2_main.CNN_hyperpara_tuning_training(B_basedir, B_labels_filename, B2_eye_dir, B_testdir, B2_test_eye_dir, B_images_dir)
B2_train_image_plot.train_image_plotting(B_images_dir, 91) ## task B2 training image plotting (feature extraction)
B2_train_image_plot.eye_image_plotting(B2_eye_dir, 97, 107, 85, 99) ## task B2 cropped eye image plotting 
