#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 19:55:29 2022

@author: ericwei

this module contains two main operations in Task A1, first one is hyperparameter tuning, the second one is training and test of the model after tuning.
which uses whole training dataset and test dataset.
"""
import A1.Preprocessing as Preprocessing
import A1.hyperpara_tuning_training as hyperpara_tuning_training
import A1.train_and_test as train_and_test
from sklearn.model_selection import train_test_split

#%%
def hyperpara_tuning_A1(basedir, labels_filename, images_dir): ## this function is the first part, extract images and labels and tuning hyper-parameters
    y = Preprocessing.extract_labels(basedir, labels_filename, images_dir)
    X_train = Preprocessing.image_processing(basedir, images_dir)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y, test_size=0.2)
    
    print('\n')
    print(' This is kernel number tuning from 4, 8, 16, 32')
    kernel_nums = [4, 8, 16, 32]
    acc_max_kn, loss_min_kn = hyperpara_tuning_training.kernel_num_tuning(kernel_nums, X_train, y_train, X_val, y_val)
    print('\n')
    print('Each choice can achive highest accuracy of:')
    print(acc_max_kn)
    print('\n')
    print('Each choice can achive lowest loss of:')
    print(loss_min_kn)
    print('\n')
    print('The final choise is 16')
    
    
    print('\n')
    print('This is kernel number tuning from (3,3), (5,5), (7,7), (11,11)')
    kernel_size = [(3,3), (5,5), (7,7), (11,11)]
    acc_max_ks, loss_min_ks = hyperpara_tuning_training.kernel_size_tuning(16, kernel_size, X_train, y_train, X_val, y_val)
    print('\n')
    print('Each choice can achive highest accuracy of:')
    print(acc_max_ks)
    print('\n')
    print('Each choice can achive lowest loss of:')
    print(loss_min_ks)
    print('\n')
    print('The final choise is (5,5)')
    
    
    print('\n')
    print('This is maxpooling size tuning from (2,2), (3,3), (5,5)')
    maxpool_size = [(2,2), (3,3), (5,5)]
    acc_max_mp, loss_min_mp = hyperpara_tuning_training.Maxpooling_size_tuning(16, (5,5), maxpool_size, X_train, y_train, X_val, y_val)
    print('\n')
    print('Each choice can achive highest accuracy of:')
    print(acc_max_mp)
    print('\n')
    print('Each choice can achive lowest loss of:')
    print(loss_min_mp)
    print('\n')
    print('The final choise is (2,2)')
    
    
    print('\n')
    print('This is fully connected layer tuning from 64, 512, 1024, 4096 neurones')
    fc_size = [64, 512, 1024, 4096]
    acc_max_fc, loss_min_fc = hyperpara_tuning_training.fc_size_tuning(16, (5,5), (2,2), fc_size, X_train, y_train, X_val, y_val)
    print('\n')
    print('Each choice can achive highest accuracy of:')
    print(acc_max_fc)
    print('\n')
    print('Each choice can achive lowest loss of:')
    print(loss_min_fc)
    print('\n')
    print('The final choise is 512')
    
    
    ## Reprocess the image to inclue RBG scale unafes
    print('\n')
    print('This is RGB or gray Scale tuning ')
    X_train_RGB = Preprocessing.image_processing_RGB(basedir, images_dir)
    X_train_RGB, X_val_RGB, y_train_RGB, y_val_RGB = train_test_split(X_train_RGB, y, test_size=0.2)
    val_acc_gr, val_loss_gr = hyperpara_tuning_training.Gray_RGB_tuning(16, (5,5), (2,2), 512, X_train, y_train, X_val, y_val, X_train_RGB, y_train_RGB, X_val_RGB, y_val_RGB)
    print('\n')
    print('Each choice can achive highest accuracy of:')
    print(val_acc_gr)
    print('\n')
    print('Each choice can achive lowest loss of:')
    print(val_loss_gr)
    print('\n')
    print('The final choise is RGB Scale for trainning')
    
    
def train_and_testing(basedir, images_dir, testdir, test_images_dir, labels_filename): ## this function is the second part, training and testing tuned model.
    # RGB Scale Training dataset
    X_train_RGB = Preprocessing.image_processing_RGB(basedir, images_dir)
    y_train_RGB = Preprocessing.extract_labels(basedir, labels_filename, images_dir)
    # Test Image and Label
    X_test = Preprocessing.image_processing_RGB(testdir, test_images_dir)
    y_test = Preprocessing.extract_labels(testdir, labels_filename, test_images_dir)
    
    # Training
    print('\n')
    print('Task A1 Model training starts:')
    print('\n')
    history, result, model = train_and_test.train_and_test(X_train_RGB, y_train_RGB, X_test, y_test)
    print('\n')
    print('The final score of the model is:')
    print(result)
    print('\n')
    print('Confusion Matrix:')
    print('\n')
    train_and_test.conf_matrix(model, X_test, y_test)
    
    return result, model

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
