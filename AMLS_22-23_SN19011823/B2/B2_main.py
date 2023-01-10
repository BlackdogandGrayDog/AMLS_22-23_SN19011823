#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:41:25 2022


This module summarises all the operations in Task B2, including three main parts, the first one will cropped and preprocessing the training image,
the second part will involve the knn training, the third part will be the cnn model tuning training and testing

"""
import B2.Preprocessing as Preprocessing
import B2.KNN_Preprocessing as KNN_Preprocessing
import B2.KNN_model as KNN_model
import B2.hyperpara_tuning_training as hyperpara_tuning_training
import B2.train_and_test as train_and_test
import B2.CNN_model as CNN_model
from sklearn.model_selection import train_test_split
import numpy as np

def image_cropped_preprocessing(images_dir, basedir, eye_dir, test_images_dir, testdir, test_eye_dir):
    print('\n')
    print('Cropping the image to left eye image:')
    image_paths, file_names = Preprocessing.extract_file_names(images_dir)
    print('\n')
    print('Dealing with the training image:')
    _ = Preprocessing.image_processing(basedir, image_paths, file_names, eye_dir)
    print('\n')
    print('Training image cropping successful, check the folder in cartoon_set:')
    print('\n')
    print('Dealing with the testing image:')
    test_image_paths, test_file_names = Preprocessing.extract_file_names(test_images_dir)
    _ = Preprocessing.image_processing(testdir, test_image_paths, test_file_names, test_eye_dir)
    print('\n')
    print('Testing image cropping successful, check the folder in cartoon_test_set:')


def KNN_training(basedir, labels_filename, eye_dir):
    print('\n')
    print('Let us test KNN Model performance:')
    print('\n')
    print('Caution: it may take a while ~')
    print('\n')
    print('Extracting labels and images ~')
    train_labels = KNN_Preprocessing.extract_labels(basedir, labels_filename, eye_dir)
    train_img = KNN_Preprocessing.image_processing(basedir, eye_dir)
    train_img = train_img.reshape((train_img.shape[0], train_img.shape[1] * train_img.shape[2] * train_img.shape[3])) ## KNN model can only intake 2 parameters, therefore, we need to convert 3 channel into 1
    X_train, X_val, y_train, y_val = train_test_split(train_img, train_labels, test_size = 0.2, shuffle = True) ## using 20% of random data for validation
    print('\n')
    print('Extracting labels and images successful~')
    print('\n')
    print('Training in progress ~')
    KNN_accuracy = KNN_model.KNN_model(X_train, y_train, X_val, y_val)
    KNN_accuracy = np.array(KNN_accuracy)
    print('\n')
    print('Training finished ~')
    print('\n')
    print('KNN can reach the highest accuracy of:')
    print(max(KNN_accuracy))
    print('with the number of neighbours:')
    position = np.where(KNN_accuracy == max(KNN_accuracy))
    position = (np.array(position) * 5) + 1 ## since the line space of number of k is 5, therefore position need to be times by 5, +1 becasue k starts from 1
    print(position)
    print('\n')
    print('Here is the plot of accuracy versus number of neighbours:')
    KNN_model.KNN_model_plot(KNN_accuracy)



def CNN_hyperpara_tuning_training(basedir, labels_filename, eye_dir, testdir, test_eye_dir, images_dir):
    print('\n')
    print('Let us now tune the cnn model hyper-parameters:')
    print('\n')
    print('Generating train, validation data generator:')
    train_df = Preprocessing.extract_image_labels_df(basedir, labels_filename)
    train_gen, val_gen, train_step_size, val_step_size = Preprocessing.train_val_generator(train_df, eye_dir)
    
    print('\n')
    print('Kernel number tuning in progress:')
    print('Tuning from 4, 8, 16, 32:')
    kernel_nums = [4, 8, 16, 32]
    acc_max_kn, loss_min_kn = hyperpara_tuning_training.kernel_num_tuning(kernel_nums, train_gen, train_step_size, val_gen, val_step_size)
    print('\n')
    print('Each choice can achive highest accuracy of:')
    print(acc_max_kn)
    print('\n')
    print('Each choice can achive lowest loss of:')
    print(loss_min_kn)
    print('\n')
    print('The final choise is 16')
    
    print('\n')
    print('Kernel size tuning in progress:')
    print('Tuning from (2,2), (3,3), (5,5), (7,7):')
    kernel_size = [(2,2), (3,3), (5,5), (7,7)]
    acc_max_ks, loss_min_ks = hyperpara_tuning_training.kernel_size_tuning(16, kernel_size, train_gen, train_step_size, val_gen, val_step_size)
    print('\n')
    print('Each choice can achive highest accuracy of:')
    print(acc_max_ks)
    print('\n')
    print('Each choice can achive lowest loss of:')
    print(loss_min_ks)
    print('\n')
    print('The final choise is (7,7)')
    
    print('\n')
    print('Max pooling size tuning in progress:')
    print('Tuning from (2,2), (3,3), (5,5):')
    maxpool_size = [(2,2), (3,3), (5,5)]
    acc_max_mp, loss_min_mp = hyperpara_tuning_training.Maxpooling_size_tuning(16, (7,7), maxpool_size, train_gen, train_step_size, val_gen, val_step_size)
    print('\n')
    print('Each choice can achive highest accuracy of:')
    print(acc_max_mp)
    print('\n')
    print('Each choice can achive lowest loss of:')
    print(loss_min_mp)
    print('\n')
    print('The final choise is (2,2)')
    
    
    print('\n')
    print('Fully connected layer size tuning in progress:')
    print('Tuning from 8, 64, 512, 4096:')
    fc_size = [8, 64, 512, 4096]
    acc_max_fc, loss_min_fc = hyperpara_tuning_training.fc_size_tuning(16, (7,7), (2,2), fc_size, train_gen, train_step_size, val_gen, val_step_size)
    print('\n')
    print('Each choice can achive highest accuracy of:')
    print(acc_max_fc)
    print('\n')
    print('Each choice can achive lowest loss of:')
    print(loss_min_fc)
    print('\n')
    print('The final choise is 512')
    
    print('\n')
    print('The tuning process is completed.')
    test_df = Preprocessing.extract_image_labels_df(testdir, labels_filename)
    train_gen, train_step_size = Preprocessing.train_generator(train_df, eye_dir)
    test_gen, test_step_size = Preprocessing.train_generator(test_df, test_eye_dir)
    print('\n')
    print('First round testing initiates, determine which epoch that overfitting starts (default training epochs: 20).')
    print('\n')
    history, result, model = train_and_test.train_and_test(train_gen, train_step_size, test_gen, test_step_size, 20)
    print('\n')
    print('The overfitting starts approximately from the 13th epoch.')
    print('\n')
    print('Revise the epoches to 13.')
    print('\n')
    history, result, model = train_and_test.train_and_test(train_gen, train_step_size, test_gen, test_step_size, 13)
    print('\n')
    print('The final score of the model is:')
    print(result)
    print('\n')
    print('Generating Non-shuffled test data for plotting confusion matrix:')
    print('\n')
    test_gen_cm, test_step_size_cm = Preprocessing.test_cm_generator(test_df, test_eye_dir)
    print('Confusion Matrix:')
    print('\n')
    train_and_test.conf_matrix(model, test_gen_cm, test_step_size_cm)
    
    print('\n')
    print('The cnn layer outputs are:')
    CNN_model.layer_outputs(images_dir, model)
    

    
