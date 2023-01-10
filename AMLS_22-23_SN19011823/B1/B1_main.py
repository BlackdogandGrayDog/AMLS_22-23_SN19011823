#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 03:39:07 2022

"""
import B1.Preprocessing_feature_extraction as Preprocessing_feature_extraction
import B1.KNN_model as KNN_model
import B1.SVM_model as svm_model
import B1.Preprocessing as Preprocessing
import B1.hyperpara_tuning as hyperpara_tuning
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np


def feature_training_knn_svm(detector, predictor, basedir, labels_filename, images_dir):
    # Features Extraction Training
    print('\n')
    print('Training the model using feature extraction:')
    X_train, X_val, y_train, y_val = Preprocessing_feature_extraction.get_data(0.2, detector, predictor, basedir, labels_filename, images_dir)

    # KNN Training (Features Extraction)
    print('\n')
    print('Training via KNN model:')
    acc = KNN_model.KNN_model(X_train, y_train, X_val, y_val)
    print('\n')
    print('The maximum accuracy KNN can reach is:')
    print(max(acc))
    print('\n')
    acc = np.array(acc)
    posi = np.where(acc == max(acc))
    posi = (np.array(posi) * 5) + 1 ## since the line space of number of k is 5, therefore position need to be times by 5, 1 becasue k starts from 1
    print('With the number of neighbours of:')
    print(posi)
    print('The accuracy vs. Number of neighbours plot is:')
    KNN_model.KNN_model_plot(acc)

    # SVM Training (Features Extraction)
    print('\n')
    print('Training via SVM model:')
    pred, acc_score_val, conf_matrix_val, pred_prob = svm_model.img_SVM(X_train, y_train, X_val, y_val)
    print('The maximum validation accuracy SVM can reach is:')
    print(acc_score_val)
    print('\n')
    print('The corresponding learning curve plot is:')
    hyperpara_tuning.training_vs_cross_validation_score(X_train, y_train)
    
    
def model_tuning(basedir, images_dir, labels_filename):
    train_img = Preprocessing.image_processing(basedir, images_dir)
    train_labels =  Preprocessing.extract_labels(basedir, labels_filename, images_dir)
    train_img = train_img.reshape((train_img.shape[0], train_img.shape[1] * train_img.shape[2] * train_img.shape[3]))
    X_train, X_val, y_train, y_val = train_test_split(train_img, train_labels, test_size = 0.2, shuffle = True)
    
    print('\n')
    print('Model tuning starts:')
    print('Caution: Time for executing is long.')
    model = hyperpara_tuning.svm_model_search(X_train, y_train)
    print('\n')
    print('Model selected is:')
    print(model.best_estimator_)
    print('\n')
    print('The corresponding learning curve is:')
    print('Caution: Time for executing is long.')
    hyperpara_tuning.training_vs_cross_validation_score(X_train, y_train)



def train_and_test_knn_svm(basedir, images_dir, labels_filename, testdir, test_images_dir):
    train_img = Preprocessing.image_processing(basedir, images_dir)
    train_labels =  Preprocessing.extract_labels(basedir, labels_filename, images_dir)
    train_img = train_img.reshape((train_img.shape[0], train_img.shape[1] * train_img.shape[2] * train_img.shape[3]))
    X_train, X_val, y_train, y_val = train_test_split(train_img, train_labels, test_size = 0.2, shuffle = True)
    
    test_img = Preprocessing.image_processing(testdir, test_images_dir)
    test_labels =  Preprocessing.extract_labels(testdir, labels_filename, test_images_dir)
    test_img = test_img.reshape((test_img.shape[0], test_img.shape[1] * test_img.shape[2] * test_img.shape[3]))
    
    
    print('Training via KNN model:')
    acc = KNN_model.KNN_model(X_train, y_train, X_val, y_val)
    print('\n')
    print('The maximum accuracy KNN can reach is:')
    print(max(acc))
    acc = np.array(acc)
    posi = np.where(acc == max(acc))
    posi = (np.array(posi) * 5) + 1  ## since the line space of number of k is 5, therefore position need to be times by 5, 1 becasue k starts from 1
    print('With the number of neighbours of:')
    print(posi)
    print('\n')
    print('The accuracy vs. Number of neighbours plot is:')
    KNN_model.KNN_model_plot(acc)
    
    print('\n')
    print('Training via SVM model:')
    print('\n')
    print('Test the model using training dataset:')
    pred, acc_score_train, conf_matrix_train, pred_prob = svm_model.img_SVM(X_train, y_train, X_train, y_train)
    train_loss = log_loss(y_true = y_train, y_pred = pred_prob)
    print('the accuracy score and loss for training are:')
    print(acc_score_train, train_loss)
    print('\n')
    print('the confusion matrix for training is:')
    svm_model.confusion_matrix_plot(conf_matrix_train, 'Training ')
    
    print('\n')
    print('Test the model using validation dataset:')
    pred, acc_score_val, conf_matrix_val, pred_prob = svm_model.img_SVM(X_train, y_train, X_val, y_val)
    val_loss = log_loss(y_true = y_val, y_pred = pred_prob)
    print('the accuracy score and loss for validation are:')
    print(acc_score_val, val_loss)
    print('\n')
    print('the confusion matrix for validation is:')
    svm_model.confusion_matrix_plot(conf_matrix_val, 'Validation ')
    
    print('\n')
    print('Test the model using test dataset:')
    pred, acc_score_test, conf_matrix_test, pred_prob = svm_model.img_SVM(train_img, train_labels, test_img, test_labels)
    test_loss = log_loss(y_true = test_labels, y_pred = pred_prob)
    print('\n')
    print('the accuracy score and loss for testing are:')
    print(acc_score_test, test_loss)
    print('the confusion matrix for test is:')
    svm_model.confusion_matrix_plot(conf_matrix_test, 'Test ')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
