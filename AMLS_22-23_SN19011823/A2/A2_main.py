#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 02:38:46 2022

"""

import A2.Preprocessing as Preprocessing
import A2.hyperpara_tuning as hyperpara_tuning
import A2.svm_model as svm_model
from sklearn.metrics import log_loss


'''
this module contains two major parts, tuning and training/testing
the training part we also compare using both mouth or face feaures via loss and accuracy score
'''

def model_tuning(detector, predictor, basedir, labels_filename, images_dir):
    X_train, X_val, y_train, y_val =  Preprocessing.get_data('face', 0.2, detector, predictor, basedir, labels_filename, images_dir)
    print('\n')
    print('Model tuning starts:')
    model = hyperpara_tuning.svm_model_search(X_train, y_train)
    print('\n')
    print('Model selected is:')
    print(model.best_estimator_)
    print('\n')
    print('The corresponding learning curve is:')
    hyperpara_tuning.training_vs_cross_validation_score(X_train, y_train)
    

def train_and_tesing_A2(detector, predictor, basedir, labels_filename, images_dir, testdir, test_images_dir):
    print('\n')
    print('Compareing using mouth and face features:')
    print('Using mouth features first:')
    X_train, X_val, y_train, y_val =  Preprocessing.get_data('mouth', 0.2, detector, predictor, basedir, labels_filename, images_dir) ## obtain only mouth feaure and train
    pred, acc_score_train, conf_matrix_train = svm_model.img_SVM(X_train, y_train, X_train, y_train)  ## returns accuracy score
    train_loss = log_loss(y_true = y_train, y_pred = pred) ## returns training loss via log loss from sklearn library
    print('\n')
    print('the accuracy score and loss for using mouth features are:')
    print(acc_score_train, train_loss)
    
    print('\n')
    print('Compareing using mouth and face features:')
    print('Using face features then:')
    X_train, X_val, y_train, y_val =  Preprocessing.get_data('face', 0.2, detector, predictor, basedir, labels_filename, images_dir)  ## obtain face feaure and train
    pred, acc_score_train, conf_matrix_train = svm_model.img_SVM(X_train, y_train, X_train, y_train) ## returns accuracy score
    train_loss = log_loss(y_true = y_train, y_pred = pred) ## returns training loss via log loss from sklearn library
    print('\n')
    print('the accuracy score and loss for using face features are:')
    print(acc_score_train, train_loss)
    
    print('\n')
    print('the selected feature is face feature')
    print('\n')
    print('the confusion matrix for training is:')
    svm_model.confusion_matrix_plot(conf_matrix_train, 'Train ')  ## ploting confusion matrix
    
    
    
    print('\n')
    print('Test the model using validation dataset:')
    pred, acc_score_val, conf_matrix_val = svm_model.img_SVM(X_train, y_train, X_val, y_val)  ## test the model using validation set returns accuracy score and loss
    val_loss = log_loss(y_true = y_val, y_pred = pred) # gives a validation loss
    print('\n')
    print('the accuracy score and loss for validation are:')
    print(acc_score_val, val_loss)
    print('\n')
    print('the confusion matrix for validation is:')
    svm_model.confusion_matrix_plot(conf_matrix_val, 'Validation ')
    
    
    X_test, y_test =  Preprocessing.get_test_data('face', detector, predictor, testdir, labels_filename, test_images_dir)
    X_train, y_train =  Preprocessing.get_test_data('face', detector, predictor, basedir, labels_filename, images_dir)
    print('\n')
    print('Test the model using test dataset:')
    pred, acc_score_test, conf_matrix_test = svm_model.img_SVM(X_train, y_train, X_test, y_test)  ## test the model using validation set returns accuracy score and loss
    test_loss = log_loss(y_true = y_test, y_pred = pred) # gives a validation loss
    print('\n')
    print('the accuracy score and loss for testing are:')
    print(acc_score_test, test_loss)
    print('the confusion matrix for test is:')
    svm_model.confusion_matrix_plot(conf_matrix_test, 'Test ')
    
    
    
    
    
    
