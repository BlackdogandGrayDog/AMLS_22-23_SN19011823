#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 02:11:51 2022

@author: ericwei
"""

from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot as plt

from sklearn.model_selection import GridSearchCV


def svm_model_search(X_train, y_train):
    
    rbf = {'C': [1, 10], 'gamma': [0.0001, 0.001],'kernel': ['rbf']} 

    poly = {'C': [0.1, 10], 'degree': [2, 4],'kernel': ['poly']} 

    linear = {'C': [0.1, 1], 'kernel': ['linear']} 
    
    model_search = [rbf, poly, linear]
      
    ideal_model = GridSearchCV(SVC(), model_search, refit = True, verbose = 3)
      
    # fitting the model for parameter search
    ideal_model.fit(X_train, y_train)
    
    return ideal_model


def training_vs_cross_validation_score(X_train, y_train):
    '''
    this method used for ploting learning curve, which is traing and cross-validation (cv) score versus training samples.
    '''
    cv = ShuffleSplit(n_splits = 100, test_size=0.2, random_state=0) ## 100 iterations used for cv, and 20% of data used for validation seleted randomly
    svc = SVC(kernel="linear", C = 0.1)
    '''
    use learning_curve from sklearn library to generate train_sizes, train_scores, and test scores for the subsequent calculation
    '''
    train_sizes, train_scores, test_scores = learning_curve(svc, X_train, y_train, cv = cv, n_jobs = -1, train_sizes = np.linspace(.1, 1.0, 5))
    
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std= np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)
    
    plt.figure(figsize=(15, 10))
    plt.title('Training vs. Cross-validation Score', fontsize = 20, fontweight='bold')
    plt.xlabel("Training examples", fontsize = 20, fontweight='bold')
    plt.ylabel("Score", fontsize = 20, fontweight='bold')
    plt.grid()
    plt.ylim(0.70, 1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    
    plt.legend(loc="best", fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
