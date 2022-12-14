#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:24:05 2022

@author: ericwei
"""

# from sklearn import svm
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
from matplotlib import pyplot as plt



def svm_model_search(X_train, y_train):
    
    rbf = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['rbf']} 
    poly = {'C': [0.1, 1, 10, 100], 'degree': [1, 2, 3, 4, 5],'kernel': ['poly']} 
    linear = {'C': [0.1, 1, 10, 100], 'kernel': ['linear']} 
    
    model_search = [rbf, poly, linear]
      
    ideal_model = GridSearchCV(SVC(), model_search, refit = True, verbose = 3)
      
    # fitting the model for parameter search
    ideal_model.fit(X_train, y_train)
    
    return ideal_model




def training_vs_cross_validation_score(X_train, y_train):
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    svc = SVC(kernel="poly", C = 10, degree = 2)
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