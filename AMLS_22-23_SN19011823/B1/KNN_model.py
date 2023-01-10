#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 00:40:12 2022

"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

'''
This model is used for constructing KNN model and corresponding plot function
'''
def KNN_model(X_train, y_train, X_val, y_val):
    accuracy_s = []
    for k in range(1,5001, 5):  ## number of neighbours varys from 1 to 4996
        knn_clf = KNeighborsClassifier(n_neighbors = k) ## build KNN model with k number of neighbours
        knn_clf.fit(X_train,y_train) ## fit with training data
        ypred = knn_clf.predict(X_val)
        score = accuracy_score(y_val, ypred)
        accuracy_s.append(score)
    
    return accuracy_s


def KNN_model_plot(accuracy_s): ## used for plotting accuracy score versus neighbour numbers
     plt.figure(figsize=(10,10))
     plt.plot(range(1,5001, 5), accuracy_s)
     plt.title('KNN Model Accuracy',fontsize = 20, fontweight='bold')
     plt.grid()
     plt.xticks(fontsize = 15, fontweight='bold')
     plt.yticks(fontsize = 15, fontweight='bold')
     plt.xlabel('Number of Neighbours', fontsize = 20, fontweight='bold')
     plt.ylabel('Accuracy Score', fontsize = 20, fontweight='bold')
