#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 18:51:02 2022

@author: ericwei
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

def KNN_model(X_train, y_train, X_val, y_val):
    KNN_accuracy = []
    for k in range(1,5001, 5):
        knn_clf = KNeighborsClassifier(n_neighbors = k)
        knn_clf.fit(X_train,y_train)
        ypred = knn_clf.predict(X_val)
        score = accuracy_score(y_val, ypred)
        KNN_accuracy.append(score)
    
    return KNN_accuracy



def KNN_model_plot(KNN_accuracy):
     plt.figure(figsize=(15,7.5))
     plt.suptitle('KNN Model Accuracy',fontsize = 20, fontweight='bold')
     plt.subplot(1,2,1)
     plt.plot(range(1,5001, 5), KNN_accuracy)
     plt.title('0 to 5000 neighbours',fontsize = 20, fontweight='bold')
     plt.grid()
     plt.xticks(fontsize = 15, fontweight='bold')
     plt.yticks(fontsize = 15, fontweight='bold')
     plt.xlabel('Number of Neighbours', fontsize = 20, fontweight='bold')
     plt.ylabel('Accuracy Score', fontsize = 20, fontweight='bold')
     plt.subplots_adjust(wspace=0.3,hspace=0.5)
     
     plt.subplot(1,2,2)
     plt.plot(range(1,1001, 5), KNN_accuracy[0:200])
     plt.title('Zoom in 0 to 1000 Neighbours',fontsize = 20, fontweight='bold')
     plt.grid()
     plt.xticks(fontsize = 15, fontweight='bold')
     plt.yticks(fontsize = 15, fontweight='bold')
     plt.xlabel('Number of Neighbours', fontsize = 20, fontweight='bold')
     plt.ylabel('Accuracy Score', fontsize = 20, fontweight='bold')