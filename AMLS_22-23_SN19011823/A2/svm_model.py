#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 03:33:07 2022

@author: ericwei
"""
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

def img_SVM(training_images, training_labels, test_images, test_labels):
    
    classifier = svm.SVC(kernel='poly', C = 10,  degree = 2)

    classifier.fit(training_images, training_labels)

    pred = classifier.predict(test_images)

    return pred, accuracy_score(test_labels, pred), confusion_matrix(test_labels, pred)


def confusion_matrix_plot(conf_matrix, name):
    plt.figure(figsize=(7.5, 7.5))
    sns.heatmap(conf_matrix, annot=True, fmt = 'g')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(name + 'Confusion Matrix', fontsize=18)
    plt.show()


