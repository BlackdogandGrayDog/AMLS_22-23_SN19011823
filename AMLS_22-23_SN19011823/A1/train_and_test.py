#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 02:05:18 2022

@author: ericwei
"""

import A1.cnn_model as cnn_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

'''
This module is used for final training. It utilise whole training dataset and test dataset
It includes training and testing function, and corresponding plotting, and constructing a confusion matrix
'''

def train_and_test(X_train, y_train, X_test, y_test):  ## this function is used for training via training dataset and testing using test dataset
    model = cnn_model.cnn_model_RGB(16, (5,5), 'relu', (2,2), 512)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
    history = model.fit(X_train, y_train, batch_size=128, epochs = 6, validation_data=(X_test, y_test), shuffle = True)
    results = model.evaluate(X_test, y_test)
    train_and_test_plot(history)
    
    return history, results, model  ## return training and testing info for image plotting and model saving


def train_and_test_plot(history):   ## this function utilise the previous one's output and plot training and testing accuracy and loss and is embeded in the previous function
    fig = plt.figure(figsize=(15,7.5))
    fig.suptitle('Training and Testing accuracy and loss', fontsize = 15, fontweight='bold')
    
    plt.subplot(1,2,1)
    plt.subplots_adjust(wspace=0.2,hspace=0.7)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Testing acc')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Epochs', fontsize = 20, fontweight='bold')
    plt.ylabel('Accuracy', fontsize = 20, fontweight='bold')
    plt.xticks(range(0,7,1), rotation=45,fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.grid('minor')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.subplots_adjust(wspace=0.2,hspace=0.7)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Testing loss')
    plt.title('Train and Test Loss')
    plt.xlabel('Epochs', fontsize = 20, fontweight='bold')
    plt.ylabel('Loss', fontsize = 20, fontweight='bold')
    plt.xticks(range(0,7,1), rotation=45,fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.grid('minor')
    plt.legend()



def conf_matrix(model, X_test, y_test):  ## this function is used for plotting a confusion matrix, takes both x_test and y_test as input. For predictiton and compare
    
    pred = model.predict(X_test)
    pred[np.where(pred < 0.5)] = 0  ## since we used sigmode which outputs some probablity, so we need to convert it to two categories.
    pred[np.where(pred > 0.5)] = 1  ## where probability over 0.5 classify to 1, while smaller than 0.5 to 0

    conf_matrix = confusion_matrix(y_test, pred)
    plt.figure(figsize=(7.5, 7.5))
    sns.heatmap(conf_matrix, annot=True, fmt = 'g')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Test Confusion Matrix', fontsize=18)
    plt.show()

