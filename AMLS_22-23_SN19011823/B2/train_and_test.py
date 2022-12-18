#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 04:19:30 2022

@author: ericwei
"""

import CNN_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns


def train_and_test(train_gen, train_step_size, test_gen, test_step_size):
    model = CNN_model.cnn_model(16, (7,7), 'relu', (2,2), 512)
    history = model.fit_generator(generator = train_gen,
                                  steps_per_epoch = train_step_size,
                                  epochs = 13,
                                  validation_data = test_gen,
                                  validation_steps = test_step_size)
    results = model.evaluate_generator(test_gen)
    train_and_test_plot(history)
    
    return history, results, model


def train_and_test_plot(history):
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
    plt.xticks(range(0,13,1), rotation=45,fontsize = 18)
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
    plt.xticks(range(0,13,1), rotation=45,fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.grid('minor')
    plt.legend()



def conf_matrix(model, test_gen, test_step_size):
    
    pred = model.predict_generator(test_gen, steps = test_step_size)
    pred = np.argmax(pred, axis = 1)
    conf_matrix = confusion_matrix(test_gen.classes, pred)
    plt.figure(figsize=(7.5, 7.5))
    sns.heatmap(conf_matrix, annot=True, fmt = 'g')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Test Confusion Matrix', fontsize=18)
    plt.show()
    return conf_matrix