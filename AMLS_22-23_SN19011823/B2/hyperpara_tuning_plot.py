#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 01:10:58 2022

@author: ericwei
"""

from matplotlib import pyplot as plt

def training_validation_kernum_plot(histories, kernel_nums):

    fig = plt.figure(figsize=(15,7.5))
    fig.suptitle('Training and validation accuracy', fontsize = 20, fontweight='bold')
    accs =[]
    for i, history in enumerate(histories):
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.7)
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('kernel number =' + str(kernel_nums[i]), fontsize = 15, fontweight='bold')
        plt.xlabel('Epochs', fontsize = 20, fontweight='bold')
        plt.ylabel('Accuracy', fontsize = 20, fontweight='bold')
        plt.xticks(range(0,21,2), rotation=45,fontsize = 18)
        plt.yticks(fontsize = 15)
        plt.grid('minor')
        plt.legend()
        accs.append(val_acc)
        
    plt.show()
    return accs


def training_validation_kernum_loss_plot(histories, kernel_num):

    fig = plt.figure(figsize=(15,7.5))
    fig.suptitle('Training and validation loss', fontsize = 20, fontweight='bold')
    losses =[]
    for i, history in enumerate(histories):
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.7)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('kernel number =' + str(kernel_num[i]), fontsize = 15, fontweight='bold')
        plt.xlabel('Epochs', fontsize = 20, fontweight='bold')
        plt.ylabel('Loss', fontsize = 20, fontweight='bold')
        plt.xticks(range(0,21,2), rotation=45,fontsize = 18)
        plt.yticks(fontsize = 15)
        plt.grid('minor')
        plt.legend()
        losses.append(val_loss)
        
    plt.show()
    return losses







def training_validation_kersize_plot(histories, kernel_size):

    fig = plt.figure(figsize=(15,7.5))
    fig.suptitle('Training and validation accuracy', fontsize = 20, fontweight='bold')
    accs =[]
    for i, history in enumerate(histories):
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.7)
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('kernel size =' + str(kernel_size[i]), fontsize = 15, fontweight='bold')
        plt.xlabel('Epochs', fontsize = 20, fontweight='bold')
        plt.ylabel('Accuracy', fontsize = 20, fontweight='bold')
        plt.xticks(range(0,21,2), rotation=45,fontsize = 18)
        plt.yticks(fontsize = 15)
        plt.grid('minor')
        plt.legend()
        accs.append(val_acc)
        
    plt.show()
    return accs


def training_validation_kersize_loss_plot(histories, kernel_size):

    fig = plt.figure(figsize=(15,7.5))
    fig.suptitle('Training and validation loss', fontsize = 20, fontweight='bold')
    losses =[]
    for i, history in enumerate(histories):
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.7)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('kernel size =' + str(kernel_size[i]), fontsize = 15, fontweight='bold')
        plt.xlabel('Epochs', fontsize = 20, fontweight='bold')
        plt.ylabel('Loss', fontsize = 20, fontweight='bold')
        plt.xticks(range(0,21,2), rotation=45,fontsize = 18)
        plt.yticks(fontsize = 15)
        plt.grid('minor')
        plt.legend()
        losses.append(val_loss)
        
    plt.show()
    return losses













def training_validation_maxpoolsize_plot(histories, maxpool_size):

    fig = plt.figure(figsize=(18,5))
    fig.suptitle('Training and validation accuracy', fontsize = 15, fontweight='bold')
    accs =[]
    for i, history in enumerate(histories):
        plt.subplot(1,3,i+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.7)
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('MaxPooling size =' + str(maxpool_size[i]), fontsize = 15, fontweight='bold')
        plt.xlabel('Epochs', fontsize = 20, fontweight='bold')
        plt.ylabel('Accuracy', fontsize = 20, fontweight='bold')
        plt.xticks(range(0,21,2), rotation=45,fontsize = 18)
        plt.yticks(fontsize = 15)
        plt.grid('minor')
        plt.legend()
        accs.append(val_acc)
        
    plt.show()
    return accs


def training_validation_maxpoolsize_loss_plot(histories, maxpool_size):

    fig = plt.figure(figsize=(18,5))
    fig.suptitle('Training and validation loss', fontsize = 15, fontweight='bold')
    losses =[]
    for i, history in enumerate(histories):
        plt.subplot(1,3,i+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.7)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('MaxPooling size =' + str(maxpool_size[i]), fontsize = 15, fontweight='bold')
        plt.xlabel('Epochs', fontsize = 20, fontweight='bold')
        plt.ylabel('Loss', fontsize = 20, fontweight='bold')
        plt.xticks(range(0,21,2), rotation=45,fontsize = 18)
        plt.yticks(fontsize = 15)
        plt.grid('minor')
        plt.legend()
        losses.append(val_loss)
        
    plt.show()
    return losses







def training_validation_fcsize_plot(histories, fc_size):

    fig = plt.figure(figsize=(15,7.5))
    fig.suptitle('Training and validation accuracy', fontsize = 15, fontweight='bold')
    accs =[]
    for i, history in enumerate(histories):
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.7)
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Fully connected layer size =' + str(fc_size[i]), fontsize = 15, fontweight='bold')
        plt.xlabel('Epochs', fontsize = 20, fontweight='bold')
        plt.ylabel('Accuracy', fontsize = 20, fontweight='bold')
        plt.xticks(range(0,21,2), rotation=45,fontsize = 18)
        plt.yticks(fontsize = 15)
        plt.grid('minor')
        plt.legend()
        accs.append(val_acc)
        
    plt.show()
    return accs


def training_validation_fcsize_loss_plot(histories, fc_size):

    fig = plt.figure(figsize=(15,7.5))
    fig.suptitle('Training and validation loss', fontsize = 15, fontweight='bold')
    losses =[]
    for i, history in enumerate(histories):
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.2,hspace=0.7)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Fully connected layer size =' + str(fc_size[i]), fontsize = 15, fontweight='bold')
        plt.xlabel('Epochs', fontsize = 20, fontweight='bold')
        plt.ylabel('Loss', fontsize = 20, fontweight='bold')
        plt.xticks(range(0,21,2), rotation=45,fontsize = 18)
        plt.yticks(fontsize = 15)
        plt.grid('minor')
        plt.legend()
        losses.append(val_loss)
        
    plt.show()
    return losses







