#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 01:12:43 2022

@author: ericwei
"""

import cnn_model

def kernel_num_tuning(kernel_nums, X_train, y_train, X_val, y_val):
    histories = []
    for kernel_num in kernel_nums:
        model = cnn_model.cnn_model(kernel_num, (3,3), 'relu', (2,2), 512)
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
        history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_val, y_val), shuffle = True)
        histories.append(history)
        
    return histories



def kernel_size_tuning(kernel_num, kernel_sizes, X_train, y_train, X_val, y_val):
    histories = []
    for kernel_size in kernel_sizes:
        model = cnn_model.cnn_model(kernel_num, kernel_size, 'relu', (2,2), 512)
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
        history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_val, y_val), shuffle = True)
        histories.append(history)
        
    return histories


def maxpool_size_tuning(kernel_num, kernel_sizes, maxpool_sizes, X_train, y_train, X_val, y_val):
    histories = []
    for maxpool_size in maxpool_sizes:
        model = cnn_model.cnn_model(kernel_num, kernel_sizes, 'relu', maxpool_size, 512)
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
        history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_val, y_val), shuffle = True)
        histories.append(history)
        
    return histories


def fc_size_tuning(kernel_num, kernel_sizes, maxpool_size, fc_sizes, X_train, y_train, X_val, y_val):
    histories = []
    for fc_size in fc_sizes:
        model = cnn_model.cnn_model(kernel_num, kernel_sizes, 'relu', maxpool_size, fc_size)
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
        history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_val, y_val), shuffle = True)
        histories.append(history)
        
    return histories


def train_and_validate_Gray(kernel_num, kernel_size, maxpool_size, fc_size, X_train, y_train, X_val, y_val):
    model = cnn_model.cnn_model(kernel_num, kernel_size, 'relu', maxpool_size, fc_size)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
    history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_val, y_val), shuffle = True)
    
    return history
    
    
def train_and_validate_RGB(kernel_num, kernel_size, maxpool_size, fc_size, X_train_RGB, y_train_RGB, X_val_RGB, y_val_RGB):
    model = cnn_model.cnn_model_RGB(kernel_num, kernel_size, 'relu', maxpool_size, fc_size)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
    history = model.fit(X_train_RGB, y_train_RGB, batch_size=128, epochs=20, validation_data=(X_val_RGB, y_val_RGB), shuffle = True)
    
    return history
