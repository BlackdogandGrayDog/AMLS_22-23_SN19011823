#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 01:01:36 2022

@author: ericwei
"""

import CNN_model

def kernel_num_tuning(kernel_nums, train_gen, train_step_size, val_gen, val_step_size):
    histories = []
    for kernel_num in kernel_nums:
        model = CNN_model.cnn_model(kernel_num, (5,5), 'relu', (2,2), 512)
        history = model.fit_generator(generator = train_gen,
                                      steps_per_epoch = train_step_size,
                                      epochs=20,
                                      validation_data = val_gen,
                                      validation_steps = val_step_size)
        histories.append(history)
        
    return histories



def kernel_size_tuning(kernel_num, kernel_sizes, train_gen, train_step_size, val_gen, val_step_size):
    histories = []
    for kernel_size in kernel_sizes:
        model = CNN_model.cnn_model(kernel_num, kernel_size, 'relu', (2,2), 512)
        history = model.fit_generator(generator = train_gen,
                                      steps_per_epoch = train_step_size,
                                      epochs=20,
                                      validation_data = val_gen,
                                      validation_steps = val_step_size)
        histories.append(history)
        
    return histories


def maxpool_size_tuning(kernel_num, kernel_sizes, maxpool_sizes, train_gen, train_step_size, val_gen, val_step_size):
    histories = []
    for maxpool_size in maxpool_sizes:
        model = CNN_model.cnn_model(kernel_num, kernel_sizes, 'relu', maxpool_size, 512)
        history = model.fit_generator(generator = train_gen,
                                      steps_per_epoch = train_step_size,
                                      epochs=20,
                                      validation_data = val_gen,
                                      validation_steps = val_step_size)
        histories.append(history)
        
    return histories


def fc_size_tuning(kernel_num, kernel_sizes, maxpool_size, fc_sizes, train_gen, train_step_size, val_gen, val_step_size):
    histories = []
    for fc_size in fc_sizes:
        model = CNN_model.cnn_model(kernel_num, kernel_sizes, 'relu', maxpool_size, fc_size)
        history = model.fit_generator(generator = train_gen,
                                      steps_per_epoch = train_step_size,
                                      epochs=20,
                                      validation_data = val_gen,
                                      validation_steps = val_step_size)
        histories.append(history)
        
    return histories
