#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 00:56:11 2022

@author: ericwei
"""

import hyperpara_tuning
import hyperpara_tuning_plot

def kernel_num_tuning(kernel_nums, X_train, y_train, X_val, y_val):

    histories = hyperpara_tuning.kernel_num_tuning(kernel_nums, X_train, y_train, X_val, y_val)
    
    accs = hyperpara_tuning_plot.training_validation_kernum_plot(histories, kernel_nums)
    acc_max = []
    for acc in accs:
        acc_max.append(max(acc))
    
    losses = hyperpara_tuning_plot.training_validation_kernum_loss_plot(histories, kernel_nums)
    loss_min = []
    for loss in losses:
        loss_min.append(min(loss))
        
    return acc_max, loss_min




def kernel_size_tuning(kernel_num, kernel_size, X_train, y_train, X_val, y_val):

    histories = hyperpara_tuning.kernel_size_tuning(kernel_num, kernel_size, X_train, y_train, X_val, y_val)
    
    accs = hyperpara_tuning_plot.training_validation_kersize_plot(histories, kernel_size)
    acc_max = []
    for acc in accs:
        acc_max.append(max(acc))
    
    losses = hyperpara_tuning_plot.training_validation_kersize_loss_plot(histories, kernel_size)
    loss_min = []
    for loss in losses:
        loss_min.append(min(loss))
    
    return acc_max, loss_min





def Maxpooling_size_tuning(kernel_num, kernel_size, maxpool_size, X_train, y_train, X_val, y_val):

    histories = hyperpara_tuning.maxpool_size_tuning(kernel_num, kernel_size, maxpool_size, X_train, y_train, X_val, y_val)
    
    accs = hyperpara_tuning_plot.training_validation_maxpoolsize_plot(histories, maxpool_size)
    acc_max = []
    for acc in accs:
        acc_max.append(max(acc))
    
    losses = hyperpara_tuning_plot.training_validation_maxpoolsize_loss_plot(histories, maxpool_size)
    loss_min = []
    for loss in losses:
        loss_min.append(min(loss))
    
    return acc_max, loss_min






def fc_size_tuning(kernel_num, kernel_size, maxpool_size, fc_size, X_train, y_train, X_val, y_val):

    histories = hyperpara_tuning.fc_size_tuning(kernel_num, kernel_size, maxpool_size, fc_size, X_train, y_train, X_val, y_val)
    
    accs = hyperpara_tuning_plot.training_validation_fcsize_plot(histories, fc_size)
    acc_max = []
    for acc in accs:
        acc_max.append(max(acc))
    
    losses = hyperpara_tuning_plot.training_validation_fcsize_loss_plot(histories, fc_size)
    loss_min = []
    for loss in losses:
      loss_min.append(min(loss))
  
    return acc_max, loss_min






def Gray_RGB_tuning(kernel_num, kernel_size, maxpool_size, fc_size, 
                    X_train, y_train, X_val, y_val, 
                    X_train_RGB, y_train_RGB, X_val_RGB, y_val_RGB):
    
    history = hyperpara_tuning.train_and_validate_Gray(kernel_num, kernel_size, maxpool_size, fc_size, X_train, y_train, X_val, y_val)
    history_RGB = hyperpara_tuning.train_and_validate_RGB(kernel_num, kernel_size, maxpool_size, fc_size, X_train_RGB, y_train_RGB, X_val_RGB, y_val_RGB)
    
    val_accs, val_losses = hyperpara_tuning_plot.training_validation_RGB_Gray_plot(history, history_RGB)
    
    return val_accs, val_losses


