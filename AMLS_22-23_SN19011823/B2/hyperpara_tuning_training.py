#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 01:11:58 2022

@author: ericwei
"""
'''
This module is simply combine hyperpara_tuning and plotting into one function and gives the max accuracy and min loss each choice can achive

'''
import B2.hyperpara_tuning_model as hyperpara_tuning_model
import B2.hyperpara_tuning_plot as hyperpara_tuning_plot

def kernel_num_tuning(kernel_nums, train_gen, train_step_size, val_gen, val_step_size): ## kernel number tuning

    histories = hyperpara_tuning_model.kernel_num_tuning(kernel_nums, train_gen, train_step_size, val_gen, val_step_size)
    
    accs = hyperpara_tuning_plot.training_validation_kernum_plot(histories, kernel_nums)
    acc_max = []
    for acc in accs:
        acc_max.append(max(acc))
    
    losses = hyperpara_tuning_plot.training_validation_kernum_loss_plot(histories, kernel_nums)
    loss_min = []
    for loss in losses:
        loss_min.append(min(loss))
        
    return acc_max, loss_min




def kernel_size_tuning(kernel_num, kernel_size, train_gen, train_step_size, val_gen, val_step_size): ## kernel size tuning

    histories = hyperpara_tuning_model.kernel_size_tuning(kernel_num, kernel_size, train_gen, train_step_size, val_gen, val_step_size)
    
    accs = hyperpara_tuning_plot.training_validation_kersize_plot(histories, kernel_size)
    acc_max = []
    for acc in accs:
        acc_max.append(max(acc))
    
    losses = hyperpara_tuning_plot.training_validation_kersize_loss_plot(histories, kernel_size)
    loss_min = []
    for loss in losses:
        loss_min.append(min(loss))
    
    return acc_max, loss_min





def Maxpooling_size_tuning(kernel_num, kernel_size, maxpool_size, train_gen, train_step_size, val_gen, val_step_size): ## maxpooling size tuning

    histories = hyperpara_tuning_model.maxpool_size_tuning(kernel_num, kernel_size, maxpool_size, train_gen, train_step_size, val_gen, val_step_size)
    
    accs = hyperpara_tuning_plot.training_validation_maxpoolsize_plot(histories, maxpool_size)
    acc_max = []
    for acc in accs:
        acc_max.append(max(acc))
    
    losses = hyperpara_tuning_plot.training_validation_maxpoolsize_loss_plot(histories, maxpool_size)
    loss_min = []
    for loss in losses:
        loss_min.append(min(loss))
    
    return acc_max, loss_min


def fc_size_tuning(kernel_num, kernel_size, maxpool_size, fc_size, train_gen, train_step_size, val_gen, val_step_size): ## fully connected layer size tuning

    histories = hyperpara_tuning_model.fc_size_tuning(kernel_num, kernel_size, maxpool_size, fc_size, train_gen, train_step_size, val_gen, val_step_size)
    
    accs = hyperpara_tuning_plot.training_validation_fcsize_plot(histories, fc_size)
    acc_max = []
    for acc in accs:
        acc_max.append(max(acc))
    
    losses = hyperpara_tuning_plot.training_validation_fcsize_loss_plot(histories, fc_size)
    loss_min = []
    for loss in losses:
      loss_min.append(min(loss))
  
    return acc_max, loss_min