#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 21:26:39 2022

@author: ericwei
"""

from keras import layers, models

def cnn_model(k_num, k_size, activation, p_size, fc_size):
    model = models.Sequential()
    model.add(layers.Conv2D(k_num, k_size, activation = activation, input_shape=(30 ,30, 3), padding="same"))
   # model.add(layers.Conv2D(8, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(p_size, padding="same"))
   # model.add(Dropout(d_value))
    
    model.add(layers.Conv2D(k_num, k_size, activation = activation, padding="same"))
    model.add(layers.MaxPooling2D(p_size, padding="same"))
   # model.add(Dropout(d_value))

    model.add(layers.Flatten())
    model.add(layers.Dense(units = fc_size, activation = activation))
    model.add(layers.Dense(units = fc_size, activation = activation))
   # model.add(Dropout(d_value))
    model.add(layers.Dense(units = 5, activation = 'softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    return model