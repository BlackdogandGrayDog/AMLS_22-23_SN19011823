#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 03:26:13 2022

"""


from keras import layers, models
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from matplotlib import pyplot as plt

'''
this module contains two identical cnn models, except input shape is different as one is for gray scale and the other one is for RGB scale.
Also the 'layer output' function gives output of filters from each layer
'''

def cnn_model(k_num, k_size, activation, p_size, fc_size): ## CNN model for gray scale
    model = models.Sequential()
    
    model.add(layers.Conv2D(k_num, k_size, activation = activation, input_shape=(80 ,80, 1)))
    model.add(layers.MaxPooling2D(p_size))
    
    model.add(layers.Conv2D(k_num, k_size, activation = activation))
    model.add(layers.MaxPooling2D(p_size))

    model.add(layers.Flatten())
    model.add(layers.Dense(units = fc_size, activation = activation))
    model.add(layers.Dense(units = fc_size, activation = activation))
    model.add(layers.Dense(units = 1, activation = 'sigmoid')) ## As it is binary classification, activation is sigmoid and output unit is 1
    
    return model


def cnn_model_RGB(k_num, k_size, activation, p_size, fc_size): ## cnn model for RGB scale
    model = models.Sequential()
    model.add(layers.Conv2D(k_num, k_size, activation = activation, input_shape=(80 ,80, 3)))
    model.add(layers.MaxPooling2D(p_size))
    
    model.add(layers.Conv2D(k_num, k_size, activation = activation))
    model.add(layers.MaxPooling2D(p_size))

    model.add(layers.Flatten())
    model.add(layers.Dense(units = fc_size, activation = activation))
    model.add(layers.Dense(units = fc_size, activation = activation))
    model.add(layers.Dense(units = 1, activation = 'sigmoid')) ## As it is binary classification, activation is sigmoid and output unit is 1
    
    return model




'''
This part of the function references parts of the code in a blog Published in Towards Data Science
https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0
'''


def layer_outputs(images_dir, model):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    img = image.load_img(image_paths[90], target_size = (80, 80))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = img / 255.

    layer_outputs = [layer.output for layer in model.layers[:4]]
    activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
    activations = activation_model.predict(img)
      
    for j in range(4):
        first_layer_activation = activations[j]
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('CNN Layer ' + str(j+1) + ' output', fontsize = 15, fontweight='bold')
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.subplots_adjust(wspace=0.2,hspace=0.5)
            plt.imshow(first_layer_activation[0, :, :, i])

