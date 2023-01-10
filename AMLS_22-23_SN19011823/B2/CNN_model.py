#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 21:26:39 2022

"""

'''
this module constructs a cnn model for multiclass classification, eye colours.
The model takes 30 x 30 x 3 image as input (RGB scale).
The output layers contains 5 neurons as there are in total 5 classes of eye colours
'''
from keras import layers, models
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os

def cnn_model(k_num, k_size, activation, p_size, fc_size):
    model = models.Sequential()
    model.add(layers.Conv2D(k_num, k_size, activation = activation, input_shape=(30 ,30, 3), padding="same"))
    model.add(layers.MaxPooling2D(p_size, padding="same"))
    
    model.add(layers.Conv2D(k_num, k_size, activation = activation, padding="same"))
    model.add(layers.MaxPooling2D(p_size, padding="same"))

    model.add(layers.Flatten())
    model.add(layers.Dense(units = fc_size, activation = activation))
    model.add(layers.Dense(units = fc_size, activation = activation))
    model.add(layers.Dense(units = 5, activation = 'softmax')) ## 5 neurons as there are in total 5 classes, use softmax output probabilities for multiclassed
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) ## 'categorical_crossentropy' as it is multiclass
    
    return model



'''
This part of the function references parts of the code in a blog Published in Towards Data Science
https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0
'''


def layer_outputs(images_dir, model):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    img = image.load_img(image_paths[90], target_size = (30, 30))
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
