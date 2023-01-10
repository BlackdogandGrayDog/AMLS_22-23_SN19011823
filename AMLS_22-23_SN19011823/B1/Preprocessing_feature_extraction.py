#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 02:51:31 2022

"""


# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.

import os
import numpy as np
from keras_preprocessing import image
import cv2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from numpy import random
#%%

'''
this model is similar to the one in the Task A2, extracting the facial features in cartoon dataset.
Detailed reference can be found on above reference.
'''

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords



def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image, detector, predictor):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout[1: 17,:], resized_image




def extract_features_labels(detector, predictor, basedir, labels_filename, images_dir):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        faceShape_labels:      an array containing the faceShape label for each image in which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    faceshape_labels =  {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    all_features = []
    all_labels = []
    null_imgs = []

    for img_path in image_paths:
        file_name= img_path.split('.')[-2].split('/')[-1]
        # load image
        img = image.load_img(img_path, target_size=target_size, interpolation='bicubic')
        img = image.img_to_array(img)
        features, _ = run_dlib_shape(img, detector, predictor)
        if features is not None:
            all_features.append(features)
            all_labels.append(faceshape_labels[file_name])
                
        if features is None:
            null_imgs.append(img_path)

    landmark_features = np.array(all_features)
    faceshape_labels = all_labels # simply converts the -1 into 0, so male=0 and female=1

    return landmark_features, faceshape_labels, null_imgs




def get_data(test_size, detector, predictor, basedir, labels_filename, images_dir):
    
    X, y, _ = extract_features_labels(detector, predictor, basedir, labels_filename, images_dir)
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = test_size, shuffle = True)

    return X_train, X_val, y_train, y_val


'''
As in task A2, this function plots the original image and the 68 points landmark on cartoon dataset face.
Also, null image which cannot be extracted any feature are also being plotted via the second function.
'''
def train_image_plotting(images_dir, num_image, detector, predictor):
    
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    
    plt.figure(figsize=(15,7))
    plt.suptitle('Facial Feature Extraction', fontsize = 20, fontweight='bold')
    img = cv2.imread(image_paths[num_image], flags = 1)
    img = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(1,2,1)
    # plt.subplots_adjust(wspace=0.1,hspace=0.1)
    plt.imshow(img)
    plt.title('Original Image',fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    
    
    plt.subplot(1,2,2)
    # plt.subplots_adjust(wspace=0.1,hspace=0.1)
    plt.imshow(img)
    img = image.img_to_array(
        image.load_img(image_paths[num_image],
                       target_size=None,
                       interpolation='bicubic'))
    features, _ = run_dlib_shape(img, detector, predictor)
    plt.scatter(features[:,0], features[:,1], c = 'r', s = 8)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.title('Face Feature Extraction',fontsize = 15)
    
    


def null_image_plot(null_images):
    image_example = random.randint(1807, size=(4))
    fig = plt.figure(figsize=(10,9))
    fig.suptitle('Null Images eliminated from training', fontsize = 20, fontweight='bold')
    for i, image_num in enumerate(image_example):
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.05,hspace=0.3)
        img = cv2.imread(null_images[image_num], flags = 1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)















