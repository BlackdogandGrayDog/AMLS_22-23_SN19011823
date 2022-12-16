#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 02:51:31 2022

@author: ericwei
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
import dlib
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from numpy import random
import KNN_model
import SVM_model
import Preprocessing
import hyperpara_tuning
from sklearn.metrics import log_loss
#%%

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
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    # X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
   
    return X_train, X_val, y_train, y_val



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


#%%
basedir = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/Datasets/cartoon_set'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/B1/shape_predictor_68_face_landmarks.dat')

testdir = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/Datasets/cartoon_set_test'
test_images_dir = os.path.join(testdir,'img')

#%% Features Extraction Training
train_image_plotting(images_dir, 93, detector, predictor)

X_train, X_val, y_train, y_val = get_data(0.2, detector, predictor, basedir, labels_filename, images_dir)

#%% KNN Training (Features Extraction)
acc = KNN_model.KNN_model(X_train, y_train, X_val, y_val)
KNN_model.KNN_model_plot(acc)

#%% SVM Training (Features Extraction)
pred, acc_score_val, conf_matrix_val = SVM_model.img_SVM(X_train, y_train, X_val, y_val)
SVM_model.training_vs_cross_validation_score(X_train, y_train)



#%% Orginal Figure Training
train_img = Preprocessing.image_processing(basedir, images_dir)
train_labels =  Preprocessing.extract_labels(basedir, labels_filename, images_dir)
train_img = train_img.reshape((train_img.shape[0], train_img.shape[1] * train_img.shape[2] * train_img.shape[3]))

X_train, X_val, y_train, y_val = train_test_split(train_img, train_labels, test_size = 0.2, shuffle = True)

#%% Orginal Figure Testing
test_img = Preprocessing.image_processing(testdir, test_images_dir)
test_labels =  Preprocessing.extract_labels(testdir, labels_filename, test_images_dir)
test_img = test_img.reshape((test_img.shape[0], test_img.shape[1] * test_img.shape[2] * test_img.shape[3]))

#%% KNN Training (Original)
acc = KNN_model.KNN_model(X_train, y_train, X_val, y_val)
KNN_model.KNN_model_plot(acc)

#%% SVM Training validation and testing (Original)
pred, acc_score_train, conf_matrix_train, pred_prob = SVM_model.img_SVM(X_train, y_train, X_train, y_train)
train_loss = log_loss(y_true = y_train, y_pred = pred_prob)

pred, acc_score_val, conf_matrix_val, pred_prob = SVM_model.img_SVM(X_train, y_train, X_val, y_val)
val_loss = log_loss(y_true = y_val, y_pred = pred_prob)

pred, acc_score_test, conf_matrix_test, pred_prob = SVM_model.img_SVM(train_img, train_labels, test_img, test_labels)
test_loss = log_loss(y_true = test_labels, y_pred = pred_prob)

#%% Confusion Matrix
SVM_model.confusion_matrix_plot(conf_matrix_train, 'Training ')
SVM_model.confusion_matrix_plot(conf_matrix_val, 'Validation ')
SVM_model.confusion_matrix_plot(conf_matrix_test, 'Test ')

#%%
hyperpara_tuning.training_vs_cross_validation_score(X_train, y_train)












