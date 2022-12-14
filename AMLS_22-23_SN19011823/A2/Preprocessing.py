#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 21:24:16 2022

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
import hyperpara_tuning
import svm_model
from matplotlib import pyplot as plt
from numpy import random



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

    return dlibout, resized_image



def run_dlib_mouth_shape(image, detector, predictor):
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

    return dlibout[49: 68,:], resized_image




def extract_features_labels(s_model, detector, predictor, basedir, labels_filename, images_dir):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        smile_labels:      an array containing the gender label (un-smile = 0 and smile = 1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    smile_labels =  {line.split('\t')[0] : int(line.split('\t')[3]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        null_imgs = []

        for img_path in image_paths:
            file_name= img_path.split('.')[-2].split('/')[-1]
            # load image
            img = image.load_img(img_path, target_size=target_size, interpolation='bicubic')
            img = image.img_to_array(img)
            if s_model == 'mouth':
                features, _ = run_dlib_mouth_shape(img, detector, predictor)
            if s_model == 'face':
                features, _ = run_dlib_shape(img, detector, predictor)
            if features is not None:
                all_features.append(features)
                all_labels.append(smile_labels[file_name])
                
            if features is None:
                null_imgs.append(img_path)

    landmark_features = np.array(all_features)
    smile_labels = (np.array(all_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1

    return landmark_features, smile_labels, null_imgs



def get_data(s_model, test_size, detector, predictor, basedir, labels_filename, images_dir):
    
    X, y, _ = extract_features_labels(s_model, detector, predictor, basedir, labels_filename, images_dir)
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = test_size, shuffle = True)
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    # X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
   
    return X_train, X_val, y_train, y_val



def get_test_data(s_model, detector, predictor, testdir, labels_filename, test_images_dir):
    
    X, y, _ = extract_features_labels(s_model, detector, predictor, testdir, labels_filename, test_images_dir)
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
   
    return X, y




def train_image_plotting(images_dir, num_image, detector, predictor):
    
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    
    plt.figure(figsize=(12,10))
    plt.suptitle('Facial and Mouth Feature Extraction', fontsize = 20, fontweight='bold')
    img = cv2.imread(image_paths[num_image], flags = 1)
    img = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(2,2,1)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    plt.imshow(img)
    plt.title('Original Image',fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    
    img = cv2.imread(image_paths[num_image], flags = 0)
    img = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(2,2,2)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    plt.imshow(img)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.title('Gray Scale Image',fontsize = 15)
    
    plt.subplot(2,2,3)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
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
    
    img = cv2.imread(image_paths[num_image], flags = 0)
    img = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(2,2,4)
    plt.subplots_adjust(wspace=0.05,hspace=0.2)
    plt.imshow(img)
    img = image.img_to_array(
        image.load_img(image_paths[num_image],
                       target_size=None,
                       interpolation='bicubic'))
    features, _ = run_dlib_mouth_shape(img, detector, predictor)
    plt.scatter(features[:,0], features[:,1], c = 'r', s = 8)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.title('Mouth Feature Extraction',fontsize = 15)






def null_image_plot(detector, predictor, basedir, labels_filename, images_dir):
    _,_,null_images = extract_features_labels('mouth', detector, predictor, basedir, labels_filename, images_dir)
    image_example = random.randint(202, size=(4))
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

#%% dir route
basedir = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/Datasets/celeba'
images_dir = os.path.join(basedir,'img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/A2/shape_predictor_68_face_landmarks.dat')

testdir = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0134 Applied ML Systems I/Assignment/AMLS_22-23_SN19011823/Datasets/celeba_test'
test_images_dir = os.path.join(testdir,'img')

#%% Hyper-Parameter Tuning
X_train, X_val, y_train, y_val =  get_data('face', 0.2, detector, predictor, basedir, labels_filename, images_dir)
# model = hyperpara_tuning.svm_model_search(X_train, y_train)
# print(model.best_params_)
# print(model.best_estimator_)
#%% Learning Curve Plotting
hyperpara_tuning.training_vs_cross_validation_score(X_train, y_train)

#%% Train and validation
pred, acc_score_train, conf_matrix_train = svm_model.img_SVM(X_train, y_train, X_train, y_train)

#%% Train Confusion matrix
svm_model.confusion_matrix_plot(conf_matrix_train, 'Train ')

#%% Validation
pred, acc_score_val, conf_matrix_val = svm_model.img_SVM(X_train, y_train, X_val, y_val)

#%% Validation Confusion Matrix
svm_model.confusion_matrix_plot(conf_matrix_val, 'Validation ')

#%% Test
X_test, y_test =  get_test_data('face', detector, predictor, testdir, labels_filename, test_images_dir)
X_train, y_train =  get_test_data('face', detector, predictor, basedir, labels_filename, images_dir)

pred, acc_score_test, conf_matrix_test = svm_model.img_SVM(X_train, y_train, X_test, y_test)

#%% Test Confusion Matrix
svm_model.confusion_matrix_plot(conf_matrix_test, 'Test ')

#%% Image Plot
train_image_plotting(images_dir, 90, detector, predictor)
null_image_plot(detector, predictor, basedir, labels_filename, images_dir)




