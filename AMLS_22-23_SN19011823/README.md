# README

The assignment is divided into four individual tasks, A1, A2 are binary classification problems on gender detection and smile detection, while B1 and B2 are multiclassification problems on face shapes and eye colours detection.

The main.py file will automatically run through all the tasks and functions with printed intructions, including train, validation and testing data generation, feature extraction, label extraction, hyperparameter tuning, model constrction and performance analysis

The 'xx_main.py' in each folder are the operation files which summaris and call all the funtions/modules constructed in the corresponding task.

The image folder in each task folder contains training, tuning or testing plot, image.

'.h5' files in Task A1 and B2 are deep learning model saved


## Task A1

The A1 folder contains several files/modules. 

The 'Preprocessing.py' module is used for extracting training labels and images then preprocess them.

The 'cnn_model.py' is the convolutional neural network constructed for this task.

The 'hyperpara_tuning.py' is used for passing all the potential options such as kernel numbers, kernel sizes, max pooling sizes to the cnn model for analysis.

The 'hyperpara_tuing_plot.py' is used for ploting performance of each tuning option, which takes the output from previous module.

The 'hyperpara_tuning_training' simply combines previous two modules/functions into one

The 'train_and_test.py' is used for traing and testing the performance of the model after tuning


## Task A2

As the name shows, the 'Preprocessing.py' model is similar to the one in A1, which extract the training image and labels

The 'svm_model' is for building a svm model used in this task

The 'hyperpara_tuning.py' is used for tuning the SVM model constructed.

The 'shape_predictor_68_face_landmarks.dat' file a pretrained model used for extracting facial and mouth features


## Task B1

'Preprocessing_feature_extraction.py' uses 'shape_predictor_68_face_landmarks.dat' file to extract facial feauture for training, while 'Preprocessing.py' is used for extracting original image and reprocess it into particular shape.

'KNN_model.py' is a K nearest neighbour model constructed for this case, while 'svm_model' contains svm model


## Task B2

The 'Preprocessing.py' module is used for extracting training labels and images crop the image and leaves only the left eye area. Then convert them into image_data_generator for cnn model training.

The 'cnn_model.py' is the convolutional neural network constructed for this task.

The 'hyperpara_tuning.py' is used for passing all the potential options such as kernel numbers, kernel sizes, max pooling sizes to the cnn model for analysis.

The 'hyperpara_tuing_plot.py' is used for ploting performance of each tuning option, which takes the output from previous module.

The 'hyperpara_tuning_training' simply combines previous two modules/functions into one

The 'train_and_test.py' is used for traing and testing the performance of the model after tuning

Above files/modules are similar to the ones in task A1.

The 'KNN_preprocessing.py' module extracts the cropped eye image and corresponding labels used for KNN model training, which contructed in 'KNN_model.py'

'Train_image_plot.py' is the files used for ploting croped eye image and other image data used for training.


## Datasets

The Datasets folder keeps empty when upload to github, but during runing the 'main.py' or other files in this repo, it should have the following structure at initial:

The dataset added in 'datasets' folder must follows the following file structure tree in order to make sure the code runs successfully

The labels.csv file in each folder contains training and test labels. For instance, in cartoon_set and cartoon_set_test, it contains, file names, face shapes and eye colours (0-4), while in celeba and celeba_test set, it contains gender and smiling labels (-1 and 1)
```
.
└── Datasets
    ├── cartoon_set
    │   ├── img
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   └── ...
    │   └── labels.csv
    ├── celeba
    │   ├── img
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   └── ...
    │   └── labels.csv
    ├── cartoon_set_test
    │   ├── img
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   └── ...
    │   └── labels.csv
    └── celeba_test
        ├── img
        │   ├── 0.png
        │   ├── 1.png
        │   └── ...
        └── labels.csv
```
When runing the code, two eye_img folders will be added to cartoon_set and cartoon_set_test automatically via functions, then the structure should be looked like:
```
.
└── Datasets
    ├── cartoon_set
    │   ├── img
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   └── ...
    │   ├── labels.csv
    │   └── eye_img (added automatically via functions)
    │       ├── 0.png
    │       ├── 1.png
    │       └── ...
    ├── celeba
    │   ├── img
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   └── ...
    │   └── labels.csv
    ├── cartoon_set_test
    │   ├── img
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   └── ...
    │   ├── labels.csv
    │   └── eye_img (added automatically via functions)
    │       ├── 0.png
    │       ├── 1.png
    │       └── ...
    └── celeba_test
        ├── img
        │   ├── 0.png
        │   ├── 1.png
        │   └── ...
        └── labels.csv
```

# Working environment and requirements

- python = 3.7

## Required packages

- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- seaborn
- keras
- tensorflow = 2.11.0
- spyder-kernels
- OpenCV-python
- keras-preprocessing
- dlib

