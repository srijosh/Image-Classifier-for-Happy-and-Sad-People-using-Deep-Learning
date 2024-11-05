# Image Classifier for Happy and Sad People using Deep Learning

This repository contains a project for classifying images of people as either happy or sad using a Convolutional Neural Network (CNN). The project demonstrates the use of deep learning for binary image classification, including data collection, model building, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)

## Introduction

Image classification is a key application of deep learning, commonly used in fields such as facial expression recognition, emotion analysis, and user experience research. This project focuses on developing a CNN-based classifier to differentiate between images of happy and sad people. The model is trained on a dataset collected from online sources and aims to achieve high accuracy in classifying emotions.

## Dataset

The dataset consists of images downloaded using the Google Chrome extension "Download All Images." Images were gathered by searching for "happy people" and "sad people" on Google and organizing them into the following folders:

- data/happy: Contains images of happy people.
- data/sad: Contains images of sad people.
  The dataset is further divided into training, validation, and testing sets for model evaluation.

## Installation

To run this project, you need to have Python installed on your machine. You can install the required dependencies using pip.

## Installation

To run this project, you need to have Python installed on your machine. You can install the required dependencies using `pip`.

```
pip install tensorflow tensorflow-gpu opencv-python matplotlib


```

Requirements
Python 3.x
TensorFlow
OpenCV
NumPy
Matplotlib

## Usage

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/Image-Classifier-for-Happy-and-Sad-People-using-Deep-Learning.git
```

2. Navigate to the project directory:
   cd Image-Classifier-for-Happy-and-Sad-People-using-Deep-Learning

3. Open and run the Jupyter Notebook:
   - jupyter notebook ImageClassifier.ipynb

## Model

The model is built using a Convolutional Neural Network (CNN) with TensorFlow and Keras. The main architecture includes:

- Conv2D and MaxPooling2D layers: Used for feature extraction from images.
- Flatten layer: Converts the 2D matrix of features into a 1D vector.
- Dense layer: Fully connected layers for classification.

The model is compiled with the Adam optimizer, BinaryCrossentropy loss function, and metrics for precision, recall, and accuracy.

### Data Preprocessing

- Loading and Resizing Images: Each image is read from its file path, decoded, and resized to a specified size for consistent input dimensions.

- Creating a TensorFlow Dataset: The images and labels are converted into a TensorFlow dataset, which is shuffled and batched for training.

### Model Training

The model is trained using a CNN architecture with the following key layers:

- Convolutional Layers: For feature extraction from the images.
- MaxPooling Layers: To reduce dimensionality and retain important features.
- Dense Layer: For final classification output.

### Evaluation

The model's performance is evaluated using the following metrics:

- Binary Crossentropy Loss: This loss function quantifies the model's performance in classifying the two classes (happy and sad), helping to optimize the model during training.
- Precision: Measures the ratio of true positive predictions to the total predicted positives, indicating how many of the predicted happy images are actually happy.
- Recall: Assesses the ratio of true positive predictions to the total actual positives, providing insight into how well the model identifies happy images.
- Binary Accuracy: Represents the percentage of correctly classified images, giving an overall indication of the model's classification performance.
