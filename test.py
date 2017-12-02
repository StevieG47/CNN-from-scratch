#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:24:26 2017
@author: steve
"""

import _pickle as pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg

from CNN import *

#################################---READ DATA---########################################
file  = gzip.open('mnist.pkl.gz', 'rb')
trainingData, validationData, testDataOriginal = pickle.load(file, encoding = 'latin') # read data from pickle file
file.close()

numTraining = 1000 # number of training images to use
numTesting = 500 # number of test images to use

trainData = trainingData[0][0:numTraining] # only pick numTraining images from 50000 training images
trainLabel = trainingData[1][0:numTraining] # corresponding training labels

testData = testDataOriginal[0][0:numTesting]
testLabel = testDataOriginal[1][0:numTesting]


#########################---RESHAPE DATA INTO IMAGES---################################
trainData = [np.reshape(x, (28,28)) for x in trainData] # reshape into 28x28 image
testData = [np.reshape(x, (28,28)) for x in testData]


#######################---ONE HOT ENCODING FOR LABELS---#############################
# Return a 10-dimensional unit vector with a 1 in the ith position and zeroe everywhere else
def oneHot(i):
    vec = np.zeros((10,1))
    vec[int(i)] = 1
    return vec

trainLabel = [oneHot(x) for x in trainLabel]
#testLabel = [oneHot(x) for x in testLabel]


###############################---PLOT SOME IMAGES---######################################
training = [trainData,trainLabel]
testing = [testData,testLabel]
training = list(zip(trainData,trainLabel))
testing = list(zip(testData,testLabel))
index = 10
im = training[index][0]
#plt.imshow(im,cmap='binary') # plot binary plt
#plt.title(np.where(training[1][index] == 1)[0][0]) # show prediction in title

#################################---CREATE MODEL---########################################
x,y = training[0][0].shape
inputShape = (1,x,y)

# Create layers to be used
# dict = {'Name': 'Zara', 'Age': 7} # keys are Name and Age, values are Zara and 7
layers = [
        {'Convolution': {'filterSize': 5, 'stride': 1, 'numFilters': 20} }, # convolution layer
        {'Pooling': {'poolSize': (2,2)} }, # pooling to reduce computation
        {'fullyConnected': {'numOutput': 30} }, # full connected layer at the end
        {'outputLayer': {'numClasses': 10} } # 10 classes for digits 0-9
        ]

# Create Model
test = CNN(inputShape,layers)
print(' ')
print('Model Initialized')

# Show a single picture with prediction
def predictNum(net,testData):
    i = np.random.randint(0,len(testData))
    im = testData[i][0].reshape(1,28,28)
    print(net.forwardPass(im))
    prediction = np.argmax(net.forwardPass(im))
    #print(net.forwardPass(im)[4])
    im = im.reshape(28,28)
    plt.imshow(im,cmap='binary') # plot binary plt
    plt.title('Prediction: %i' % prediction) # show prediction in title
    
def getAccuracy(net,testData):
    numCorrect = 0
    for i in range(len(testData)):
        im = testData[i][0].reshape(1,28,28)
        label = testData[i][1]
        prediction = np.argmax(net.forwardPass(im))
        if prediction == label: numCorrect += 1
    print('Accuracy: ',numCorrect/len(testData)*100)

