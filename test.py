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

# smax True to use softmax method with cross entropy loss function
# smax False to use MSE loss function
smax = 1
if smax: from CNNSoftmax import *
else: from CNN import *


# 71% accuracy on softmax w/ learningRate = .1, numTraining = 30,000, numEpochs = 10, batchSIze = 10
# This ^ didnt train for the full 30000 though

#################################---READ DATA---########################################
file  = gzip.open('mnist.pkl.gz', 'rb')
trainingData, validationData, testDataOriginal = pickle.load(file, encoding = 'latin') # read data from pickle file
file.close()

numTraining = 50000 # number of training images to use
numTesting = 1000 # number of test images to use

trainData = trainingData[0][0:numTraining] # only pick numTraining images from 50000 training images
trainLabel = trainingData[1][0:numTraining] # corresponding training labels

testData = testDataOriginal[0][0:numTesting] # same for testing
testLabel = testDataOriginal[1][0:numTesting]


#########################---RESHAPE DATA INTO IMAGES---################################
trainData = [np.reshape(x, (28,28)) for x in trainData] # reshape into 28x28 image
testData = [np.reshape(x, (28,28)) for x in testData]


#######################---ONE HOT ENCODING FOR LABELS---#############################
# Return a 10-dimensional unit vector with a 1 in the ith position and zero everywhere else
def oneHot(i):
    vec = np.zeros((10,1))
    vec[int(i)] = 1
    return vec

trainLabel = [oneHot(x) for x in trainLabel] # each label is 10x1 vector
#testLabel = [oneHot(x) for x in testLabel]


###############################---PLOT SOME IMAGES---######################################
training = [trainData,trainLabel]
testing = [testData,testLabel]
training = list(zip(trainData,trainLabel))
testing = list(zip(testData,testLabel))

# plot example image
index = 3
im = training[index][0]
#plt.imshow(im,cmap='binary') # plot binary plt
#plt.title(np.where(training[1][index] == 1)[0][0]) # show prediction in title

#################################---CREATE MODEL---########################################
x,y = training[0][0].shape
inputShape = (1,x,y) # images are 1 channel

# Create layers to be used
# dict = {'Name': 'Zara', 'Age': 7} # keys are Name and Age, values are Zara and 7
layers = [
        {'Convolution': {'filterSize': 5, 'stride': 1, 'numFilters': 20} }, # convolution layer
        {'Pooling': {'poolSize': (2,2)} }, # pooling to reduce computation
        {'fullyConnected': {'numOutput': 50} }, # full connected layer at the end
        {'outputLayer': {'numClasses': 10} } # 10 classes for digits 0-9
        ]

# Create Model
test = CNN(inputShape,layers) # init model
print(' ')
print('Model Initialized')

##################################---TRAIN MODEL---#########################################
batchSize = 10 # number of training images per batch, weights updated after each batch
if smax: learningRate = .1 # Best learning rate found for softmax (from testing)
else: learningRate = 1.5 # Best learning rate foudn for mse (from testing)
numEpochs = 5 # num of epochs to loop through
test.train(training,batchSize,learningRate,numEpochs) # train the model


#################################---GET ACCURACY---############################################
def getAccuracy(net,testData):
    print('Begin Testing')
    numCorrect = 0
    
    # Loop throught testing data
    for i in range(len(testData)):
        
        # Get current image and label
        im = testData[i][0].reshape(1,28,28)
        label = testData[i][1]
        
        # Make prediction w/ forward pass
        prediction = np.argmax(net.forwardPass(im))
       # print(prediction)
       
       # Mark if correct
        if prediction == label: numCorrect += 1
        
        # print at 10% iterations during testing
        if (i+1) % int(0.01 * len(testData)) == 0:
            print( '{0}% Completed'.format(int(float(i+1) / len(testData) * 100)))
    
    # Final accuracy
    print('Accuracy: ',numCorrect/len(testData)*100)
    
# Check accuracy
getAccuracy(test,testing)

# print parameters used
print('Learning Rate: ', learningRate)
print('Num Training Images: ',numTraining)
print('Num Epochs: ',numEpochs)
print('Batch Size: ',batchSize)


# Show a single picture with prediction as title
def predictNum(net,testData):
    i = np.random.randint(0,len(testData))
    im = testData[i][0].reshape(1,28,28)
    print(net.forwardPass(im))
    prediction = np.argmax(net.forwardPass(im))
    #print(net.forwardPass(im)[4])
    im = im.reshape(28,28)
    plt.imshow(im,cmap='binary') # plot binary plt
    plt.title('Prediction: %i' % prediction) # show prediction in title
    

# For predictin my own handwritten digits
def predictCustomIn(net,im):
    im = im.reshape(1,28,28)
    #print(net.forwardPass(im))
    prediction = np.argmax(net.forwardPass(im))
    #print(net.forwardPass(im)[4])
    im = im.reshape(28,28)
    plt.imshow(im,cmap='binary') # plot image
    plt.title('Prediction: %i' % prediction) # show prediction in title

