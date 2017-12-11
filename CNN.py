#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:02:58 2017

@author: steve
"""

import numpy as np
import random
from func import *

class CNN(object):
    
    # Initiate neural network with input size and all layers to be used
    def __init__(self,inputShape,layers):
        
        self.errorTrack = []
        
        self.inputShape = inputShape
        
        # Convert layers input to correct layer class
        layerClasses = {'Convolution' : ConvolutionalLayer,
                       'Pooling' : PoolingLayer,
                       'fullyConnected' : FullyConnectedLayer,
                       'outputLayer' : ClassificationLayer}
        
        CNNLayers = []
        shape = inputShape
        for i in range(len(layers)):
            
            # get name of key in layer, ex: Convolution, Pooling etc, the string inputted
            layerName = list(layers[i].keys())[0] 
            
            # convert input string to class name ex: 'Convolution' to ConvolutionLayer class
            layerClass = layerClasses[layerName] 
            
            # arguments of the layer, ex: filterSize, numFilters etc
            layerArguments = list(layers[i].values())[0] 
            
            # **kwargs allows you to pass keyworded variable length of arguments, our arguments have keys,values
            currentLayer = layerClass(shape,**layerArguments)
            
            # change shape so output shape of this layer is input to next layer
            shape = currentLayer.output.shape
            
            # Add layer to the list of layers
            CNNLayers.append(currentLayer)
        
        # Define layers
        self.layers = CNNLayers
        
        # Loop through layers and get the weight shapes for each
        # Pooling layer doesnt have weights that just makes data smaller for easier computation
        self.weightShapes = [currentLayer.weights.shape for currentLayer in self.layers if type(currentLayer).__name__ != 'PoolingLayer']
        self.biasShapes = [currentLayer.biases.shape for currentLayer in self.layers if type(currentLayer).__name__ != 'PoolingLayer']
        
    def forwardPass(self, im):
        
        # Define previous Output var to be used when moving through layers
        previousOutput = im
        
        # Do a forward pass thorugh 
        for currentLayer in self.layers:
            
            # Set input as the previous output
            inputData = previousOutput
            
            # Get name of current class
           # print(currentLayer)
            className = type(currentLayer).__name__
           # print(className)
            
            # If convolutional layer, do convolution
            if className == 'ConvolutionalLayer':
                #print('Convoluting')
                currentLayer.convolution(inputData)
                #print('Done Convulting')
                #print(currentLayer.output[0][10][10])
                
            # If pooling layer, do pooling
            if className == 'PoolingLayer':
                #print('Pooling')
                currentLayer.pool(inputData)
                #print('Done Pooling')
               # print(currentLayer.output[0][5][5])
                
            # If Fully connected Layer, do forward pass through the layer
            if className == 'FullyConnectedLayer':
                #print('Begin Fully connected')
                currentLayer.forwardPass(inputData)
                #print('Done w/ fully connected')
                #print(currentLayer.output.shape)
                
            # If classification layer, do forward pass on it
            if className == 'ClassificationLayer':
                #print('Begin classify')
                currentLayer.classify(inputData)
                #print('Done w/ classify layer')
                
            # Get output, set it to previous Output var
            previousOutput = currentLayer.output
            
        finalOutput = previousOutput
        return finalOutput
    
    
    # Use gradient descent to train network
    def train(self,trainingData, batchSize, learningRate, numEpochs, lamdaVal = None):
        
        # Define training size
        trainSize = len(trainingData)
        
        # Keep track of error
        meanError = []
        
        # Keep track of what epoch we are on
        epochNum = 1
        
        # Total trainingRuns
        numTrainingRuns = len(trainingData)/batchSize * numEpochs * 1.0
        currentRun = 1
        
        # Loop through all epochs
        for currentEpoch in range(numEpochs):
            print('Starting Epoch ', epochNum, ' of ',numEpochs)
            
            # Shuffle the training Data
            random.shuffle(trainingData)
            
            # Define batches to be used for training
            # Each batch has batchSize elements
            # There are a total of trainSize/batchSize batches
            batches = [trainingData[i:i+batchSize] for i in range(0,trainSize, batchSize)]
          
            # Start loss at 0
            losses = 0
            
            # Keep track of what batch we are on
            batchNum = 1
            
            # Loop through batches, backpropogate
            for currentBatch in batches:
                
                print('Epoch ', epochNum,' Batch ',batchNum, ' of ',len(batches), ' ',
                      round(currentRun/numTrainingRuns*100,2),'% Done Training')
                batchNum += 1
                currentRun +=1
            
                # Updated loss
                batchLoss = self.updateLoss(currentBatch,learningRate)
                losses = losses + batchLoss
               # print('Batch Loss: ', batchLoss)
               
           # meanError.append(round(losses/batchSize,2))
            print ('Mean Error: ', round(losses/batchSize,2))
            epochNum += 1 # move to next epoch
        
        print('Done Training')
        
    
    # Take a batch and back propogate to update weights
    def updateLoss(self, batch, LearningRate):
        
        # Initialize derivatives using list of weight/bias shapes for each layer
        derivW = [np.zeros(shape) for shape in self.weightShapes]
        derivB = [np.zeros(shape) for shape in self.biasShapes]
        
        # Initialize batch length
        batchLength = len(batch)
        
        # Loop through each image in the batch
        # batch[0] is the image, batch[1] is the length 10 one hot encoded label
        for image, label in batch:
            
            # Include the depth in image
            im = image.reshape((1,28,28))
            
            # Do forward pass to update self.layers (compute outputs), variable flag doesnt matter
            flag = self.forwardPass(im)
            
            # Get partial derivatives for weights, biases
            finalO, partialB, partialW = self.backpropogate(im,label)
            
            # derivB is list with biases for convolutional layer, fully connected layer, and classify layer
            # partialB is the partialL/partialBias values for all the same layers, is is also
            # a list with each element being the same shape
            # For each batch continously add the partialL/partialBias to the derivB term
            # Same for derivW
            derivB = [nb + db for nb, db in zip(derivB, partialB)]
            derivW = [nw + dw for nw, dw in zip(derivW, partialW)]
        
        # Get error for last label, final output i batch
        error = loss(label, finalO)
        self.errorTrack.append(error)
        
        # Make list of layer indices that have weights, for this single block model of conv-->pool-->fc--->classify,
        # wIndex will just be [0,2,3]
        ind = 0
        
        # list of indices of layers that have weights
        wIndex = []
        
        # loop thorught layers
        for layer in self.layers:
            
            # if it's not pooling (everything but pooling has weights)
            if type(layer).__name__ != 'PoolingLayer':
                
                # add layer indices number
                wIndex.append(ind)
            
            # increase layer indices we're on
            ind += 1
        
        
        for iterationNum, (lnw, lnb) in enumerate(zip(derivW, derivB)):
            
            # iterationNum will be 0,1,2 and our wIndex (for this single block method) is 
            # [0 2 3] so when we iterate we'll get wIndex[0],wIndex[1], wIndex[2] or 
            # layer[0],layer[2],layer[3], skipping pooling layer
            layer = self.layers[wIndex[iterationNum]]
            
            # Update current layers weights with mean partialW
            # We had gotten the derivW by continusouly adding to derivW for each batch, so
            # divide by the number of batches to get the mean. 
            # Move in the negative gradient deirection to move toward loss minimum        
            layer.weights -= LearningRate * lnw / batchLength
            layer.biases -= LearningRate * lnb / batchLength
        
        # return the error so we can keep track of it
        return error
    
    
    # Backpropogation function
    def backpropogate(self,im,label):
        
        # Initialize derivatives using list of weight/bias shapes for each layer
        derivW = [np.zeros(shape) for shape in self.weightShapes]
        derivB = [np.zeros(shape) for shape in self.biasShapes]
        
        # Prediction is the output vector of the final layer
        prediction = self.layers[len(self.layers)-1].output
        
        # Parital Loss / partial Z, where z is the summed values before activation of the final layer
        # Z = w transpose x + bias, the output (yhat) is just sigmoid(z2)
        # So it looks like partialL_Z = (yhat-y) * deriv sigmoid(z2) where z2 is summedValues of last layer ie Classification layer
        partialL_Z = (prediction - label) * dSigmoid(self.layers[len(self.layers)-1].summedValues)
        
        # Get layer transition, so layer1 --> layer2
        # Loop through the transitions and get partial derivatives 
        # Loop ends at layerNum = 0th layer (convolution layer)
        # classify layer is layers[3], fully connected is layers[2], pooling is layers[1], convolution is layers[0], the raw image is before that
        for layerNum in range(len(self.layers)-1, -1, -1): # Loop backwards from layers
            
            # With n layers, len(self.layers)-1 is the classify layer
            # Get consecutive layers 1 and 2 where layer1 --> layer2
            layer2 = layerNum
            layer1 = layerNum - 1
            
            # Define current layer
            currentLayer = self.layers[layer2]
            
            # Define previous output, the output from 
            # If num is zero or above is is convolution,pooling, fully connected, or classification
            if layer1 > -1:
                prevOut = self.layers[layer1].output
            
            # if layer1 is -1, then layerNum is convolution and the prevOut is just the image
            elif layer1 == -1:
                prevOut = im
                
            
            # Get names of consecutive layers
            if layer1 > -1: layer1Name = type(self.layers[layer1]).__name__ # get layer 1 name
            if layer1 == -1: layer1Name = 'image' # if layer2 is convolution, layer1 isnt a layer it's just the im
            layer2Name = type(self.layers[layer2]).__name__ # get layer 2 name
            
            
            # Find Derivatives based on layer transition
            # Update classification layer weights
            if layer1Name == 'FullyConnectedLayer' and layer2Name == 'ClassificationLayer':
                
                deltaB, deltaW, partialL_Z= AutoDiff_CL(partialL_Z,prevOut,
                                                        currentLayer.summedValues)
               # print('Class to Full')

            # Update weights of fully connected layer
            if layer1Name == 'PoolingLayer' and layer2Name == 'FullyConnectedLayer':
                
                # prevOut is output of pooling layer, summedValues is Z of fully connected layer
                # partialL_Z is (yhat-y)*partialSigmoid(z2)/partial_z2 where z2 is summed values of CL layer
                deltaB, deltaW, partialL_Z = AutoDiff_FC(partialL_Z, prevWeights, 
                                                         prevOut, currentLayer.summedValues)
               # print('Full to Pool')
                # The new partialL_Z is now wrt Z1, or the summedValues of the fully connected layer
                
                
            # Update pool layer partials
            # The pooling layer doesnt have weights, but it is another layer so we need to differentiate to get back to the conv layer
            # only difference now is that some values are zeros since they didnt contribute to output (they werent the max value)
            if layer1Name == 'ConvolutionalLayer' and layer2Name == 'PoolingLayer':
                
                partialL_Z = AutoDiff_PL(partialL_Z, prevWeights, prevOut, 
                                         currentLayer.maxIndices, currentLayer.poolSize, currentLayer.output)
               # print('Pool to Conv')
            
            
            
            # Update conv layer weights
            # partialL_Z is now the size of conv layer output
            if layer1Name == 'image' and layer2Name == 'ConvolutionalLayer':
                
                # Get weights of the convolutional layer 
                prevWeights = currentLayer.weights 
                
                # Find partial bias, partial weights
                # partialL_Z is partial of convolutional output, prevWeights is conv weights, stride is stride, im is input image, last arg is output of conv
                deltaB, deltaW = AutoDiff_ConvL(partialL_Z, prevWeights, currentLayer.stride, im, currentLayer.outputValues)
            
              #  print('Conv to im')
            
            
            
          
            
            # Set previous weights
            # Pool has no weights so dont do it then 
            if not(layer1Name == 'ConvolutionalLayer' and layer2Name == 'PoolingLayer'):
                
                # Avoid putting derivB[-1] since that changes last value of array
                # layer will only be one when we're at image --> convolutional
                if layer1 == -1:
                    layer1 = 0
                derivB[layer1], derivW[layer1] = deltaB, deltaW
                prevWeights = currentLayer.weights
               
    
        # Return the output of the classification to check error, and the partial weights, biases
        return self.layers[-1].output, derivB, derivW
                
            
            
            
            
            
          
        
        
        
        
            
            
                
            
class ConvolutionalLayer(object):
    
    def __init__(self,inputShape,filterSize,numFilters,stride):
        
        # Get height, width, depth of image
        self.depth = inputShape[0] # number of channels of the image, 3 if RGB image, 1 if binary/grayscale
        self.height = inputShape[1]
        self.width = inputShape[2]
        
        # Add arguments for filter, stride and padding
        self.filterSize = filterSize
        self.stride = stride
        self.numFilters = numFilters
        self.padding = 0 # area outside the input is padded with zeros
        
        # Initialize random weights and biases
        
        # We will slide/convolve each filter/weight across the width/height of the image
        # Will compute dot products between filter and input at the position in the image
        # This will produce 2d map that gives responses of a filter at every position
        # The network will learn filters that activate when they see some feature like an edge or curve
        # Each filter will produce a separate 2d map
        
        # EXAMPLE:  suppose that the input volume has size [32x32x3].
        # If the receptive field (or the filter size) is 5x5, then each neuron in the Conv Layer will 
        # have weights to a [5x5x3] region in the input volume, for a total of 5*5*3 = 75 weights
        # (and +1 bias parameter). Notice that the extent of the connectivity along the depth axis
        # must be 3, since this is the depth of the input volume.
        
        # With MNIST the depth is 1, so we have 20 5x5 filters that will slide across image
        # the output will be , since each filter outputs a 5x5x1. 
        
        # randn returns samples from standard normal distribution
        self.weights = np.random.randn(numFilters, self.depth, filterSize, filterSize) # num filter, num of channels in image, filtr size
        self.biases = np.random.rand(self.numFilters,1) # bias for each filter is just 1 number, will add that numeber after summing weight values
        
        
        # Set output with height and width of image H1, W1, size of filter F, padding P, stride S
        # Output dimension is W2xH2xD2 (widthxheightxdepth) where:
        # W2 = (W1 - F + 2P)/S + 1
        # H2 = (H1 - F + 2P)/S + 1
        # D2 = K
        self.outputRows =  int( (self.height - self.filterSize + 2*self.padding)/self.stride  + 1 )
        self.outputCols = int ( (self.width - self.filterSize + 2*self.padding)/self.stride + 1 )
    
        # Set output
        self.output = np.zeros((self.numFilters, self.outputRows, self.outputCols))
        self.outputValues = np.zeros((self.numFilters, self.outputRows, self.outputCols)) # values before activation
        
        print('Convolutional Layer Initialized')

    # The actual convolution
    def convolution(self,inputData):
        
        # Flatten rows/columns into one long array so output is 20x(rows*columns)
        self.outputValues = self.outputValues.reshape((self.numFilters,self.outputRows * self.outputCols))
        self.output = self.output.reshape((self.numFilters,self.outputRows * self.outputCols))
        
        # Get length of flattened output value. This is the total number of values in the output of convolution
        outputLength = self.outputRows * self.outputCols
        
        # Loop through all filters
        for i in range(self.numFilters):
            col = 0
            row= 0
            
            # Loop through sliding the filter across the image, loop until every value of outputLength is found
            for j in range(outputLength):
                
                # Take dot product of filter with part of image it's on
                # weights[i] is a 1x5x5 since filterSize is 5 and depth is 1
                # inputData[: , row:row+self.filterSize, col:col+self.filterSize] is also a 1x5x5 since depth is 1, and were specifying rows and columns to be a:a+filterSize
                # Element-wise multiplication is done with *
                dotProduct = inputData[: , row:row+self.filterSize, col:col+self.filterSize] * self.weights[i]
                
                # Sum the element multiplaction values
                sumValue = np.sum(dotProduct)
                
                # Add the bias, outputValues has values before activation
                self.outputValues[i][j] = sumValue + self.biases[i]
                
                # Activation function
                self.output[i][j] = sigmoid(self.outputValues[i][j])
                
                # Move horizontally across row
                col += self.stride
                
                # Check if we are at the end of the row (at last column)
                if col + self.filterSize -self.stride >= self.width:
                    col = 0 # reset column to zero
                    row += self.stride # move row of filter down
        
        # Reshape output back into correct shape
        self.outputValues = self.outputValues.reshape((self.numFilters, self.outputRows, self.outputCols))
        self.output = self.output.reshape((self.numFilters, self.outputRows, self.outputCols))
        
class PoolingLayer(object):
    
    def __init__(self, inputShape, poolSize):
        
        # Get height, width, depth of input
        self.depth = inputShape[0] # number of channels of the image, 3 if RGB image, 1 if binary/grayscale
        self.height = inputShape[1]
        self.width = inputShape[2]
        
        # Define pool size and strid
        self.poolSize = poolSize
        self.stride = 2
        
        # Pooling layer takes in a volume of w1xh1xd1 (width,height,depth)
        # hyperparameters for pooling are poolSize F and stride S
        # Outputs a volume W2xH2xD2 where:
        # W2 = (W1-F)/S + 1
        # H2 = (H1-F)/W + 1
        # D2 = D1
        
        # Only 2 common seen variations, poolSize = 3, stride = 2, and poolSize = 2, stride = 2
        # poolSizes larger are too destructive
        # Max pooling works better than average or L2 pooling
        
        # Set output dimensions
        self.outputHeight = (self.height - self.poolSize[0])/self.stride + 1
        self.outputWidth = (self.width - self.poolSize[0])/self.stride + 1
        
        # Set output shape
        # np.empty just sets the shape
        self.outputHeight = int(self.outputHeight)
        self.outputWidth = int(self.outputWidth)
        self.output = np.empty((self.depth, self.outputHeight, self.outputWidth))
        
        # Set max indicies matrix, because "during the forward pass of a pooling layer it is 
        # common to keep track of the index of the max activation (sometimes also called the switches)
        # so that gradient routing is efficient during backpropagation."
        self.maxIndices = np.empty((self.depth, self.outputHeight, self.outputWidth,2)) # coordinates are row, column so 2
        
        print('Pooling Layer Initialized')
        
    # The actual pooling
    def pool(self,inputData):
        
        # Flatted height width of input
        self.Length = self.outputHeight * self.outputWidth
        
        # Reshape output and max indices matrix with flattened length
        self.output = self.output.reshape((self.depth, self.Length))
        self.maxIndices = self.maxIndices.reshape((self.depth,self.Length,2)) 
        
        # Loop through each filter (input from convolution is numFilters x imshape x imshape)
        for i in range(self.depth):
            row = 0
            col = 0
            
            # Loop through each value of Length, loop until every value of height,width is found
            for j in range(self.Length-1):
                
                # Define section pool filter is over
                section = inputData[i][row:row + self.poolSize[0], col:col + self.poolSize[0]]
                
                # Get the max value of the section, set output data
                maxVal = np.amax(section) # amax gives max of an array
                self.output[i][j] = maxVal
                
                # Get max indices, if theres a tie just take first index
                # np.where will check each element over an array
                maxIndex = np.where(section == np.max(section)) 
                if len(maxIndex[0]) > 1:
                    maxIndex = [maxIndex[0][0], maxIndex[1][0]]
                
                # maxIndex is just indices of the current section, not the inputData, so 
                # add row, col to get the actual indices
                maxIndex = int(maxIndex[0]) + row, int(maxIndex[1]) + col
                
                # Update max indices
                self.maxIndices[i][j] = maxIndex
                
                # Move across horizontally
                col += self.stride
                
                # If we're at the end of the row (at last column) move down stride rows
                if col >= self.width:
                    col = 0 # reset column
                    row += self.stride # move row down
                
        # Reshape output and maxIndices back to original shape
        self.output = self.output.reshape((self.depth, self.outputHeight, self.outputWidth))
        self.maxIndices = self.maxIndices.reshape((self.depth, self.outputHeight, self.outputWidth, 2))
                
                
# class to be used for the fully connected layer and classification layer 
class SingleLayer(object):
    
    # Define summed values (summed wT*x at a node) and output (summed values after activation)
    def __init__(self, inputShape, outputNum):
        self.output = np.ones((outputNum,1))
        self.summedValues = np.ones((outputNum,1))
        #print("Hello There")
        
        
class FullyConnectedLayer(SingleLayer):
    
    def __init__(self,inputShape, numOutput):
        
        # Use the base class Single Layer to initialize the input and output
       # super(SingleLayer,self).__init__(inputShape, numOutput)
        super(SingleLayer,self).__init__()
        
        self.output = np.ones((numOutput,1))
        self.summedValues = np.ones((numOutput,1))
        
        # Get depth, width, height
        self.depth = inputShape[0]
        self.width = inputShape[1]
        self.height = inputShape[2]
        
        # Set output number
        self.numOutput = numOutput
        
        # Neurons in the fc layer have full connections like how regular neural networks usually are
        # Can do matrix multiplication with input, so at this stage theres no more volume, just flat array
        
        # Set weights and biases
        self.weights = np.random.randn(self.numOutput,self.depth,self.height,self.width) # weights are numberOutpus x volume of pooling output
        self.biases = np.random.rand(self.numOutput,1)
        
        print('Fully Connected Layer Initialized')
    
    # Forward pass through the layer, outputs input to final layer
    def forwardPass(self,inputData):
        
        # We'll do matrix multiplication so we want weights x data = output where
        # output is ouputNum x 1, so
        # weights is outputNum x volume 
        # data is volume x 1
        # so weights x data becomes a outputNum x 1 matrix
        
         # Flatten weights and flatten input data
        self.weights = self.weights.reshape((self.numOutput, self.depth*self.height*self.width))
        inputData = inputData.reshape((self.depth*self.height*self.width,1))
        
        # w transpose x + bias
        self.summedValues = np.dot(self.weights,inputData) + self.biases
        
        # Output is summedValues pushed through activation
        self.output = sigmoid(self.summedValues)
        
        
        # Reshape weights back
        self.weights = self.weights.reshape((self.numOutput, self.depth, self.height, self.width))

class ClassificationLayer(SingleLayer):
    
    def __init__(self, inputShape, numClasses):
        
        # Use base class Single Layer to to initialize input and output
        # Output is the classes 0-9
       # super(SingleLayer,self)._init_(inputShape, numClasses)
        super(SingleLayer,self).__init__()
        
        self.output = np.ones((numClasses,1))
        self.summedValues = np.ones((numClasses,1))
        
        # Set output classes
        self.numClasses = numClasses
        
        # Set weights and biases
        self.weights = np.random.randn(self.numClasses,inputShape[0]) # weights are 10xinputshape
        self.biases = np.random.randn(self.numClasses,1)
        
        print('Classification Layer Initialized')
    
    # The actual classification
    def classify(self,data):
        
        # Get summed values, w transpose x, add bias
        # w transpose x will be [10 x inputShape] x [inputShape x 1] to give a [10x1] output matrix
        self.summedValues = np.dot(self.weights,data) + self.biases
        
        # Output is summed values through activation
        self.output = sigmoid(self.summedValues)
        
        
        
        
# MSE Error
def loss(desired, final):
    return .5 * np.sum(desired-final)**2
                    
                
                
        
        
    