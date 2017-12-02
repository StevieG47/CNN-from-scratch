#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:02:58 2017

@author: steve
"""

import numpy as np

class CNN(object):
    
    # Initiate neural network with input size and all layers to be used
    def __init__(self,inputShape,layers):
        
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
        
        # TODO: add shapes of weights for each layer, shapes of biases
        
        
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
                currentLayer.convolution(inputData)
                #print(currentLayer.output[0][10][10])
                
            # If pooling layer, do pooling
            if className == 'PoolingLayer':
                currentLayer.pool(inputData)
               # print(currentLayer.output[0][5][5])
                
            # If Fully connected Layer, do forward pass through the layer
            if className == 'FullyConnectedLayer':
                currentLayer.forwardPass(inputData)
                #print(currentLayer.output.shape)
                
            # If classification layer, do forward pass on it
            if className == 'ClassificationLayer':
                currentLayer.classify(inputData)
                
            # Get output, set it to previous Output var
            previousOutput = currentLayer.output
            
        finalOutput = previousOutput
        return finalOutput
            
            
                
            
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
                
                
# cBase lass to be used for the fully connected layer and classification layer 
class SingleLayer(object):
    
    # Define summed values (summed wT*x at a node) and output (summed values after activation)
    def __init__(self, inputShape, outputNum):
        self.output = np.ones((outputNum,1))
        self.summedValues = np.ones((outputNum,1))
        print("Hello There")
        
        
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
        
        
        
        
        
        
        
        
        
                
# Sigmoid Activation Function
def sigmoid(x):
    sig = 1/(1+np.exp(-x))
    return sig
                
        
        
        
        
        
        
        
        
        
                    
                
                
        
        
    