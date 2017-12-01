#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:02:58 2017

@author: steve
"""

import numpy as np

class CNN(object):
    
    # Initiate neural network with input size and all layers to be used
    def _init_(self,inputShape,layers):
        
        self.inputShape = inputShape
        
        # Convert layers input to correct layer class
        layerClasses = {'Convolution' : ConvolutionLayer,
                       'Pooling' : PoolingLayer,
                       'fullyConnected' : FullyConnectedLayer,
                       'outputLayer' : ClassifyLayer}
        
        CNNLayers = []
        shape = inputShape
        for i in range(layers):
            
            # get name of key in layer, ex: Convolution, Pooling etc
            layerName = list(layers[i].keys())[0] 
            
            # convert string to class name ex: 'Convolution' to ConvolutionLayer class
            layerClass = layerClasses(layerName) 
            
            # arguments of the layer, ex: filterSize, numFilters etc
            layerArguments = list(layers[0].values())[0] 
            
            # **kwargs allows you to pass keyworded variable length of arguments, our arguments have keys,values
            currentLayer = layerClass(shape,**layerArguments)
            
            # change shape so output shape of this layer is input to next layer
            shape = currentLayer.output.shape
            
            # Add layer to the list of layers
            CNNLayers.append(currentLayer)
        
        # Define layers
        self.layers = CNNLayers
        
        # TODO: add shapes of weights for each layer, shapes of biases
                
                
            
class ConvolutionalLayer(object):
    
    def _init_(self,inputShape,filterSize,numFilters,stride):
        
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
        # the output will be 5x5x20, since each filter outputs a 5x5x1. 
        
        # randn returns samples from standard normal distribution
        self.weights = np.random.randn(numFilters, self.depth, filterSize, filterSize) # num filter, num of channels in image, filtr size
        self.biases = np.random.rand(self.numFilters,1) # bias for each filter is just 1 number, will add that numeber after summing weight values
        
        
        # Set output with height and width of image H1, W1, size of filter F, padding P, stride S
        # Output dimension is W2xH2xD2 (widthxheightxdepth) where:
        # W2 = (W1 - F + 2P)/(S+1)
        # H2 = (H1 - F + 2P)/(S+1)
        # D2 = K
        self.outputRows =   (self.height - self.filterSize + 2*self.padding)/self.stride + 1
        self.outputHeight = (self.width - self.filterSize + 2*self.padding)/self.stride + 1
    
        # Set output
        self.output = np.zeros((self.numFilters, self.outputRows, self.outputHeight))
        
        
    