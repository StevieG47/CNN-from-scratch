#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:36:34 2017

@author: steve
"""

import numpy as np
import pickle # for saving  and loading out serialized model 
from app.model.preprocessor import Preprocessor as img_prep #image preprocessing

# Class for loading our saved model and classufying new images
class LiteOCR: # Optical Character Recognition Class
    
    # load weights from picle file then store all labels, load CNN stuff
    # For like predicting a new image, actual CNN class under this
    def _init_(self, fn = "alpha_weights.pkl",pool_size = 2):
        
        # load the weights from the pickle file and the meta data
		[weights, meta] = pickle.load(open(fn, 'rb'), encoding='latin1') #currently, this class MUST be initialized from a pickle file
		
        # list to store labels
        self.vocab = meta["vocab"]
        
        # define how many rows and columns in an image
        self.img_rows = meta["img_side"] ; self.img_cols = meta["img_side"]
        
        # load the convolutional network using LiteCNN function
        self.CNN = LiteCNN()
        
        # Assuming weve already trained our network we load it with the saved weights from pickle file
        self.CNN.load_weights(weights)
        
        # Define the pooling layers size 
        self.CNN.pool_size = int(pool_size)
        
    # predict function for a new image, outputs prediction probability
    def predict(self,image):
        print(image.shape)
        
        # Reshape the image so it's the correct size to do the dot product between
        # the image and the first layer of the CNN
        X = np.reshape(image,(1,1,self.img_rows,self,img_cols))
        X = X.astype("float32")
        
        # Make prediction
        predicted_i = self.CNN.predict(X)
        
        # return predicted labels
        return self.vocab[predicted_i]


# Convolutional Neural Network class
class LiteCNN:
    
    def _init_(self): # initialize two lists, store layers weve learned (weights of each layer), and size of pooling area 
        
        # store the layers
        self.layers = []
        
        # size of pooling area for ma pooling
        self.pool_size = None
        
    def load_weights(self,weights): # load weights from pickle file
        assert not self.layers, "Weights can only be loaded once"
        
        # Add saved matrix values to the convolutional nerual network
        for k in range(len(weights.keys())):
            self.layers.append(weights['layer_{}'.format(k)])
    
    
    def predict(self,X): # Where all the magic happens
        # Feed input though all layers
        # If using a library like Keras, this function would be all you needed to do
        
        # Ordering within block can be different, ordering of layers, will get different output
        # but dont need to use this order of layering. 
        
        # First layer is convolutional layer
        # first layer so zero-th layey, layer_i = 0
        h = self.cnn_layer(X, layer_i = 0, border_mode = "full")
        X = h 
        
        # Next layer is activation RELU
        # Apply activation to the output of the previous layer, so input to this is output of previous layer
        h = self.relu_layer(X)
        X = h 
        
        # Now do another convolutional layer
        # input to this is output of previous relu
        h = self.cnn_layer(X,layer_i = 2, border_mode="valid")
        X = h
        
        # Add another activation, non-linearity layer
        h = self.relu_layer(X)
        X = h 
        
        # Now add a max pooling layer, get most relevant features 
        h = self.maxpooling_layer(X)
        X = h
        
        # Add a dropout layer, to prevent overfitting
        chance = .25 # .25 percent change that neuron will be de-activated, set to zero
        h = self.dropout_layer(X, chance)
        X = h
        
        # Classification part after feature learning part:
        # Flatten layer
        h = self.flatten_layer(X,layer_i = 7)
        X = h
        
        # Add a fully connected dense layer
        h = self.dense_layer(X,fully,layer_i = 10)
        X = h
        
        # Add softmax to get probabilities of classes
        h = self.softmax_layer2D(X)
        X = h
        
        # Take max probability from softmax output
        max_i = self.classify(X)
        
        # Return the classified class
        return max_i[0]
    
    
    # DEFINE ALL FUNCTIONS USED IN THE PREDICTION FUNCTION
    
    # Convolutional layer function
    def 
        
        
        
        
        
        
        
        
        
         