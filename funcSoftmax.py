#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:57:10 2017

@author: steve
"""
import numpy as np

# Sigmoid Activation Function
def sigmoid(x):
    sig = 1/(1+np.exp(-x))
    return sig

# Derivative of Sigmoid
def dSigmoid(x):
    dsig  = sigmoid(x) * (1-sigmoid(x))
    return dsig

# Softmax function
def softmax(x):
    maxs = np.amax(x)
    expScores = np.exp(x-maxs)
    out = expScores / np.sum(expScores, axis = 0, keepdims = True) 
    return out
                
        
# Backpropogation functions
    
# Fully Connected to Classify
def AutoDiff_CL(partialL_Z,prevOut,summedVals):
    
    # partialL_Z is partial Loss / partial summedVals where summed vals 
    # is the Z before activation that leads to yhat, the summedValues of classification layer
    dB = partialL_Z   
    dW = np.dot(partialL_Z, prevOut.transpose())

    return dB, dW, partialL_Z


# Pooling to Fully Connected
def AutoDiff_FC(partialL_Z, prevWeights, prevOut, summedValues):
    
    # sigmoid derivative of summed values from full connected layer
    # partial a1 partial z1 (see what a1 and z1 are in comment below)
    deltaSig = dSigmoid(summedValues)
    
    # Partial L partial Z1, where z1 are summed values of fully connected layer,
    # is equal to pL/pyhat * pyhat/pz2 * pz2/pa1 * pa1/pz1
    # yhat is prediction from output, z2 is summedVals of classify layer, a1 is output of fully connected
    # z1 is summed vals of fully connected. a1 = sigmoid (z1)
    # partial z2 partial a1 is just W2 since z2 = W2.transpose*a1 + b2
    # The partialL_Z inputted to the function is partial L partial Z2
    partialLZ = np.dot(prevWeights.transpose(),partialL_Z) * deltaSig
    
    # Define partial L partial Bias
    dB = partialLZ
    
    # Define partial L partial Weight
    d0, d1, d2 = prevOut.shape
    prevOut = prevOut.reshape((1,d0*d1*d2))
    dW = np.dot(partialLZ,prevOut)
    dW = dW.reshape((partialLZ.shape[0], d0,d1,d2))

    
    return dB, dW, partialLZ


def AutoDiff_PL(partialL_Z, prevWeights, prevOut, maxIndices, poolSize, output):
    # partialL_Z is partial L / partial Z where Z is summed values of fully connected layer
    # partialL_Z = (yhat-y)*dSigmoid(z2) * Weight_cl.transpose() * dSigmoid(z1)
    
    # Previous weights are weights from fully connected layer
    # prevOut is output of the convolutional layer
    # maxIndicies are the indices of the max values from pooling, remember we kept track of those
    # poolSize is the same, 2x2, output is the output of the fully connected layer
    # output is the output of the pooling layer
    
    # Get the shape of the output
    # The output of the pooling layer (right now with mnist data) is 20,12,12
    # Remember there were 20 filters with 5x5 filter size so the output of convolution was 
    # 20,24,24 (20 deep, and the windows size is 28(imsize) - 4)
    
    # So yea input to pooling was 20x24x24, it took 2x2 sections and returned 1 value for each section
    # So 24 * 24 is 576 total values, 576/4 is 144, so our window was changed from 24x24 to 12x12
    # Depth stays so output of pooling 20x12x12
    
    # Get the 20x12x12 output
    x,y,z = output.shape

    # Get shape of the fully connected layer weights, 30x20x12x12. THe weights are the smae
    # shape as the output of pooling, and there are 30 of them
    a,b,c,d = prevWeights.shape
    
    # Reshape to flatten the volume of values into one long array, sot he weights goes from 
    # Thirty 20x12x12 volumes to a 30 x 2880 matrix, each weight is a row
    prevWeights = prevWeights.reshape((a,b*c*d))
    
    # ALso reshape output of pooling from 20x12x12 to 2880x1
    # With this shape we have weights 30x2880 and poolingOutput/FC_input as 2880x1
    # Which leads to the 30x1 output of fully connected layer
    output = output.reshape((x*y*z,1))
    
    # Max indices whape was 20x12x12x2, flatten the windows to make it 20x144x2
    maxIndices = maxIndices.reshape((x,y*z,2))
    
    # Get derivative of sigmoid(poolOutput). This is partial activation/ partial z
    # since the output of a layer a = sigmoid(z)
    sp = dSigmoid(output)
    
    # Set partialL/partialZ where Z is the output of pool layer before activation
    # We have FC_out = Weight_fc.transpose * poolOut + biasFc
    # So partial L/partialZ = partialL/partialZ_fc * partialZ_fc/partialZ_pool
    # since partialZ_fc/partialZ_pool = partialZ_fc/partialOutputpool * partialOutputpool/partialZ_pool
    # and outputpool = activation(z_pool)
    # so we do Weights_fc trapnspose * partialL/partialZ_fc * partialZ_fc/partialpoolOut * partialpoolOut/partialZ_pool
    partialL_Z = np.dot(prevWeights.transpose(),partialL_Z) * sp
    
    # Flatten the partial and the pool output to 20x144, so keep depth, flatten length width window
    partialL_Z = partialL_Z.reshape((x,y*z))
    output = output.reshape((x,y*z))
    
    # Get shape of convolutional layer output, 20x24x24
    depth, height, width = prevOut.shape
    
    # Initialize partial L/partialConvout. Needs to be same shape as conv output
    partialNew = np.zeros((depth, height, width))
    
    # Loop through every filter
    for d in range(depth):
        
        # Row and column to slide through conv output
        row = 0
        col = 0
        
        # We had changed maxIndices to 20x144x2
        # Loop throough every value that makes up the 12x12 square filter for the current filter d
        for i in range(maxIndices.shape[1]):
            
            # Get the section of the Conv output currently covered by our 2x2 pool filter
            section = prevOut[d][row:row + poolSize[0], col:col + poolSize[0]]
            
            # Get partial values for the current section
            # output of pool to get max val of this section, for the max val (or vals) of this section,
            # set its partialL value to partialL_Zpool of outputPool layer (since this is part of the poolOutput)
            partialPool = getPartialSec(output[d][i], partialL_Z[d][i], section)
            
            # Update partial values for the current section
            partialNew[d][row:row + poolSize[0], col:col + poolSize[0]] = partialPool
            
            # Slide over horizontally
            col += poolSize[1]
            
            # if were at edge of conv output go back and move down
            if col >= width:
                col = 0 # go back to left of conv output
                row += poolSize[1] # move down
    
    # Return the partial, which has zero values for vals that werent the max of the section
    return partialNew

def AutoDiff_ConvL(partialL_Z, prevWeights, stride, im, summedValues):
   #  partialL_Z is partial of convolutional output, prevWeights is conv weights, 
   # stride is stride, im is input image, summedVals is output of conv
   
   # Weights are 20x1x5x5
    numFilters, depth, filterSize, filterSize = prevWeights.shape
    
    # Initialize partial bias and partial weights, same shape as bias and weights
    deltaB = np.zeros((numFilters,1))
    deltaW = np.zeros((prevWeights.shape))
    
    # Get number of vals in conv output, the height, width output (which has depth d), 
    # conv output is 20,24,24
    convOutNum = (partialL_Z.shape[1]) * (partialL_Z.shape[2])

    # Reshape partialL/partialZ to be  20x576, basically flatten the height, width portion of output
    partialL_Z = partialL_Z.reshape((partialL_Z.shape[0], partialL_Z.shape[1] * partialL_Z.shape[2]))

    # Loop throught the 20 filters
    for i in range(numFilters):
        
        # Row and column to slide over image
        row = 0
        col = 0
        
        # Loop thorough every value of conv output (for this filter i)
        for j in range(convOutNum):
            
            # section of image for which the filter is currently over
            sec = im[:,row:row + filterSize, col:col + filterSize]
            
            # partialL_Z[i][j] is single value corresponding to output partial, which is now 20x576
            # sec is the section of the image that was multiplied by the weight to get the output value
            # at [i][j] (so map output value that came from current section).
            # convOut = Weight_conv transpose * image + bias, so partial convOut/partial Weight_conv
            # is equal to image. So partialL/partialWeightconv = partialL/partialConvout * partialCOnvout/partialW_conv
            # which equals = partialL_Z * sectionOfImage
            deltaW[i] += sec * partialL_Z[i][j]
            
            # With convOut = Weight_conv transpose * image + bias, partial Convout/partial bias
            # just equals 1, so partialL/partialBiasConv is just partialL/partialConvout, which is partialL_Z (what we solved for in the AutoDiff PL function)
            deltaB[i] += partialL_Z[i][j]
            
            # Move over horizontally
            col += stride
            
            # If at end move back to the left and move down
            if (col + filterSize) - stride >= im.shape[2]:
                col = 0 # back left
                row += stride # move down
    
    # delta B is flat array, numFiltersx1
    # deltaW is 20x1x5x5, same as weights of course
    return deltaB, deltaW
    

# Get partial values of Convolution output, partialL/partialConvOut
# 1st arg is pooling output[dth filter][ith incides (from maxIndices)]  (ith of 144)
# 2nd arg is partialL/partialZ where Z is output of pool Layer
# 3rd arg is current section of of conv output layer (all of which go to the same max value)
def getPartialSec(val, partialL_Z, section):
    
    # Get shape of section, 2x2
    dim1, dim2 = section.shape
    
    # Flatten to 1x4
    section = section.reshape((dim1 * dim2))
    
    # initialize partial of section, same shape as section
    partialSection = np.zeros((section.shape))
    #print(' ')
    # loop through section
    for i in range(len(section)):
        
        # Get value of ith number in section
        num = section[i]
        
        # 3 values in each section were not used to their partial derivatives will be zero
        # 1 value was used so the partial derivative from pool output was from that value so 
        # set it equal to the corresponding partial from pool output
        # If it's less than the max val of the section
        if num < val:
            
            # Set it equal to zero
            partialSection[i] = 0
        else:
            
            # set it equal to the partialL/partialZ from the output of pool layer
            partialSection[i] = partialL_Z
    
       # print(partialSection[i])
        
    # return the partial section (three zeros and one value, unless there was a tie then they all get partial values)
    return partialSection.reshape((dim1, dim2))

        
        
        
        