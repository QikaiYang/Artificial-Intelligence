# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import math
import numpy as np
import random

def output_judge(parameters, vector, bias):
    result = parameters @ vector.T + bias 
    if result >= 0:
        return True
    else:
        return False

def transfer(num):
    if num == True:
        return 1
    else:
        return -1

def train_model(train_set, train_labels, learning_rate, max_iter, bias):
    parameters = np.zeros(len(train_set[0]))
    i = 0
    while(i < max_iter):
        order = [i for i in range(len(train_set))]
        real_rate = 1000/(1000+max_iter)*learning_rate
        for m in order:
            curr = m
            to_judge = output_judge(parameters, train_set[curr], bias)
            if(to_judge != train_labels[curr]):
                parameters += train_set[curr]*transfer(train_labels[curr])*real_rate
                bias += transfer(train_labels[curr])*real_rate
        i += 1
    return parameters,bias

def classify(train_set, train_labels, dev_set, learning_rate, max_iter):
    """
    train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
                This can be thought of as a list of 7500 vectors that are each
                3072 dimensional.  We have 3072 dimensions because there are
                each image is 32x32 and we have 3 color channels.
                So 32*32*3 = 3072
    train_labels - List of labels corresponding with images in train_set
    example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
             and X1 is a picture of a dog and X2 is a picture of an airplane.
             Then train_labels := [1,0] because X1 contains a picture of an animal
             and X2 contains no animals in the picture.

    dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
              It is the same format as train_set
    """
    #---------------training model------------------------------
    bias = 0.01
    parameters, bias = train_model(train_set, train_labels, learning_rate, max_iter, bias)
    print(bias)
    #---------------start predicting----------------------------
    result = []
    for i in range(len(dev_set)):
        result.append(output_judge(parameters, dev_set[i], bias))
    return result

def train_model_ec(train_set, train_labels, learning_rate, max_iter, bias):
    parameters = np.zeros(len(train_set[0]))
    i = 0
    while(i < max_iter):
        order = random.sample([i for i in range(len(train_set))], len(train_set))
        real_rate = 1000/(1000+max_iter)*learning_rate
        for m in order:
            curr = m
            to_judge = output_judge(parameters, train_set[curr], bias)
            if(to_judge != train_labels[curr]):
                parameters += train_set[curr]*transfer(train_labels[curr])*real_rate
                bias += transfer(train_labels[curr])*real_rate
        i += 1
    return parameters,bias

def classifyEC(train_set, train_labels, dev_set, learning_rate, max_iter):
    # Write your code here if you would like to attempt the extra credit
    seats = 31
    list_bias = np.array([0.01 for i in range(seats)])
    list_parameters = []
    for i in range(seats):
        parameters, list_bias[i] = train_model_ec(train_set, train_labels, learning_rate, max_iter, list_bias[i])
        list_parameters.append(parameters)
    result = []
    for i in range(len(dev_set)):
        record_0 = 0
        record_1 = 0
        for j in range(seats):
            cmp = output_judge(list_parameters[j], dev_set[i], list_bias[j])
            if cmp == True:
                record_1 += 1
            else:
                record_0 += 1
        if record_1 > record_0:
            result.append(True)
        else:
            result.append(False)  
    return result