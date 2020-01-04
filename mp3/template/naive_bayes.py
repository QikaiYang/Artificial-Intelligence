# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import nltk
import numpy as np
import math

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)
    """
    # TODO: Write your code here
    log_p_positive = 0
    log_p_negative = 0
    count_both = {}
    count_positive = {}
    count_negative = {}
    pro_positive = {}
    pro_negative = {}
    positive_count = 0
    negative_count = 0
    for i in range(len(train_labels)):
        if (train_labels[i]==1):
            log_p_positive += 1.0
        else:
            log_p_negative += 1.0
    log_p_positive = math.log(log_p_positive) # log(P(positive))
    log_p_negative = math.log(log_p_negative) # log(P(negative))
    #formulate the frequency calculator 
    for sentence in range(len(train_set)):
        for word in train_set[sentence]:
            if word not in count_both:
                count_both[word] = 1.0
                if(train_labels[sentence] == 1):
                    positive_count += 1
                    count_positive[word] = 1.0
                    count_negative[word] = 0.0
                else:
                    negative_count += 1
                    count_positive[word] = 0.0
                    count_negative[word] = 1.0
            else:
                count_both[word] += 1.0
                if(train_labels[sentence] == 1):
                    positive_count += 1
                    count_positive[word] += 1.0
                    count_negative[word] += 0.0
                else:
                    negative_count += 1
                    count_positive[word] += 0.0
                    count_negative[word] += 1.0
    #get the probability of different words
    for word in count_both:
        pro_positive[word] = math.log((count_positive[word]+smoothing_parameter)/(positive_count+smoothing_parameter*count_both[word])*1.0)
        pro_negative[word] = math.log((count_negative[word]+smoothing_parameter)/(negative_count+smoothing_parameter*count_both[word])*1.0)
    #finally test the dev_set
    #------------------------------------------------------------
    lamada = 0.6
    #------------------------------------------------------------
    log_p_positive_ = 0
    log_p_negative_ = 0
    count_both_ = {}
    count_positive_ = {}
    count_negative_ = {}
    pro_positive_ = {}
    pro_negative_ = {}
    positive_count_ = 0
    negative_count_ = 0
    for i in range(len(train_labels)):
        if (train_labels[i]==1):
            log_p_positive_ += 1.0
        else:
            log_p_negative_ += 1.0
    log_p_positive_ = math.log(log_p_positive_) # log(P(positive))
    log_p_negative_ = math.log(log_p_negative_) # log(P(negative))
    for sentence in range(len(train_set)):
        for word in range(len(train_set[sentence])-1):
            if((train_set[sentence][word] + train_set[sentence][word+1]) not in count_both_):
                count_both_[(train_set[sentence][word] + train_set[sentence][word+1])] = 1.0
                if(train_labels[sentence] == 1):
                    positive_count_ += 1
                    count_positive_[(train_set[sentence][word] + train_set[sentence][word+1])] = 1.0
                    count_negative_[(train_set[sentence][word] + train_set[sentence][word+1])] = 0.0
                else:
                    negative_count_ += 1
                    count_positive_[(train_set[sentence][word] + train_set[sentence][word+1])] = 0.0
                    count_negative_[(train_set[sentence][word] + train_set[sentence][word+1])] = 1.0
            else:
                count_both_[(train_set[sentence][word] + train_set[sentence][word+1])] += 1.0
                if(train_labels[sentence] == 1):
                    positive_count_ += 1
                    count_positive_[(train_set[sentence][word] + train_set[sentence][word+1])] += 1.0
                    count_negative_[(train_set[sentence][word] + train_set[sentence][word+1])] += 0.0
                else:
                    negative_count_ += 1
                    count_positive_[(train_set[sentence][word] + train_set[sentence][word+1])] += 0.0
                    count_negative_[(train_set[sentence][word] + train_set[sentence][word+1])] += 1.0
    for word in count_both_:
        pro_positive_[word] = math.log((count_positive_[word]+smoothing_parameter)/(positive_count_+smoothing_parameter * count_both_[word])*1.0)
        pro_negative_[word] = math.log((count_negative_[word]+smoothing_parameter)/(negative_count_+smoothing_parameter * count_both_[word])*1.0)
    #------------------------------------------------------------
    result = []
    for sentence in range(len(dev_set)):
        p0_0 = log_p_negative
        p1_0 = log_p_positive
        for word in dev_set[sentence]:
            if word in count_both:
                p0_0 += pro_negative[word]
                p1_0 += pro_positive[word]
            else:
                p0_0 += math.log(smoothing_parameter / (negative_count + (smoothing_parameter*(len(count_negative)+1))))
                p1_0 += math.log(smoothing_parameter / (positive_count + (smoothing_parameter*(len(count_positive)+1))))
        #----------------------------------------------------------------------------------------------------------------
        p0_1 = log_p_negative_
        p1_1 = log_p_positive_
        for word in range(len(dev_set[sentence])-1):
            if (dev_set[sentence][word]+dev_set[sentence][word+1]) in count_both_:
                p0_0 += pro_negative_[(dev_set[sentence][word]+dev_set[sentence][word+1])]
                p1_0 += pro_positive_[(dev_set[sentence][word]+dev_set[sentence][word+1])]
            else:
                p0_0 += math.log(smoothing_parameter / (negative_count_ + (smoothing_parameter*(len(count_negative_)+1))))
                p1_0 += math.log(smoothing_parameter / (positive_count_ + (smoothing_parameter*(len(count_positive_)+1))))
        #----------------------------------------------------------------------------------------------------------------
        p1 = lamada*p1_0 + (1-lamada)*p1_1
        p0 = lamada*p0_0 + (1-lamada)*p0_1
        if(p1 > p0):
            result.append(1)
        else:
            result.append(0)
    return result
