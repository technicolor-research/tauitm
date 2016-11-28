#########################################################################################################################################
## Copyright (c) 2016 - Technicolor R&D France
## 
## The source code form of this Open Source Project components is subject to the terms of the Clear BSD license.
##
## You can redistribute it and/or modify it under the terms of the Clear BSD License (http://directory.fsf.org/wiki/License:ClearBSD)
##
## See LICENSE file for more details.
##
## This software project does also include third party Open Source Software: See data/LICENSE file for more details.
#########################################################################################################################################

import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sklearn.metrics
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

def similarity(account1,account2,household_size=2):
    """ Compute the similarity between 2 accounts and their assignements. They must both have the exact same movies. """
    perm=itertools.permutations(range(household_size)) # Get all equivalent user assignements
    max_correct=0
    for p in perm:
        correct=0 # number of time the assignement is right
        for a1,a2 in zip(account1[1],account2[1]): # compare all assignement
            if p[a1[1]]==a2[1]: # if the assignement is correct increment our counter
                correct+=1
        if max_correct < correct: # Take the best result
            max_correct=correct
    return max_correct/float(len(account1[1]))
        

# Still there for backward compatibility but prefer using the more general function stats
def stats_similarity(truth,predictions,household_size=2,cdf=True):
    """ Get stats about the prediction and plot a ROC curve """
    res={}
    for name,prediction in predictions.iteritems():
        similarities=[similarity(a,b,household_size) for a,b in zip(truth,prediction)]
        if cdf:
            similarities.sort()
            plt.plot([0.5]+similarities,[0.0]+list(np.arange(len(similarities))/float(len(similarities))),label=name)
        res[name]={'mean':np.mean(similarities),'std':np.std(similarities),'median':np.median(similarities),'min':min(similarities),'max':max(similarities)}
    if cdf:
        plt.axis([0.5, 1, 0, 1])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
    return res

def confusion(account1,account2):
    """Compute the confusion matrix of 2 assignements"""
    conf=defaultdict(int)
    for a1,a2 in zip(account1[1],account2[1]):
        conf[(a1[1]),a2[1]]+=1
    return conf

def purity(account1,account2,household_size=2):
    conf=confusion(account1,account2) # Get confusion matrix
    
    correct=0.
    for i in range(household_size): # For each line of the confusion matrix add the highest value to correct
        best_map_value=-float("inf")
        for j in range(household_size):
            if conf[(j,i)]>best_map_value:
                best_map_value=conf[(j,i)]
        correct+=best_map_value
    # Get the error
    return correct/sum(conf.values())

def mutual_information(account1,account2,household_size=2):
    return sklearn.metrics.adjusted_mutual_info_score([m[1] for m in account1[1]],[m[1] for m in account2[1]])

def adjusted_rand_index(account1,account2,household_size=2):
    return sklearn.metrics.adjusted_rand_score([m[1] for m in account1[1]],[m[1] for m in account2[1]])


def stats(truth,predictions,household_size=2,cdf=True,measure=similarity,shape=None,color=None):
    """ Get stats about the prediction and plot a ROC curve
The measure can be either a function or a string corresponding to one of the measure functions in this file"""
    if shape==None:
        shape=['-']*len(predictions)
    if color==None:
        color=['b','g','r','c','m','y','k']*int(len(predictions)/7+1)
    
    if isinstance(measure, str): # Get the right fucntion if we get a string
        if measure=="similarity":
            measure_function=similarity
        elif measure=="purity":
            measure_function=purity
        elif measure=="mi" or measure=="mutual_information" or measure=="ami":
            measure_function=mutual_information
        elif measure=="arand" or adjusted_rand_index:
            measure_function=adjusted_rand_index
    else:
        measure_function=measure
    res={}
    for i,(name,prediction) in enumerate(predictions.iteritems()):
        measures=[measure_function(a,b,household_size) for a,b in zip(truth,prediction)]
        if cdf:
            measures.sort()
            plt.plot(measures,list(np.arange(len(measures))/float(len(measures))),shape[i]+color[i],label=name)
            plt.ylabel('cdf')
            if isinstance(measure, str):
                plt.xlabel(measure)
        res[name]={'mean':np.mean(measures),'std':np.std(measures),'median':np.median(measures),'min':min(measures),'max':max(measures)}
    if cdf:
        if measure_function==similarity or measure_function==purity:
            plt.axis([min(min(measures),0.5), 1, 0, 1])
        else:
            plt.axis([-0.1, 1, 0, 1])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
    return res




def stats_composite(truth,predictions):
    results={}
    plt.figure()
    for name,prediction in predictions.iteritems():
        result={}
        fpr, tpr, _ = roc_curve(truth,prediction)
        auc=roc_auc_score(truth,prediction)
        plt.plot(fpr, tpr, label=name+' (area = %0.2f)' % auc)
        result['auc']=auc
        result['f1']=f1_score(truth, prediction>0.5)
        results[name]=result

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    return results





