#Michael Silverstein
#Boston University
#BE 562 - Computational Biology
#Naive Bayes Classifier

import cPickle as pickle
import pandas as pd
import numpy as np
import math
pd.options.mode.chained_assignment = None  # default='warn'


def cov_mat_calc(data):
    #Calculates covariance matrix
    #|Input: Data BY CLASS
    #|Output: Dictionary of covariance matrix by class
    cov_matrices = {}
    for label in labels:
        d = data[label]
        #https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Estimation_of_parameters
        cov_matrices[label] = float(1)/len(d)*np.sum(np.array(np.dot(np.array([x-d.mean()]).T,np.array([x-d.mean()])))
                                                     for x in np.array(d))
    return cov_matrices

def priors_calc(data):
    #Calculates priors based off of frequency in data (counting)
    labels_list = data['Labels'].tolist()
    total = len(labels_list)
    priors = {}
    for i in labels_list:
        priors[i] = priors.get(i,0)+1
    for k,v in priors.items():
        priors[k] = float(v)/total
    return priors

def class_parameterizer(data):
    #Calculates parameters (mean, cov) for each feature for each class
    #parameters = {'CLASS1' : [mean1 ,cov1],...,'CLASS_N' : [mean_N, cov_N]}
    cov_matrices = cov_mat_calc(data)
    parameters = {}
    for label in labels:
        parameters[label] = [np.array(data[label].mean()),cov_matrices[label]]
    return parameters

def multi_variate(x,u,cov):
    #Calculate P(Class | X, parameters)
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    #| u = mean, cov = covariance matrix
    #|f_x(x_1,...,x_k) = e^(-1/2 * (x-u).T * inv(cov) * (x-u))
    #|                   ------------------------------------
    #|                               sqrt([2*pi]^k*det(cov)))
    cov = np.matrix(cov)
    det_cov = np.linalg.det(cov)
    x_mu = np.matrix(x-u)
    numerator = math.exp(-0.5*x_mu*cov.I*x_mu.T)
    normalization = 1/(math.sqrt(math.pow(2*math.pi,len(x))*det_cov))
    return numerator*normalization

def naivebayes(data,parameters,priors,fuzzy=True):
    # G(x) = argmax(P(X_i|Class,params)P(X_i))
    predicted_labels = []
    for x in data:
        probs_by_class = {}
        for label in labels:
            probs_by_class[label] = multi_variate(x,parameters[label][0],parameters[label][1])*priors[label]
        if fuzzy:
            norm_probs = [probs_by_class[label]/sum(probs_by_class.values()) for label in labels]
            predicted_labels.append(np.random.choice(labels,p=norm_probs))
        else:
            predicted_labels.append(max(probs_by_class,key=probs_by_class.get))
    return predicted_labels

def accuracy_calculator(data):
    correct = data['Labels'].tolist()
    predictions = data['Predictions'].tolist()
    total = len(correct)
    count = 0
    for i in range(len(correct)):
        if correct[i] == predictions[i]:
            count += 1
    return float(count)/total
