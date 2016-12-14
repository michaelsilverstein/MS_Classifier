#Michael Silverstein
#Boston University
#BE 562 - Computational Biology
#Naive Bayes Classifier
#Uses either Naive Bayes (product of individual normal univariate distributions)
# or Non-Naive Bayes (multivariate distribution) for each class

import cPickle as pickle
import pandas as pd
import numpy as np
import math
pd.options.mode.chained_assignment = None  # default='warn'
import scipy

def data_prep(df,cols,p = 8./10):
    #Splits data into training and test sets
    #Input:
    #| df: All data - Pandas Dataframe
    #| cols: Columns of interest
    #| p: splitting parameter - what fraction of data will be place into the test set (default = 80%)
    #Output:
    #| train: Training set
    #| test: Test set
    df = df[cols]
    # Filter and organize data by class
    by_class = {}

    #Generate training set
    train = {}
    for label in labels:
        by_class[label] = df[df.Labels == label]
        train[label] = df[df.Labels == label].iloc[:int(len(df.Labels) * p), 3:]

    #Generate test set
    test = [pd.DataFrame(np.array(by_class[k])[int(len(by_class[k]) * p):, :]) for k in by_class.keys()]
    test = pd.concat(test)
    test.columns = cols
    return train, test

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
    #parameters = {'CLASS_1' : [mean_1 ,cov_1,std_1],...,'CLASS_N' : [mean_N, cov_N,std_N]}
    #|mean_N and std_N are vectors for class N
    #|cov_N is a covariance matrix for class N
    cov_matrices = cov_mat_calc(data)
    parameters = {}
    for label in labels:
        parameters[label] = [np.array(data[label].mean()),cov_matrices[label],np.array(data[label].std())]
    return parameters

def univariate(x,u,std):
    # Calculate P(x_i | Class, parameters)
    #|Input: x_i = sample observation, u = mean value of feature i for a given class,
    #| std = standard deviation of a feature i
    #|Output:
    #|f(x_i) = exp(-(x-u)^2/[2*std^2])
    #|         -----------------------
    #|            sqrt(2*pi*std^2)
    var = math.pow(std,2)
    numerator = math.exp(-math.pow((x-u),2)/(2*var))
    normalization = 1/math.sqrt(2*var*math.pi)
    return numerator*normalization

def multivariate(x,u,cov):
    #Calculate P(X | Class, parameters)
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    #|Input: x = sample vector, u = mean vector for all features of a given class, cov = covariance matrix of features
    #| of a given class
    #|Output:
    #|f_x(x_1,...,x_k) = e^(-1/2 * (x-u).T * inv(cov) * (x-u))
    #|                   ------------------------------------
    #|                           sqrt([2*pi]^k*det(cov)))
    cov = np.matrix(cov)
    det_cov = np.linalg.det(cov)
    x_mu = np.matrix(x-u)
    numerator = math.exp(-0.5*x_mu*cov.I*x_mu.T)
    normalization = 1/(math.sqrt(math.pow(2*math.pi,len(x))*det_cov))
    return numerator*normalization

def naivebayes(data,parameters,priors,method='m'):
    #| G(x) = choose(P(X_i|Class,params)P(X_i))
    #|Input: data = numpy array of data, parameters = dictionary of class parameters (mean and respective variance)
    #| priors = dictionary of class prior probabilities, method = 'm' for Multivariate (Non-Naive Bayes) or
    #| 'n' for Naive Bayes classification
    #|Output: predicted_labels = vector of predicted labels for each inputted data

    predicted_labels = []
    for x in data:
        probs_by_class = {}
        for label in labels:
            u = parameters[label][0] #Mean vector for given class
            cov = parameters[label][1] #Covariance matrix for given class
            std = parameters[label][2] #Standard deviation
            if method == 'm': #Multivariate distribution (Non-Naive Bayes)
                probs_by_class[label] = multivariate(x,u,cov)*priors[label]
            if method == 'n': #Prodcut of univariate distributions (Naive Bayes)
                probs_by_class[label] = \
                    reduce(lambda x,y:x*y,[univariate(x[i],u[i],std[i]) for i in range(len(x))])*priors[label]
            #"Fuzzify" - Assign label stochastically
        norm_probs = [probs_by_class[label]/sum(probs_by_class.values()) for label in labels]
        predicted_labels.append(np.random.choice(labels, p=norm_probs))

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
