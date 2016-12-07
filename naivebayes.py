#Naive Bayes Classifier

import cPickle as pickle
import pandas as pd
import numpy as np
import math

def normalize(df,columns):
    #data = Pandas dataframe
    #Columns = columns containing Data
    index = df.index
    cols = df.columns
    data = df[columns].as_matrix()
    data = data.astype(float)
    err_states = np.seterr(divide='raise')
    ignored_states = np.seterr(**err_states)
    a = np.array([np.divide(data[i],np.nansum(data[i])) for i in range(len(data))])
    new_df = pd.DataFrame(data=a,index=index,columns=columns)
    other_labels = list(set(cols.values.tolist()) - set(columns))
    new_df[other_labels] = df[other_labels]
    new_df = new_df[cols] #Rearrange columns
    return new_df

def gaussian(x,mean,std):
    #Gaussian distribtuion
    return math.exp(-0.5*math.pow((x-mean)/std,2))

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
    #Calculates parameters (mean, std) for each feature for each class
    #parameters = {'CLASS1' : [[mean1,std1],[mean2,std2],...,[meanN,stdN],CLASS2 : [...],...'CLASSn':[...]}
    by_class = [data[data.Labels==label]for label in labels]
    means = [by_class[i].mean() for i in range(len(by_class))]
    stds =   [by_class[i].std() for i in range(len(by_class))]
    parameters = {}
    for i in range(len(labels)):
        parameters[labels[i]] = [[means[i][x],stds[i][x]] for x in range(len(means[0]))]
    return parameters

def naivebayes(data,parameters,priors):
    # G(x) = argmax(P(X_i|Class,params)P(X_i))
    predicted_labels = []
    for i in range(len(data)):
        probs_by_class = {}
        for label in labels:
            p = [gaussian(x,parameters[label][feature][0],parameters[label][feature][1])
                 for x in data.iloc[i,:] for feature in range(len(cols))]
            probs_by_class[label] = np.prod(p)*priors[label]
        predicted_labels.append(max(probs_by_class,key=probs_by_class.get))
    return predicted_labels

def accuracy_calculator(data):
    correct = data['Labels']
    predictions = data['Predictions']
    total = len(correct)
    count = 0
    for i in range(len(correct)):
        if correct[i] == predictions[i]:
            count += 1
    return float(count)/total
############
#   MAIN   #
############
df = pickle.load(open('GeneralLabeleddata_NOTnormalized.p','rb'))
cols = df.columns.values.tolist()[3:]
labels = list(set(df['Labels'])) ##GLOBAL
# df = normalize(df,cols)
df = df.fillna(value=0)
parameters = class_parameterizer(df)

priors = priors_calc(df)
data = df.iloc[:,3:]
predictions = naivebayes(data,parameters,priors)
df['Predictions'] = predictions
accuracy = accuracy_calculator(df)
print accuracy
