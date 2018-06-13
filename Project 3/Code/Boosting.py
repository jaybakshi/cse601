
# coding: utf-8

# In[7]:

import numpy as np
import random
import math as math
import sys


# In[8]:

def getGiniIndex(samples, classes):
    total_length = float(0)
    gini = float(0)
    for item in samples:
        total_length += float(len(item))
        
    for sample in samples:
        if(len(sample)==0):
            continue
        sample_len  = float(len(sample))
        score = float(0)
        for i in classes:
            pr = (list(sample[:,-1]).count(i))/sample_len
            score += (pr * pr)
        gini += (1-score) * (sample_len/total_length)
    return gini


# In[9]:

def splitData(data, val, index):
    left, right = list(), list()
    for item in data:
        if(item[index] >= val):
            right.append(item)
        else:
            left.append(item)
    return np.array(left), np.array(right)


# In[10]:

def evalSplit(data):
    classes = np.unique(data[:,-1])
    splitIndex, splitVal, spiltSample, minscore = sys.maxsize, sys.maxsize, None, sys.maxsize
    
    for index in range(len(data[0])-1):
        for row in data:
            
            samples = list(splitData(data, row[index], index))
            giniScore = getGiniIndex(samples, classes)
            if(giniScore<minscore):
                splitIndex, splitVal, spiltSample, minscore = index, row[index], samples, giniScore
    return {'index':splitIndex, 'val':splitVal, 'sample':spiltSample, 'score':minscore}


# In[11]:

def splitNode(node):
    left, right = node['sample']
    
    if((left.size==0) or (right.size==0)):
        classes = list()
        if(left.size==0):
            classes = right[:,-1]  
        else:
            classes = left[:,-1]
            
        node['left'] = node['right'] = max(np.unique(classes), key = list(classes).count)
        return
    
    if (len(left)<=1):
        classes = left[:,-1]
        node['left'] =  max(np.unique(classes), key = list(classes).count)
    else:
        node['left'] = evalSplit(left)
        splitNode(node['left'])
    
    if(len(right)<=1):
        classes = right[:,-1]
        node['right'] = max(np.unique(classes), key = list(classes).count)
    else:
        node['right'] = evalSplit(right)
        splitNode(node['right'])


# In[12]:

def generateTree(data):
    root = evalSplit(data)
    splitNode(root)
    return root


# In[13]:

def print_tree(node, depth=0):
    if isinstance(node, dict):
        print(depth*'-', 'X', (node['index']+1),'<',node['val'])
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print(depth*'-', node)


# In[14]:

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['val']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']


# In[53]:

def main(traindata, testdata):
    predictions = list()
    actual = list(testdata[:,-1])
    tree = generateTree(traindata)
    for row in testdata:
        predictions.append(predict(tree, row))
    
#     print_tree(tree)
    w = [0]*len(predictions)
    for i in range(len(predictions)):
        w[i] = float(1/(len(predictions)))

    w = boosting(predictions, actual, w, 1, 0)
    return actual, predictions, w


# In[ ]:

def boosting(prediction, actual, w, error, iterations):
    terror = list()
    preverror = error
    
    for i in range(len(prediction)):
        if(actual[i]!=prediction[i]):
            error += w[i]
            terror.append(1)
        else:
            terror.append(0)
    
    error = error/sum(w)
    if((error == float(0)) or (error == preverror) or (iterations==10)):
        return w
    alpha = 0.5*math.log((1-error)/error)
    
    for i in range(len(w)):
        if terror[i]==1:
            w[i] = w[i]*math.exp(alpha)
        else:
            w[i] = w[i]*math.exp(-1*alpha)
    return boosting(prediction, actual, w, error, iterations+1)

class METRICS(object):
    """
    Input: Takes in two lists of size n with numeric values 0 or 1
           Can also handle numpy arrays
    Usage: metrics=METRICS(trueLables = list,generatedLabels=list)
    """

    def __init__(self, trueLabels, predictedLabels):
        self.true = trueLabels
        self.pred = predictedLabels
        self.a = self.b = self.c = self.d = 0
        for i, j in zip(self.true, self.pred):
            if ((j == 1) & (i == j)):
                self.a += 1
            elif ((j == 0) & (i != j)):
                self.b += 1
            elif ((j == 1) & (i != j)):
                self.c += 1
            elif ((j == 0) & (i == j)):
                self.d += 1

    def accuracy(self):
        self.accuracy = (self.a + self.d) / (self.a + self.b + self.c + self.d)
        return self.accuracy

    def precision(self):
        self.precision = (self.a) / (self.a + self.c)
        return self.accuracy

    def recall(self):
        self.recall = (self.a) / (self.a + self.b)
        return self.recall

    def f1(self):
        self.f1 = (2 * self.a) / (2 * self.a + self.b + self.c)
        return self.f1



trainSet=sys.argv[1]

trainData=[i.rstrip().split() for i in open(trainSet, 'r').readlines()]
trainData=np.array(trainData)


try:
    testData = [i.rstrip().split() for i in open(testSet, 'r').readlines()]
    testData = np.array(testData)
    testSet = sys.argv[2]
    K = 1
except:
    K = 10
    pass

# In[15]:

accuracies = np.zeros(shape=(K,))
precision = np.zeros(shape=(K,))
recall = np.zeros(shape=(K,))
f1 = np.zeros(shape=(K,))

# In[16]:
perm = np.random.permutation(len(trainData))
for i in range(K):
    trainingData = []
    trainingLabels = []
    validationData = []
    validationLabels = []
    validationDataStart = int(i * len(perm) // 10)
    validationDataEnd = int((i + 1) * len(perm) // 10)
    for j in range(len(trainData)):
        if (j in range(validationDataStart, validationDataEnd)):
            validationData.append(trainData[perm[j]])
            validationLabels.append(trainData[perm[j]][-1])
        else:
            trainingData.append(trainData[perm[j]])
            trainingLabels.append(trainData[perm[j]][-1])
    if (K == 1):
        trainingData = np.array(trainingData)
        validationData = testData
        trainingLabels = np.array(trainingData)[:, -1]
        validationLabels = testData[:, -1]
    else:
        trainingData = np.array(trainingData)
        validationData = np.array(validationData)

    print("\nIteration " + str(i))
    print("Training data: " + str(len(trainingData)))
    print("Validation data: " + str(len(validationData)))
    trueLabels, predictedLabels, weight = main(traindata=trainingData, testdata=validationData)
    mt = METRICS(np.array(trueLabels).astype(int), np.array(predictedLabels).astype(int))
    try:
        accuracies[i] = mt.accuracy()
    except:
        print("Accuracy cannot be computed")
    try:
        precision[i] = mt.precision()
    except:
        print("Precision cannot be computed")
    try:
        recall[i] = mt.recall()
    except:
        print("Recall cannot be computed")
    try:
        f1[i] = mt.f1()
    except:
        print("F1 measure cannot be computed")

print(weights)
print("\n")
print("Average accuracy: " + str(accuracies.mean()))
print("Average precision: " + str(precision.mean()))
print("Average recall: " + str(recall.mean()))
print("Average F1-measure: " + str(f1.mean()))

