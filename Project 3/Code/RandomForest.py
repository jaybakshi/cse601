
# coding: utf-8

# In[19]:

import numpy as np
import random as random
import sys


# In[20]:

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


# In[21]:

def splitData(data, val, index):
    left, right = list(), list()
    for item in data:
        if(item[index] >= val):
            right.append(item)
        else:
            left.append(item)
    return np.array(left), np.array(right)


# In[35]:

def evalSplit(data, num_features):
    classes = data[:,-1]
    splitIndex, splitVal, spiltSample, minscore = sys.maxsize, sys.maxsize, None, sys.maxsize
    features = list()
    
    while len(features)<num_features:
        colnum = random.randrange(len(data[0])-1)
        if colnum not in features:
            features.append(colnum)
        
    for col in features:
        for row in data:
            samples = list(splitData(data, row[col], col))
            giniScore = getGiniIndex(samples, classes)
            if(giniScore<minscore):
                splitIndex, splitVal, spiltSample, minscore = col, row[col], samples, giniScore
                    
    return {'index':splitIndex, 'val':splitVal, 'sample':spiltSample, 'score':minscore}
    


# In[39]:

def splitNode(node, num_features):
    left, right = node['sample']
    
    if((left.size==0) or (right.size==0)):
        classes = list()
        if(left.size==0):
            classes = right[:,-1]  
        else:
            classes = left[:,-1]
            
#         print('classes', max(np.unique(classes), key = list(classes).count))
        node['left'] = node['right'] = max(np.unique(classes), key = list(classes).count)
        return
    
    if (len(left)<=1):
        classes = left[:,-1]
#         print('classes', classes, max(np.unique(classes), key = classes))
        node['left'] =  max(np.unique(classes), key = list(classes).count)
    else:
        node['left'] = evalSplit(left, num_features)
        splitNode(node['left'], num_features)
    
    if(len(right)<=1):
        classes = right[:,-1]
        node['right'] = max(np.unique(classes), key = list(classes).count)
    else:
        node['right'] = evalSplit(right, num_features)
        splitNode(node['right'], num_features)


# In[24]:

def generateTree(data, num_features):
    root = evalSplit(data, num_features)
    splitNode(root, num_features)
    return root


# In[25]:

def getSample(data, sample_size):
    sample = list()
    while(len(sample)<sample_size):
        i = random.randrange(len(data))
        sample.append(data[i])
    return np.array(sample)


# In[44]:

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


# In[83]:

def randomForest(traindata, testdata, num_trees):
    forest = list()
    pred = list()
    num_features = int((len(traindata[0])-1)**0.5)
    sample_size = int(len(traindata)/num_trees)
    for i in range(0, num_trees):
        sample = getSample(traindata, sample_size)
        tree = generateTree(sample, num_features)
        forest.append(tree)

    for row in testdata:
        cur_pred = [predict(tree, row) for tree in forest]
        pred.append(max(np.unique(cur_pred), key = list(cur_pred).count))

    actual = list(testdata[:,-1])
    return actual, pred

# print(trueLabels)
# print(predictedLabels)

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


numTrees=eval(sys.argv[1])
trainSet=sys.argv[2]

trainData=[i.rstrip().split() for i in open(trainSet, 'r').readlines()]
trainData=np.array(trainData)


try:
    testData = [i.rstrip().split() for i in open(testSet, 'r').readlines()]
    testData = np.array(testData)
    testSet = sys.argv[3]
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
        trainingData = trainData
        validationData = testData
        trainingLabels = trainData[:, -1]
        validationLabels = testData[:, -1]
    else:
        trainingData = np.array(trainingData)
        validationData = np.array(validationData)

    print("\nIteration " + str(i))
    print("Training data: " + str(len(trainingData)))
    print("Validation data: " + str(len(validationData)))
    trueLabels, predictedLabels = randomForest(traindata=trainingData, testdata=validationData, num_trees=numTrees)
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

print("\n")
print("Average accuracy: " + str(accuracies.mean()))
print("Average precision: " + str(precision.mean()))
print("Average recall: " + str(recall.mean()))
print("Average F1-measure: " + str(f1.mean()))
