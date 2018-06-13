# coding: utf-8

# # Import libraries

# In[1]:

import numpy as np
import collections
import sys

# # Import dataset

# In[2]:

k = eval(sys.argv[1])

trainSet = sys.argv[2]

# In[8]:
try:
    testSet = sys.argv[3]
    testData = [i.rstrip().split() for i in open(testSet, 'r').readlines()]
    testData = np.array(testData)
except:
    pass

# In[3]:

trainData = [i.rstrip().split() for i in open(trainSet, 'r').readlines()]

# In[4]:

trainData = np.array(trainData)


# # Helper Functions

# In[8]:

def isString(data):
    try:
        float(data)
        return False
    except:
        pass
        return True


# In[9]:

def distance(a, b):
    dist = []
    for i in range(len(a)):
        if (isString(a[i])):
            if (a[i] == b[i]):
                dist.append(0)
            else:
                dist.append(1)
        else:
            dist.append(eval(a[i]) - eval(b[i]))
            #     np.square(np.array(dist)).sum()**(0.5)
    return np.square(np.array(dist)).sum() ** (0.5)


# In[10]:

def knn(trainingData, validationData, trainingLabels, k):
    validationLabels = []
    for i in validationData:
        count = collections.Counter(
            [trainingLabels[j] for j in np.array([distance(i, j) for j in trainingData]).argsort()[:k]])
        validationLabels.append(count.most_common()[0][0])
    return validationLabels


# In[11]:

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


# # Normalize Data

# In[12]:

def normalizeData(data):
    for i in range(data.shape[1] - 1):
        if (not (isString(data[0][i]))):
            data[:, i] = (data[:, i].astype(float) - data[:, i].astype(float).min()) / (
            data[:, i].astype(float).max() - data[:, i].astype(float).min())
    return data


# # Split Training and testing data by K-fold implementation

# In[13]:

perm = np.random.permutation(len(trainData))
K=10
accuracies = np.zeros(shape=(K,))
precision = np.zeros(shape=(K,))
recall = np.zeros(shape=(K,))
f1 = np.zeros(shape=(K,))
# In[14]:

print("k: " + str(k))
for i in range(K):
    trainingData = []
    trainingLabels = []
    validationData = []
    validationLabels = []
    validationDataStart = int(i * len(perm) // 10)
    validationDataEnd = int((i + 1) * len(perm) // 10)
    for j in range(len(trainData)):
        if (j in range(validationDataStart, validationDataEnd)):
            validationData.append(trainData[perm[j]][:-1])
            validationLabels.append(trainData[perm[j]][-1])
        else:
            trainingData.append(trainData[perm[j]][:-1])
            trainingLabels.append(trainData[perm[j]][-1])
    if (K == 1):
        trainingData = normalizeData(trainData)
        validationData = normalizeData(testData)
        trainingLabels = trainData[:, -1]
        validationLabels = testData[:, -1]
    else:
        trainingData = np.array(trainingData)
        validationData = np.array(validationData)

    print("\nIteration " + str(i))
    print("Training data: " + str(len(trainingData)))
    print("Validation data: " + str(len(validationData)))
    predictedLabels = np.array(knn(trainingData, validationData, trainingLabels, k)).astype(int)
    trueLabels = np.array(validationLabels).astype(int)
    mt = METRICS(trueLabels, predictedLabels)
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

try:
    testSet
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

print("k: " + str(k))
for i in range(K):
    trainingData = []
    trainingLabels = []
    validationData = []
    validationLabels = []
    validationDataStart = int(i * len(perm) // 10)
    validationDataEnd = int((i + 1) * len(perm) // 10)
    for j in range(len(trainData)):
        if (j in range(validationDataStart, validationDataEnd)):
            validationData.append(trainData[perm[j]][:-1])
            validationLabels.append(trainData[perm[j]][-1])
        else:
            trainingData.append(trainData[perm[j]][:-1])
            trainingLabels.append(trainData[perm[j]][-1])
    if (K == 1):
        trainingData = normalizeData(trainData)
        validationData = normalizeData(testData)
        trainingLabels = trainData[:, -1]
        validationLabels = testData[:, -1]
    else:
        trainingData = np.array(trainingData)
        validationData = np.array(validationData)

    print("\nIteration " + str(i))
    print("Training data: " + str(len(trainingData)))
    print("Validation data: " + str(len(validationData)))
    predictedLabels = np.array(knn(trainingData, validationData, trainingLabels, k)).astype(int)
    trueLabels = np.array(validationLabels).astype(int)
    mt = METRICS(trueLabels, predictedLabels)
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