
# coding: utf-8

# # Import libraries

# In[95]:

import numpy as np
import collections
import sys


# # Import dataset

# In[99]:

file=sys.argv[1]
test=False
try:
    testData=(sys.argv[2]).split()
    print(testData)
    test = True
except:
    pass

# In[100]:

# data=[i.rstrip().split() for i in open(input(), 'r').readlines()]
data=[i.rstrip().split() for i in open(file, 'r').readlines()]


# In[101]:

data=np.array(data)


# # Helper Functions

# In[85]:

def isString(data):
    try:
        float(data)
        return False
    except:
        pass
        return True


# # Metrics

# In[86]:

class METRICS(object):
    """
    Input: Takes in two lists of size n with numeric values 0 or 1
           Can also handle numpy arrays
    Usage: metrics=METRICS(trueLables = list,generatedLabels=list)
    """
    def __init__(self, trueLabels, predictedLabels):
        self.true = trueLabels
        self.pred = predictedLabels
        self.a=self.b=self.c=self.d=0
        for i,j in zip(self.true,self.pred):
            if((j==1)&(i==j)):
                self.a+=1
            elif((j==0)&(i!=j)):
                self.b+=1
            elif((j==1)&(i!=j)):
                self.c+=1
            elif((j==0)&(i==j)):
                self.d+=1
                
    def accuracy(self):
        self.accuracy=(self.a+self.d)/(self.a+self.b+self.c+self.d)
        return self.accuracy
    
    def precision(self):
        self.precision=(self.a)/(self.a+self.c)
        return self.accuracy
    
    def recall(self):
        self.recall=(self.a)/(self.a+self.b)
        return self.recall
    def f1(self):
        self.f1=(2*self.a)/(2*self.a+self.b+self.c)
        return self.f1


# # ============================================================

# # Split Nominal and Continuous Data

# In[87]:

def NaiveBayes(trainingData, validationData, trainingLabels, validationLabels):
    
    # Split Nominal and Continuous Data
    trainingDataContinuous=np.empty(shape=(trainingData.shape[0],0), dtype='float64')
    trainingDataNominal=np.empty(shape=(trainingData.shape[0],0), dtype='<U7')
    validationDataContinuous=np.empty(shape=(validationData.shape[0],0), dtype='float64')
    validationDataNominal=np.empty(shape=(validationData.shape[0],0), dtype='<U7')

    for i in range(trainingData.shape[1]-1):
        if(isString(trainingData[0,i])):
            trainingDataNominal=np.append(trainingDataNominal,trainingData[:,i].reshape(trainingData.shape[0],1),axis=1)
        else:
            trainingDataContinuous=np.append(trainingDataContinuous,trainingData[:,i].reshape(trainingData.shape[0],1).astype(float),axis=1)

    for i in range(validationData.shape[1]):
        if(isString(validationData[0,i])):
            validationDataNominal=np.append(validationDataNominal,validationData[:,i].reshape(validationData.shape[0],1),axis=1)
        else:
            validationDataContinuous=np.append(validationDataContinuous,validationData[:,i].reshape(validationData.shape[0],1).astype(float),axis=1)

    # Calculate means and standard deviation for Continuous Data

    means=np.zeros(shape=(len(np.unique(trainingLabels)),trainingDataContinuous.shape[1]))
    std=np.zeros(shape=(len(np.unique(trainingLabels)),trainingDataContinuous.shape[1]))

    classes=np.unique(np.array(trainingLabels))

    for i in range(len(classes)):            
        means[i]=trainingDataContinuous[np.array(trainingLabels)==classes[i]].mean(0)
        std[i]=trainingDataContinuous[np.array(trainingLabels)==classes[i]].std(0)

    # Calculate probabilities for Continuous

    # Calculate the probabilities for d attributes and c classes

    ec=np.exp((-1)*(validationDataContinuous[0] - means).dot((validationDataContinuous[0] - means).T)/(2*(std.dot(std.T))))

    post=(1/(((2*np.pi)**(0.5))*std)).T.dot(ec)

    # Multiply all the probabilities to get 1xc

    # Calculate all probabilities for continuous validation data

    valPostContinuous=np.empty(shape=(validationDataContinuous.shape[0],len(classes)))

    for i in range(len(validationDataContinuous)):
        ec=np.exp((-1)*(validationDataContinuous[i] - means).dot((validationDataContinuous[i] - means).T)/
                  (2*(std.dot(std.T))))
        post=(1/(((2*np.pi)**(0.5))*std)).T.dot(ec)
        valPostContinuous[i]=post.prod(0)

    # Dealing with Nominal Data

    # Calculate probabillities for Nominal Data

    valPostNominal=np.empty(shape=(validationDataNominal.shape[0],len(classes),validationDataNominal.shape[1]))

    for i in range(validationDataNominal.shape[0]):
        for j in range(len(classes)):
            for k in range(validationDataNominal.shape[1]):
                p=((trainingDataNominal[np.array(trainingLabels)==classes[j]][:,k]==validationDataNominal[i,k]).astype(float).sum()/(trainingDataNominal.shape[0]))
                valPostNominal[i,j,k]=p
#     print((valPostNominal.prod(2)*valPostContinuous))
    for i in range(len(valPostContinuous)):
        print("p(H0|X)"+str((valPostNominal.prod(2)*valPostContinuous)[i,0])+"\t"+"p(H1|X)"+str((valPostNominal.prod(2)*valPostContinuous)[i,1]))
    predictedLabels=(valPostNominal.prod(2)*valPostContinuous).argmax(1)

    validationLabels=np.array(validationLabels).astype(float)
    
    return predictedLabels, validationLabels


# # Split Training and testing data by K-fold implementation

# In[88]:

perm=np.random.permutation(len(data))


# In[89]:
if(test):
    K=1
else:
    K=10

accuracies=np.zeros(shape=(K,))
precision=np.zeros(shape=(K,))
recall=np.zeros(shape=(K,))
f1=np.zeros(shape=(K,))


# In[90]:

# print("k: "+str(k))
for i in range(K):
    trainingData=[]
    trainingLabels=[]
    validationData=[]
    validationLabels=[]
    validationDataStart=int(i*len(perm)//10)
    validationDataEnd=int((i+1)*len(perm)//10)
    for j in range(len(data)):
        if(j in range(validationDataStart,validationDataEnd)):
            validationData.append(data[perm[j]][:-1])
            validationLabels.append(data[perm[j]][-1])
        else:
            trainingData.append(data[perm[j]])
            trainingLabels.append(data[perm[j]][-1])
    if (K == 1):
        trainingData = data
        validationData = np.array(testData).reshape(1,len(testData))
        trainingLabels = data[:, -1]
        print(validationData.shape)
        # print(np.array(testData)[:, -1])
        validationLabels = np.array(testData).reshape(1,len(testData))[:,-1]
    else:
        trainingData = np.array(trainingData)
        validationData = np.array(validationData)

    print("\nIteration "+str(i))
    print("Training data: "+str(len(trainingData)))
    print("Validation data: "+str(len(validationData)))
    predictedLabels, trueLabels = NaiveBayes(trainingData=trainingData, validationData=validationData, trainingLabels=trainingLabels, validationLabels=validationLabels)
    mt=METRICS(trueLabels, predictedLabels)
    try:
        accuracies[i]=mt.accuracy()
    except:
        print("Accuracy cannot be computed")
    try:
        precision[i]=mt.precision()
    except:
        print("Precision cannot be computed")
    try:
        recall[i]=mt.recall()
    except:
        print("Recall cannot be computed")
    try:
        f1[i]=mt.f1()
    except:
        print("F1 measure cannot be computed")
print("\n")
print("Average accuracy: "+str(accuracies.mean()))
print("Average precision: "+str(precision.mean()))
print("Average recall: "+str(recall.mean()))
print("Average F1-measure: "+str(f1.mean()))
#     result=knn(trainingData,validationData,trainingLabels,k)
#     accuracies[i]=(np.array(result)==np.array(validationLabels)).astype(float).sum()/len(validationLabels)
#     print("Accuracy:"+str(accuracies[i])+str('\n'))



