#!/usr/bin/python3

#import pandas as pd
#import random
import numpy as np
import sys

#print(sys.stdin)
file=open("temp", "r")
k=int(file.readline())
metricOrder=int(file.readline())
file.close()
centroids=np.load("centroids.npy")
#print(centroids.shape)
for line in sys.stdin:
    DB=np.fromstring(line, dtype=float, sep=' ')
    DB.reshape(len(DB),1)
#    print(DB.shape)
#    print(type(DB))
#    print(DB)
    dist = np.linalg.norm(DB - centroids, ord=metricOrder, axis=1)
    print(str(np.append(np.argmin(dist),DB).tolist()))
