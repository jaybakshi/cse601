#!/usr/bin/python3
import sys
import numpy as np
import glob

flag=None
files=glob.glob("output/part*")
for fle in files:
	with open(fle) as f:
		for line in f.readlines():
			data=np.array(eval(line))
			if(flag!=None):
				centroids=np.vstack((centroids, data.reshape(1,len(data))))
			else:
				centroids=data.reshape(1,len(data))
				flag=1
print(centroids)
np.save("centroids.npy",centroids)
