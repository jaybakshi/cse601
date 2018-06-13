#!/usr/bin/python3
import sys
import numpy as np

prevKey=None
centroids=None

for line in sys.stdin:
	data=np.array(eval(line))
	dataVal=data[1:]
	key=data[0:1]
	#print(key)
	if((prevKey!=None) and (prevKey!=key)):
		print(str(centroids.mean(0).tolist()))
		centroids=None
	if(prevKey==key):
		centroids=np.vstack((centroids, dataVal.reshape(1,len(dataVal))))
	else:
		centroids=dataVal.reshape(1,len(dataVal))
	prevKey=key
if(prevKey!=None):
	print(str(centroids.mean(0).tolist()))
