#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import MapReduce
import numpy as np


mr = MapReduce.MapReduce()

def mapper(record):
    matrixID = record[0]
    rowID = record[1]
    colID = record[2]
    value = record[3]
    
    mr.emit_intermediate(1, (matrixID, rowID, colID, value))
    
def reducer(key, list_of_values):
    aMatrix = []
    bMatrix = []
    
    for v in list_of_values:
        if v[0] == "a":
            aMatrix.append(v)
        elif v[0] == "b":
            bMatrix.append(v)
    
    nRows = max(x[1] for x in aMatrix) + 1
    nCols = max(x[2] for x in bMatrix) + 1
    
    #for i in range(len(aMatrix)):
    #    for j in range(len(bMatrix[0])):
    #        for k in range(len(bMatrix)):
    #            results[i][j] += bMatrix[i][k] * bMatrix[k][j]
    
    product = 0
    for i in range(nRows):
        for j in range(nCols):
            for k in range(len(aMatrix)):
                for l in range(len(bMatrix)):
                    product += aMatrix[k][3] * bMatrix[k][3]

        mr.emit((i,j,product))

if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)


# In[ ]:




