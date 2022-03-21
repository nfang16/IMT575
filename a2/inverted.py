#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import MapReduce

mr = MapReduce.MapReduce()

def mapper(record):
    key = record[0]
    value = record[1]
    words = value.split()
    for w in words:
        mr.emit_intermediate(w, key)
        
def reducer(key, list_of_values):
    total = 0
    newList = []
    for v in list_of_values:
        if v not in newList:
            newList.append(v)
    mr.emit((key, newList))
    
    
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)

