#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import MapReduce

mr = MapReduce.MapReduce()

def mapper(record):
    key = record[0]
    value = record[1][:-10]
    mr.emit_intermediate(value, key)
    
def reducer(key, list_of_values):
    mr.emit((key))
    
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)

