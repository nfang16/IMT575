#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import MapReduce

mr = MapReduce.MapReduce()

def mapper(record):
    key = record[1]
    value = list(record)
    mr.emit_intermediate(key, value)
    
def reducer(key, list_of_values):
    for i in range(len(list_of_values)):
        mr.emit(list_of_values[0] + list_of_values[i])
    
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)

