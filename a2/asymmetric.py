#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import MapReduce

mr = MapReduce.MapReduce()

def mapper(record):
    #key = person
    #value = friend
    
    key = record[0]
    value = record[1]
    mr.emit_intermediate(key, value)
    mr.emit_intermediate(value, key)

##https://stackoverflow.com/questions/16705407/mapreduce-solution-in-python-to-identify-asymmetric-pairs
def reducer(key, list_of_values):
    for person in list_of_values:
        if list_of_values.count(person) < 2:
            output = key, person
            mr.emit((output))
    
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)

