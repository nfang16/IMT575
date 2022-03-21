#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import json

def main():

    tweetfile = open(sys.argv[1])
    
    terms = {}
    count = 0

    for line in tweetfile:
        tweet = json.loads(line)
        try:
            text = tweet['text']
            if text is not None:
                text = text.split()
                for word in text:
                    count += 1
                    terms[word] = terms.get(word, 0) + 1
        except:
            pass

    for x in terms:
        print(x, " ", terms[x]/float(count))
        

if __name__ == '__main__':
    main()

