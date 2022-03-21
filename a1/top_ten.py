#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sys
import json

def main():

    tweetfile = open(sys.argv[1])
    
    hashlist = {}
    
    for line in tweetfile:
        try:
            tweet = json.loads(line)
            entities = tweet['entities']
            if entities is not None:
                hashtags = entities['hashtags']
                if hashtags is not None:
                    for hashtag in hashtags:
                        hashlist[hashtag['text']] = hashlist.get(hashtag['text'],0) + 1
        except:
            pass
        
    x = list(sorted(hashlist.items(), key = lambda x: x[1], reverse=True))
    topten = x[:10]
    
    for hashtag in topten:
        print(hashtag[0], " ", hashtag[1])        

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()

