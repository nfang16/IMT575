import sys
import json

def hw():
    print('Hello, world!')

def lines(fp):
    print(str(len(fp.readlines())))

def main():
    
    afinn_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])
    
    #hw()
    #lines(sent_file)
    #lines(tweet_file)
    
    sentiments = {}
    for line in afinn_file:
        word, value = line.split("\t")
        sentiments[word] = int(value)
    
    sentimentKeys = sentiments.keys()
    
    tweetscore = []
    otherwords = {}
    time = {}
    
    count = 0
    for line in tweet_file:
        tweet = json.loads(line)
        try:
            tweetscore.append(0)
            text = tweet['text']
            if text is not None:
                text = text.split()
                for word in text:
                    if word in sentimentKeys:
                        tweetscore[count] += sentiments[word]
                    else:
                        pass
                for word in text:
                    if word not in sentimentKeys:
                        if word in otherwords:
                            time[word] += 1
                            otherwords[word] += tweetscore[count]
                        else:
                            otherwords[word] = tweetscore[count]
                            time[word] = 1
        except:
            pass

        count += 1

    for word in otherwords:
        print(word, " ", otherwords[word]/float(time[word][0]))

if __name__ == '__main__':
    main()
