
import sys
import json

def hw():
    print('Hello, world!')

def lines(fp):
    print(str(len(fp.readlines())))

def main():
    afinn_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])
    
    sentiments = {}
    for line in afinn_file:
        word, value = line.split("\t")
        sentiments[word] = int(value)
    
    for line in tweet_file:
        tweet = json.loads(line)
        try:
            text = tweet['text']
            if text is not None:
                words = text.split()
                score = 0
                for word in words:
                    if word in sentiments:
                        score += sentiments[word]
                print(score)
        except:
            pass

if __name__ == '__main__':
    main()