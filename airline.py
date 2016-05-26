# read in airline sentiment data.
from __future__ import division
import csv
import string
import operator


def readCSV(fileName,delim):
    # Need csv module
    # delim is delimiter, and quoting is set to none. It has something to do with escape character. Not sure what that means
    # identation  =  4 spaces
    newFile = []
    with open(fileName,'rbU') as f:
        reader = csv.reader(f,delimiter=delim)
        for row in reader:
            newFile.append(row)
    return newFile

# read in the data. 20 columns, 14641 rows, with header.
# col 0 - ID, 5--state: pos,neg,neu , 14 -- words

airlineSent = readCSV('Airline-Sentiment-2-w-AA.csv',',')
airlineSent.pop(0)

def obtainColumns(dataRow):
    # reove the signs and @ for each comment.
    temp = dataRow[14]
    asking = "".join(word for word in temp if word not in string.punctuation)
    return (dataRow[0],dataRow[5],dataRow[7],asking)
    
cleanAirline = map(lambda x:obtainColumns(x), airlineSent)

# split into train and test.
ind = 9 * len(cleanAirline)/10
train = cleanAirline[0:ind]
test = cleanAirline[ind:len(cleanAirline)]

def createDictionary(trainingSet):
    negDic = {}
    posDic = {}
    neuDic = {}
    
    for item in trainingSet:
        if item[1] == 'negative':
            temp = item[3].lower().split()
            for item2 in temp:
	        if item2 not in negDic.keys():
		    negDic[item2] = 1
		else:
		    negDic[item2] = negDic[item2] +1
    for item3 in trainingSet:
        if item3[1] == 'positive':
	    temp = item3[3].split()
	    for item4 in temp:
	        if item4 not in posDic.keys():
		    posDic[item4] = 1
		else:
		    posDic[item4] = posDic[item4] +1
    
    for item5 in trainingSet:
        if item5[1] == 'neutral':
	    temp = item5[3].split()
	    for item6 in temp:
	        if item6 not in neuDic.keys():
		    neuDic[item6] = 1
		else:
		    neuDic[item6] = neuDic[item6] +1    
    
    return (posDic,negDic,neuDic)





# then I will calculate individual probability...
# if the posterior prob for postitive is > posterior probability for negative... then classify as positive
wordCount =createDictionary(train) 
#len(wordCount[0])

newPos = {}
newNeg = {}
newNeu = {}
sorted_pos = sorted(wordCount[0].items(), key=operator.itemgetter(1),reverse=True)[10:len(wordCount[0])-10]
for item in sorted_pos:
    newPos[item[0]] = item[1]
sorted_neg = sorted(wordCount[1].items(), key=operator.itemgetter(1),reverse=True)[10:len(wordCount[1])-10]
for item in sorted_neg:
    newNeg[item[0]] = item[1]
    
sorted_neu = sorted(wordCount[2].items(), key=operator.itemgetter(1),reverse=True)[10:len(wordCount[1])-10]
for item in sorted_neu:
    newNeu[item[0]] = item[1]    
    
wordCountNew = (newPos,newNeg)



def findProb(train,wordCount,str1):
  
     # so now i have a dictionary for negative and positive comments.
    proPos = sum(map(lambda x:x[1]=='positive',train))
    proNeg = sum(map(lambda x:x[1]=='negative',train))

    inputVec = str1.split()
    posValue = 0
    negValue = 0
    
    WCPos = reduce(lambda x,y:x+y,wordCount[0].values())
    WCNeg = reduce(lambda x,y:x+y,wordCount[1].values())
    
    PP = proPos / (proPos + proNeg)
    PN = proNeg / (proPos + proNeg)
    
    
    for item in inputVec:
        if item in wordCount[0].keys() and item in wordCount[1].keys():
	    top1 = math.log(wordCount[0][item]) - math.log(proPos) +math.log(PP) 
	    top2 = math.log(wordCount[1][item]) - math.log(proNeg) + math.log(PN)
            bot1 = math.log(wordCount[0][item] + wordCount[1][item])
            
            posValue = posValue + top1 - bot1 
	    negValue = negValue + top2 - bot1
	
	
    return (posValue,negValue)


findProb(train,wordCountNew,train[13][3])
# now we can evaluate the probability... since i used log, the smaller the probability, the smaller the log value...
pre = []
for item in test:
    score = findProb(train,wordCountNew,item[3])
    if item[1] != 'neutral':
        if score[0] > score[1]:
            pre.append((item[0],item[1],'positive'))
        else:
            pre.append((item[0],item[1],'negative'))
# let's look at len of postive vocab and negative vocab. i want to trim top 10 and bottom 10

# check for scores:
count = 0
for item in pre:
    if item[1] == item[2]:
      count = count +1
count/len(pre)



# the following code includes the neutral category



def findProbNeural(train,wordCount,str1):   # only use words show up in all 3 categories. need to find a way to make it so that
  ## we can have words from some categories but not all.
  
     # so now i have a dictionary for negative and positive comments.
    proPos = sum(map(lambda x:x[1]=='positive',train))
    proNeg = sum(map(lambda x:x[1]=='negative',train))
    proNeu = sum(map(lambda x:x[1]=='neutral',train))

    inputVec = str1.split()
    posValue = 0
    negValue = 0
    neuValue = 0
    
    WCPos = reduce(lambda x,y:x+y,wordCount[0].values())
    WCNeg = reduce(lambda x,y:x+y,wordCount[1].values())
    WCNeu = reduce(lambda x,y:x+y,wordCount[2].values())
    
    PP = proPos / (proPos + proNeg + proNeu)
    PN = proNeg / (proPos + proNeg + proNeu)
    PU = proNeu / (proPos + proNeg + proNeu)
    
    
    for item in inputVec:
        if item in wordCount[0].keys() and item in wordCount[1].keys() and item in wordCount[2].keys():
	    top1 = math.log(wordCount[0][item]) - math.log(proPos) + math.log(PP) 
	    top2 = math.log(wordCount[1][item]) - math.log(proNeg) + math.log(PN)
	    top3 = math.log(wordCount[2][item]) - math.log(proNeu) + math.log(PU)
            bot1 = math.log(wordCount[0][item] + wordCount[1][item]+ wordCount[2][item]) 
            
            posValue = posValue + top1 - bot1 
	    negValue = negValue + top2 - bot1
	    neuValue = neuValue + top3 - bot1
	
	
    return (posValue,negValue,neuValue)

findProbNeural(train,wordCountNew,train[13][3])

wordCountNew = (newPos,newNeg,newNeu)
pre2 = []
for item in test:
    score = findProbNeural(train,wordCountNew,item[3])
    temp = ['positive','negative','neutral']
    for i in range(3):
        if score[i] == max(score):
	    pre2.append((item[0],item[1],temp[i]))
        
        
count = 0
for item in pre2:
    if item[1] == item[2]:
      count = count +1
count/len(pre2)

# without neutral, 89, with neutral, 76%...

