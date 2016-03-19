# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 00:34:37 2016

@author: Kaira
"""

import csv
import random
import math

import numpy as np
import cv2
from matplotlib import pyplot as plt

import StringIO


#Load Data
def loadCsv(filename):
    lines = csv.reader(open(filename,"rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset
    
#filename = 'data.csv'
#dataset = loadCsv(filename)
#print ('Loaded datafile {0} with {1} rows').format(filename,len(dataset))

#Split Data
def splitDataset(dataset,splitRatio):
    trainSize = int(len(dataset)* splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]
    
##test
#Data = [[1],[2],[3],[4],[5]]
#splitRatio = 0.67
#train,test =splitDataset(Data,splitRatio)
#print('Split {0} rows into train with {1} and test with {2}').format(len(Data), train, test)
    
##Summarize Data
#Separate by Class

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated
        
#data2 = [[1,20,1],[2,21,0],[3,22,1]]
#separated = separateByClass(data2)
#print('Separated Instances: {0}').format(separated)

#Caluclate Mean
def mean(numbers):
    return sum(numbers)/float(len(numbers))
    
#calculate Standard Deviation
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
    
#numbers=[1,2,3,4,5]
#print ('Summary of {0}: mean={1},stdev={2}').format(numbers,mean(numbers),stdev(numbers))

#Summarize Dataset
def summarize(dataset):
    summaries=[(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries
#    
#summary = summarize(data2)
#print('Attribute summaries:{0}').format(summary)

#Summarize by Class
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue,instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries
    
#data3 = [[1,20,1],[2,21,0],[3,22,1],[4,22,0]]
#summary = summarizeByClass(data3)
#print('Summary by Class Value: {0}').format(summary)

#Calculate Gaussian Probability
def calculateProbability(x,mean,stdev):
    exponent = math.exp(-math.pow(x-mean,2)/(2*math.pow(stdev,2)))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent
    
#x = 71.5
#mean = 73
#stdev = 6.2
#probability = calculateProbability(x,mean,stdev)
#print('Probability of belonging to this class: {0}').format(probability)

#Calculating Class Probabilities
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
        
#summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
#inputVector = [1.1, '?']
#probabilities = calculateClassProbabilities(summaries, inputVector)
#print('Probabilities for each class: {0}').format(probabilities)

#Make a Prediction
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
#summaries2 = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
#inputVector = [1.1, '?']
#result = predict(summaries2, inputVector)
#print('Prediction: {0}').format(result)

#Make Predictions
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
#summaries3 = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
#testSet = [[1.1, '?'], [19.1, '?']]
#predictions = getPredictions(summaries3, testSet)
#print('Predictions: {0}').format(predictions)

#Get Accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
 
#testSet2 = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
#predictions2 = ['a', 'a', 'a']
#accuracy = getAccuracy(testSet2, predictions2)
#print('Accuracy: {0}').format(accuracy)
 
##Cany
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged
 


###Get Feature Vectors
ddepth = cv2.CV_16S
NG_RESIZE = 8
NG_WINDOW_SIZE = 350

def buildSobelMap(m):
    gradX = cv2.Sobel(m, ddepth, 1, 0, 3)
    gradY = cv2.Sobel(m, ddepth, 0, 1, 3)
    absGradX = cv2.convertScaleAbs(gradX)
    absGradY = cv2.convertScaleAbs(gradY)
    sobelMat = cv2.addWeighted(absGradX, 0.5, absGradY, 0.5, 0)
    return sobelMat
  
def buildNgMat(m):
    resizedMat = cv2.resize(m, (NG_RESIZE,NG_RESIZE));
    return resizedMat

#still figuring this out
def getNGFeatureVector(m):
    feature = []
    for i in range (0,8):
        for j in range (0, 8):
            values = m[i][j]
            feature.append(values)
    return feature
    
    
## Convert to float
def fn(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0    


##Draw Rectangles for Images and Stuff
def drawStuff(image,summaries):
    #load image
    orig = cv2.imread(image,cv2.IMREAD_UNCHANGED)
    img = cv2.imread(image,cv2.IMREAD_UNCHANGED)
    
    #autocanny
    edges = auto_canny(img) #returns edged image
    
    ret,thresh = cv2.threshold(edges,127,255,0)
    im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    print len(contours)
    
    for i in range(0, len(contours)-1):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        r = cv2.boundingRect(cnt)

        a = buildSobelMap(orig[r[1]:r[1]+r[3], r[0]:r[0]+r[2]])
        b = buildNgMat(a)
        feature = getNGFeatureVector(b)
        
        output = StringIO.StringIO()
        for i in range(0,len(feature)):
            output.write(str(feature[i]))
            output.write(",")
        output.write("'?'")
        content = output.getvalue()
        output.close()
        
        split = content.split(",")
        split2 =([s.replace('\'', '') for s in split])
        
        for i in range(0,(len(split2)-1)):
            d=0
            a = split2[i].replace('[','')
            split2[i] = a
            b = split2[i].replace(']','')
            split2[i] = b
            c = split2[i].split(" ")
            
            for j in range(0,len(c)):
                d+=fn(c[j])
                split2[i]=d/3
                
        
        
        pred = predict(summaries,split2)
        print pred
        if (pred=='1.0'):
            cv2.rectangle(orig,(x,y),(x+w,y+h),(225,0,0))
        elif (pred=='0.0'):
            cv2.rectangle(orig,(x,y),(x+w,y+h),(0,0,225))

        #cv2.rectangle(orig,(x,y),(x+w,y+h),(0,0,255))
    
    #display
    cv2.imshow(image,orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


####Main

def main():
    filename = 'data.csv'
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet,testSet = splitDataset(dataset,splitRatio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
    
    summaries = summarizeByClass(trainingSet)
    #print('Summary by Class Value: {0}').format(summaries)
    
    predictions = getPredictions(summaries,testSet)
    accuracy = getAccuracy(testSet,predictions)
    print('Accuracy:{0}%').format(accuracy)
    
    inputVector = [24,10,16,29,26,59,12,35,35,33,15,25,31,41,30,28,28,28,14,18,12,35,66,29,37,9,12,8,14,22,39,24,26,12,8,12,11,29,13,22,33,17,7,12,26,8,17,44,22,55,19,14,28,18,47,31,7,11,35,16,24,30,27,5,'?']
    result = predict(summaries, inputVector)
    print('Prediction: {0}').format(result)
    
    inputVector2 = [10,13,10,12,20,25,11,4,34,41,37,33,33,27,16,7,28,34,42,53,55,48,38,23,24,26,35,42,48,54,45,28,21,23,13,28,35,48,41,11,17,45,25,22,32,41,39,8,24,67,44,16,32,52,45,16,8,59,49,19,20,43,38,8,'?']
    result2 = predict(summaries, inputVector2)
    print('Prediction: {0}').format(result2)
    
    image = 'blocks.jpg'
    drawStuff(image,summaries)
    
    
    
    
main()