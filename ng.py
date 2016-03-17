import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import os

ddepth = cv2.CV_16S
NG_RESIZE = 8
NG_WINDOW_SIZE = 350

# 5
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

datafile = "data.csv"
target = open (datafile, 'w').close() 

# extracting the feature vectors from the Positive folder (Objects)
pathForPositive = "C:\Users\Jose Emmanuel\Documents\Python Scripts\Positive"

for filename in os.listdir(pathForPositive):
    img = pathForPositive
    img += "\\"
    img += filename
    original = cv2.imread(img,cv2.IMREAD_UNCHANGED)
    
    newOg = cv2.GaussianBlur(original, (3,3), 0)
    # convert to grayscale
    grayscale = cv2.cvtColor(newOg,cv2.COLOR_BGR2GRAY)
    
    gmGrayscale = buildSobelMap(grayscale)
    ngMapGrayscale = buildNgMat(gmGrayscale)
    data = getNGFeatureVector(ngMapGrayscale)
    ngMapGrayscale = cv2.resize(ngMapGrayscale, (NG_WINDOW_SIZE,NG_WINDOW_SIZE));
    target = open (datafile, 'a')
    for i in range(0,len(data)):
        target.write(str(data[i]))
        target.write(",")
    target.write("1\n")
    target.close()

# extracting the feature vectors from the Negative folder (Non-objects)
pathForNegative = "C:\Users\Jose Emmanuel\Documents\Python Scripts\Negative"

for filename in os.listdir(pathForNegative):
    img = pathForNegative
    img += "\\"
    img += filename
    original = cv2.imread(img,cv2.IMREAD_UNCHANGED)
    
    newOg = cv2.GaussianBlur(original, (3,3), 0)
    # convert to grayscale
    grayscale = cv2.cvtColor(newOg,cv2.COLOR_BGR2GRAY)
    gmGrayscale = buildSobelMap(grayscale)
    ngMapGrayscale = buildNgMat(gmGrayscale)
    data = getNGFeatureVector(ngMapGrayscale)
    ngMapGrayscale = cv2.resize(ngMapGrayscale, (NG_WINDOW_SIZE,NG_WINDOW_SIZE));
    target = open (datafile, 'a')
    for i in range(0,len(data)):
        target.write(str(data[i]))
        target.write(",")
    target.write("0\n")
    target.close()

print "Done!"
cv2.waitKey(0)
cv2.destroyAllWindows()