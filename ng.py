# Extraction of feature vectors from a folder of images
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import os

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

# Extract the feature vectors from all the images in a folder
def extractDataFromFolder(datafile, path, label):
    for filename in os.listdir(path):
        img = path
        img += "\\"
        img += filename
        original = cv2.imread(img,cv2.IMREAD_UNCHANGED)
        
        newOg = cv2.GaussianBlur(original, (3,3), 0)
        grayscale = cv2.cvtColor(newOg,cv2.COLOR_BGR2GRAY)
        
        gmGrayscale = buildSobelMap(grayscale)
        ngMapGrayscale = buildNgMat(gmGrayscale)
        data = getNGFeatureVector(ngMapGrayscale)
        ngMapGrayscale = cv2.resize(ngMapGrayscale, (NG_WINDOW_SIZE,NG_WINDOW_SIZE));
        target = open (datafile, 'a')
        for i in range(0,len(data)):
            target.write(str(data[i]))
            target.write(",")
        if(label == 'positive'):
            target.write("1\n")
        else:
            target.write("0\n")
        target.close()
    
# Will create a csv file or overwrite the existing file
datafile = raw_input("Name of csv file: ")
datafile += ".csv"
target = open (datafile, 'w').close()

pathForPositive = raw_input("Directory of Positive Folder: ")
pathForNegative = raw_input("Directory of Negative Folder: ")
extractDataFromFolder(datafile, pathForPositive, 'positive')
extractDataFromFolder(datafile, pathForNegative, 'negative')

print datafile + " has been created!"
cv2.waitKey(0)
cv2.destroyAllWindows()