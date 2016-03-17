import cv2
import numpy as np
from matplotlib import pyplot as plt
# import the necessary packages
 
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

og = cv2.imread('test.jpg',cv2.IMREAD_UNCHANGED) # original picture
ogWithContours = cv2.imread('test.jpg',cv2.IMREAD_UNCHANGED) # original picture with contours
img = cv2.imread('test.JPG',0) # image for Canny
edges = auto_canny(img) # 
ret,thresh = cv2.threshold(edges,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

#For every contour, draw a rectangle surrounding it
for i in range(0, len(contours)):
    cnt = contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    r = cv2.boundingRect(cnt)
    cv2.rectangle(ogWithContours,(x,y),(x+w,y+h),(0,0,255))
    #M = cv2.moments(cnt)
    
    # 3 to 4
    # create image file for each ROI
    fileName = "ROI-"
    fileName += str(i)
    fileName += ".jpg"    
    cv2.imwrite(fileName, og[r[1]:r[1]+r[3], r[0]:r[0]+r[2]])
    

cv2.imshow("Original",ogWithContours)
cv2.imshow("Edges",edges)

cv2.waitKey(0)
cv2.destroyAllWindows()