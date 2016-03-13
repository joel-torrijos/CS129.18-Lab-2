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

og = cv2.imread('test.jpg',cv2.IMREAD_UNCHANGED)
img = cv2.imread('test.JPG',0)
edges = auto_canny(img)
ret,thresh = cv2.threshold(edges,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

for i in range(0, len(contours)-1):
    cnt = contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(og,(x,y),(x+w,y+h),(0,0,255))
    #M = cv2.moments(cnt)

cv2.imshow("Original",og)
cv2.imshow("Edges",edges)