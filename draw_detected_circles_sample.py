# code derived from Hough Circle Tranform example on 
# opencv-python-tutroals.readthedocs.org

"""
- uses built in cv2.HoughCircles() function to detect circles in image
- experimentally had to determine best paratmers for cv2.HoughCircles() 
    function in order to make it most accurate
- paratmers needed to be determined included minimum distance between detected
    circles, minimum radius of circle, and maximum radius of circle
"""
import
import cv2
import numpy as np

img = cv2.imread('bubblecolor.jpg',0)
# img = cv2.imread('answersheet2.jpg',0)


#threshold from thresholding.py (from 112 demo)

cimg = cv2.adaptiveThreshold(img,255,\
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,75,10)

cimg = cv2.cvtColor(cimg,cv2.COLOR_GRAY2RGB)

circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,minDist = 40 ,
                            param1=50,param2=10,minRadius=10,maxRadius=20)

#answersheet.jpg
# circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,minDist = 20,
#                             param1=50,param2=10,minRadius=10,maxRadius=13)

if circles is not None: 
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        #draw center of circle

        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        cv2.imshow('detected circles',cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# cv2.imshow('detected circles',cimg)
cv2.imshow('original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()