'''
This program illustrates the use of findContours and drawContours.
The original image is put up along with the image of drawn contours.
Usage:
    contours.py
A trackbar is put up which controls the contour level from -3 to 3
'''

import numpy as np
import cv2
import math
import sys
from scan_new import *

COL_1_X = 105

    
if __name__ == '__main__':
    print __doc__


    print __doc__
    try:
        fn = sys.argv[1]
    except:
        fn = "test1.PNG"

    img = cv2.imread(fn,cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_c = cv2.imread(fn)
    h, w = img.shape[:2]

    print type(img[0,0])
    img_t = np.zeros((h, w, 3), np.uint8)
    img_c = img_c.astype(np.float32)
    img_t = cv2.subtract(img_c,img_c * np.float32(0.999))
    # cv2.rectangle(img,(0,0),(2448,700),255,-1)
    # img = cv2.resize(img,(0,0),fx=1.00,fy=1.00)
    # img = reposition_object(fn)
    # print img
    # print type(img)
    # img = cv2.resize(img,(500,500))
    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 1)
    

    contours0, hierarchy = cv2.findContours( img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print contours0[1], contours0[0], contours0[2]
    contours = [cv2.approxPolyDP(cnt, 1, True) for cnt in contours0]
    
    # contours = []
    # for cnt in contours0:
    #   if len(cnt) > 7:
    #     contours.append(cv2.fitEllipse(cnt))

    
    def getOct(conts):
      octs = []
      for cnt in conts:
        # if cv2.contourArea(cnt) >=500 and len(cnt) > 7 and len(cnt) < 25:
          octs.append(cnt)
      
      return octs

    def distance(p,q):
      return (p[0][0] - q[0][0]) + (p[0][1] - q[0][1])
    
    def mdistance(cnt):
      m = 0
      for i in range(0,len(cnt)):
        for j in range(i,len(cnt)):
          if distance(cnt[i],cnt[j]) > m:
            m = distance(cnt[i],cnt[j])
      return m

    def center(circle):
      (x,y,w,h) = cv2.boundingRect(circle)
      return (x + w / 2,y + h / 2)

    def sortByRow(circles):
      circles = removeDup(circles,lambda (x,y): x)
      c = []
      for i in range(0,6):
        if i == 5:
          c.append(circles[i*52:])
        else:
          c.append(circles[i * 52:(i + 1) * 52])
      for i in range(0,6):
        sort = sorted(c[i],key=lambda x: x[0])
        sorted_octs = map(lambda (x,y): y,sort)
        c[i] = sorted_octs
      for i in range(0,6):
        print len(c[i])
      return [item for sublist in c for item in sublist]

    def sortByCol(circles):
      octs = getOct(circles)
      centers = map(lambda x: (center(x),x),octs)
      unique = removeDup(centers,lambda (x,y): x)
      sort = sorted(unique,key=lambda x: x[0])
      sorted_octs = map(lambda (x,y): y,unique)
      return sorted_octs[::-1]

    def blackDensity(circle):
      rect = cv2.boundingRect(circle)
      x,y,w,h = rect
      vis = drawOct(contours)
      cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
      count = 0
      for i in range(x,x+w):
        for j in range(y,y+h):
          if vis[j][i][0] == 0:
            count += 1
      return count / cv2.contourArea(circle)

    def epsilon((x,y),(w,z)):
      s_d = math.sqrt((x - w)**2 + (y - z)**2)
      return abs(s_d) <= 2

    def isIn(elem, l, comp):
      for item in l:
        if comp(elem,item) == True:
          return True
      return False

    def removeDup(l,f):
      unique = []
      print len(l)
      for elem in l:
        if not isIn(f(elem),map(f,unique),epsilon):
          unique.append(elem)
      print len(unique)
      return unique

    def display(image,contours):
      for i in range(0,len(contours)):
        cv2.drawContours(image, [contours[i]], -1, (128,255,255),
          1, cv2.CV_AA)
        cv2.putText(image,str(i),center(contours[i]), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        
      # vis = cv2.resize(vis,(0,0),fx=0.4,fy=0.4)
      cv2.imshow('contours', image) 
    # update(6)
    # cv2.createTrackbar( "levels+3", "contours", 3, 7, update )

    sorted_by_order = sortByCol(contours)
    vis = np.zeros((h, w, 3), np.uint8)
    display(img_t,sorted_by_order)

    # print getOct(contours)
    0xFF & cv2.waitKey()
    cv2.destroyAllWindows()