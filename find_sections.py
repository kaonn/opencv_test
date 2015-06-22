from four import *
from crop_test import *
from ellipse import *
import numpy as np
import cv2
import os.path

IMG_WIDTH_PERCENTAGE = 0.3
MAX_Y_DIFF = 50
MAX_LENGTH_DIFF = 200

def span(l,p):
  i = 0
  pref = []
  while i < len(l) and p(l[i]):
    pref = pref + [l[i]]
    i += 1
  return (pref,l[i:],len(pref))

def mode(l,cmp):
  freqList = []
  copy = list(l) 
  while copy != []:
    (pref,tail,length) = span(copy, lambda x: cmp(copy[0],x) == 0)
    if length != 0:
      freqList += [(pref,length)]
    copy = tail
  md = max(freqList,key = lambda x:x[1])
  return md

def getContours(filename,threshold_value):
  img = cv2.imread(filename)

  (height, width) = img.shape[:2]
  if not(isVert(height,width)):
    b = int((width - height)/2)
    image_r = cv2.copyMakeBorder(img,b,b,0,0,cv2.BORDER_CONSTANT)
    image_r = rotate(image_r,-90)
    img = image_r
  
  img_t = img.copy()
  img_g = cv2.cvtColor(img,cv2.cv.CV_BGR2GRAY)

  (h,w,_) = img.shape
  (_,img_c) = cv2.threshold(img_g,threshold_value,255,cv2.THRESH_BINARY)
  
  (contours,_) = cv2.findContours(img_c.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  
  perimeter = map(lambda x: cv2.arcLength(x,True), contours)
  approxPoly = map(lambda c,p: cv2.approxPolyDP(c,0.02*p,True),contours,perimeter)
  onlyTwoPoints = filter(lambda x: len(x) == 2, approxPoly)
  longSegments = filter(lambda x: IMG_WIDTH_PERCENTAGE * w < cv2.arcLength(x,False) < h,onlyTwoPoints)

  longSegments = sorted(longSegments, key=lambda x: cv2.arcLength(x,False))
  
  (mostCommon,l) = mode(longSegments,lambda x,y: 0 if abs(cv2.arcLength(x,False) - cv2.arcLength(y,False)) < MAX_LENGTH_DIFF else 1)  
  centers = map(lambda x: (x,tuple(x[0][0]),tuple(x[1][0])), mostCommon)

  unique = map(lambda (x,p1,p2): sorted([p1,p2],key = lambda x: x[0]),removeDup(centers, lambda (foo,p1,p2): p1[1], lambda (x,p1,p2): cv2.arcLength(x,False), lambda x,y,e: int(not (abs(x - y) < e))*(int(x > y) - int(x < y)), MAX_Y_DIFF)) 

  unique = sorted(unique,key = lambda x: x[1][1])
  cv2.drawContours(img_t,longSegments,-1,(0,255,0),2)
  cv2.imshow("f",img_t)
  cv2.waitKey(0)

  if len(unique) == 6:
    unique = unique[1:]
  elif len(unique) == 7:
    unique = unique[2:]
  return unique

def isVert(h,w): 
  return h > w

def rotate(img,deg):
  (h, w) = img.shape[:2]
  M = cv2.getRotationMatrix2D((w/2,h/2),deg,1)
  dst = cv2.warpAffine(img,M,(w,h))
  return dst

def getImage(filename):
  img = cv2.imread(filename)

  (height, width) = img.shape[:2]
  if not(isVert(height,width)):
    b = int((width - height)/2)
    image_r = cv2.copyMakeBorder(img,b,b,0,0,cv2.BORDER_CONSTANT)
    image_r = rotate(image_r,-90)
    img = image_r
  
  return img

def getSections(image,points):
  sections = []

  points = [item for sublist in points for item in sublist]

  for i in range(4):
    sections.append(four_point_transform(image,points[2*i:2*i+4]))
  
  #for i in range(4):
  #  cv2.imshow("f",sections[i])
  #  cv2.waitKey(0)
  return sections

