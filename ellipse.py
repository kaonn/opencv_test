#!/usr/bin/env python

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
import itertools as it
from scan_new import *

COL_1_X = 105888
    
def getImages(img):
  img_o = cv2.cvtColor(img,cv2.cv.CV_BGR2GRAY)
  # img_o = cv2.imread(fn,cv2.CV_LOAD_IMAGE_GRAYSCALE)

  img_c = img
  h, w = img_c.shape[:2]

  #for displaying purposes
  img_t = np.zeros((h, w, 3), np.uint8)
  img_c = img_c.astype(float)
  img_t = cv2.subtract(img_c,img_c * np.float32(0.999))

  #for detecting contours
  (thresh, img_c) = cv2.threshold(img_o, 100, 255, cv2.THRESH_BINARY)
  # (thresh, img_o) = cv2.threshold(img_o, 100, 255, cv2.THRESH_BINARY)

  #for detecting bubbles
  img_o = cv2.adaptiveThreshold(img_o,255,cv2.THRESH_BINARY,cv2.ADAPTIVE_THRESH_MEAN_C,75,0)
  
  return (img_t,img_c,img_o)

def getContours(img_c):
  contours0, hierarchy = cv2.findContours( img_c.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  contours = []
  for cnt in contours0:
    if len(cnt) > 7:
      contours.append(cv2.fitEllipse(cnt))
  return contours


def filterEllipse(boxes):
  ar_a = map(lambda (x,(w,h),a): ((w,h),a,(x,(w,h),a)), boxes)
  l = filter(lambda ((w,h),a,b): 500 < w*h < 5000 and 60 < a and a < 120, ar_a)
  median = np.median(map(lambda (ar,a,b): ar,ar_a))
  std = np.std(map(lambda (ar,a,b): ar, l))
  mean = np.mean(map(lambda (ar,a,b): ar,l))
  ls = filter(lambda ((w,h),a,b): mean - 0.9 * std < h*w < mean + 0.9 * std,l)
  return map(lambda (ar,a,b):b,l)

def sort(ells,f):
  sort = sorted(ells, cmp = lambda x,y: epsilon(x,y,10), key = f)
  return sort

def whiteDensity(image,ellipse):
  ((x,y),(h,w),a) = ellipse
  xi,xf = int(x - w / 2), int(x + w / 2)
  yi,yf = int(y - h / 2), int(y + h / 2)
  cv2.rectangle(image,(xi,yi),(xf,yf),(0,255,0),2)
  white = 0
  for i in range(xi,xf):
    for j in range(yi,yf):
      white += image[j][i]
      
  return round(white / (w * h * 255),2)

def epsilon((x,y),(w,z),e):
  if abs(x - w) < e:
    if abs(y - z) < e:
      return 0
    elif y > z:
      return 1
    else:
      return -1
  elif x > w:
    return 1
  else:
    return -1

def isIn(elem, l, comp):
  count = 0
  for item in l:
    if comp(elem,item,10) == 0:
      return (True,count)
    count += 1
  return (False,None)

def removeDup(l,f):
  unique = []
  for elem in l:
    (b,foo) = isIn(f(elem),map(f,unique),epsilon)
    if not b:
      unique.append(elem)
  return unique

def stats(img,ellg):
  l = map(lambda x: whiteDensity(img,x),list(ellg))
  mean = np.mean(l)
  std = np.std(l)
  (i,mini) = min([(i,w) for i,w in enumerate(l)], key = lambda (i,w): w)
  return (mean,std,mini,i)

def averageDist(ells,f):
  pos = map(f, ells)
  pos_t = pos[1:]
  pos = pos[:len(pos) - 1]
  widths = map(lambda x,y: abs(x - y),pos,pos_t)
  widths_f = filter(lambda x: x < 100, widths)
  ave = sum(widths_f) / len(widths_f)
  return int(ave)

def averageSize(ells):
  h_w = map(lambda (c,(h,w),a): (h,w), ells)
  ave = reduce(lambda (h1,w1),(h2,w2): ((h1 + h2) / 2,(w1 + w2) / 2), h_w, (0,0))
  return ave

def patch(ells,ints):
  centers = map(lambda (c,foo,bar): c, ells)
  ells_patched = []
  ave = averageSize(ells)
  count = 0
  for i in ints:
    (b,idx) = isIn(i,centers,epsilon)
    if not(b):
      ells_patched.insert(count,(i,ave,90))
    else:
      ells_patched.insert(count,ells[idx])
    
  return ells_patched

def lines(ells,f):
  cols = [[]]
  counter = 0
  for i in range(1,len(ells)):
    cur = ells[i]
    prev = ells[i - 1]
    (e,(x,y)) = f(cur,prev)
    if e < 20:
      cols[counter].append((np.float32(x),np.float32(y)))
    else:
      counter += 1
      cols.append([])
      cols[counter].append((np.float32(x),np.float32(y)))
  f = filter(lambda x: len(x) > 4, cols)
  return f

def sortByCol(ells):
  if len(ells) == 0:
    raise Exception("no ellipses!")
  cols = [[]]
  counter = 0
  cols[0].append(ells[0])
  for i in range(1,len(ells)):
    ((x,y),foo,bar) = ells[i]
    ((xp,yp),foo,bar) = ells[i - 1]
    f = (lambda x,xp: 0 if abs(x - xp) < 100 else x - xp)
    e = (f(x,xp)) * (f(y,yp))
    if e == 0:
      cols[counter].append(ells[i])
    else:
      counter += 1
      cols.append([])
      cols[counter].append(ells[i])
  fin = map(lambda x: sort(x,f = lambda x: x[0][::-1]), cols)
  return [item for sublist in fin for item in sublist]

def lineFit(d2_arr):
  L = map(lambda p: (cv2.fitLine(np.asarray(p),cv2.cv.CV_DIST_L2,0,0.01,0.01)),d2_arr)
  return L

def intersection(line_v,line_h):
  ([vx1],[vy1],[cx1],[cy1]) = line_v
  ([vx2],[vy2],[cx2],[cy2]) = line_h
  a = np.array([[vx1,-vx2],[vy1,-vy2]])
  b = np.array([cx2 - cx1,cy2-cy1])
  [t,s] = np.linalg.solve(a, b)
  p = (vx1*t+cx1,vy1*t+cy1)
  return p

def grouper(iterable, n, fillvalue=None):
  args = [iter(iterable)] * n
  return list(it.izip_longest(*args,fillvalue=fillvalue))


def display(image,contours,img_o,COL_W,filename):
  f_d = contours
  sort_c = sort(f_d, f = lambda x: x[0])
  sort_r = sort(f_d, f = lambda x: x[0][::-1])

  h, w = image.shape[:2]

  for points in lines(sort_c,f=lambda ((x,y),fo,ba),((xp,yp),foo,bar): (abs(x - xp),(x,y)) ):
    vx, vy, cx, cy = cv2.fitLine(np.asarray(points),cv2.cv.CV_DIST_L2,0,0.01,0.01)
    cv2.line(image,(int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)),(0,255,0),2)

  for points in lines(sort_r,f=lambda ((x,y),fo,ba),((xp,yp),foo,bar): (abs(y - yp),(x,y)) ):
    vx, vy, cx, cy = cv2.fitLine(np.asarray(points),cv2.cv.CV_DIST_L2,0,0.01,0.01)
    cv2.line(image,(int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)),(0,0,255),2)

  groups = grouper(contours,COL_W)

  # count = 0
  # for i in flattened:
  #   cv2.putText(image,str(count),tuple(map(int,i)), cv2.FONT_HERSHEY_PLAIN, 0.7, (255,0,255))
  #   count += 1

  count = 1
  for i,g in enumerate(groups):

    (mean,std,mini,i) = stats(img_o,g)
    if mini < mean:
      cv2.ellipse(image,g[i],(0,255,0),-1,cv2.CV_AA)

    j = 0
    for j in range(len(g)):
      (x,y) = g[j][False]
      cv2.putText(image,str(count),(int(x) - 15,int(y) + 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (255,0,0))
      cv2.ellipse(image,g[j],(0,255,0))
    count += 1

  cv2.imshow('contours', image)
  cv2.imwrite(filename,image)
  0xFF & cv2.waitKey()
  cv2.destroyAllWindows()

def finalEllipse(ells,COL_L_CUR):
  f_d = removeDup(filterEllipse(ells), f=(lambda (c,foo,bar): c) )
  sort_c = sort(f_d, f = lambda x: x[0])
  sort_r = sort(f_d, f = lambda x: x[0][::-1])

  lines_v = lineFit(lines(sort_c,f=lambda ((x,y),fo,ba),((xp,yp),foo,bar): (abs(x - xp),(x,y))))
  lines_h = lineFit(lines(sort_r,f=lambda ((x,y),fo,ba),((xp,yp),foo,bar): (abs(y - yp),(x,y))))
  intersections = map(lambda h: map(lambda v: intersection(v,h),lines_v), lines_h)
  flattened = [item for sublist in intersections for item in sublist]
  final = sortByCol(sort(patch(sort_c,flattened),f = lambda x: x[0]))[:COL_L_CUR]
  return final


def answers(final,COL_W,img_o):
  ans = []
  groups = grouper(final,COL_W)
  for idx,g in enumerate(groups):
    (mean,std,mini,i) = stats(img_o,g)
    if mini < mean:
      a = str(unichr(i + 65 + 5 * (idx % 2) + int(idx % 2 == 1 and i == 3)))
      ans.append((idx + 1,a))
    else: 
      ans.append((idx + 1,None))
  return ans

