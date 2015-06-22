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

IMAGE_WIDTH_DEF = 2656
MIN_ROT = 60
MAX_ROT = 120
MIN_AREA = 500
MAX_AREA = 5000
ADAPT_THRESH_DEF = 75
FINE_FILTER = 0.9
MAX_WIDTH = 100
ELL_NUMBER = [300,160]
COL_GROUP_LENGTHS = [52,40,50,28,20]

def getImages(img, contour_threshold):
  h, w = img.shape[:2]
  img_o = cv2.cvtColor(img,cv2.cv.CV_BGR2GRAY)
  f = float(IMAGE_WIDTH_DEF)/float(w)
  img_o = cv2.resize(img_o,(0,0),fx = f,fy = f,interpolation = cv2.INTER_LANCZOS4)

  #cv2.imshow("f",img_o)
  #cv2.waitKey(0)

  img_c = cv2.resize(img,(0,0),fx = f,fy = f,interpolation = cv2.INTER_LANCZOS4)
  h, w = img_c.shape[:2]

  #for displaying purposes
  img_t = np.zeros(img_o.shape, np.uint8)
  img_c = img_c.astype(float)
  img_t = cv2.subtract(img_c,img_c * np.float32(0.999))

  #for detecting contours
  (thresh, img_c) = cv2.threshold(img_o, contour_threshold, 255, cv2.THRESH_BINARY)

  #for detecting bubbles
  img_o = cv2.adaptiveThreshold(img_o,255,cv2.THRESH_BINARY,cv2.ADAPTIVE_THRESH_MEAN_C,ADAPT_THRESH_DEF,0)

  return (img_t,img_c,img_o)

def getContours(img_c,img_t):
  contours0, hierarchy = cv2.findContours( img_c.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  contours = []
  for cnt in contours0:
    if len(cnt) > 7:
      e = cv2.fitEllipse(cnt)
      contours.append(e)
      cv2.ellipse(img_t,e,(0,255,0),2)
  #cv2.imshow("s",img_t)
  #cv2.waitKey(0)
  return contours


def filterEllipse(boxes):
  ar_a = map(lambda (x,(w,h),a): ((w,h),a,(x,(w,h),a)), boxes)
  l = filter(lambda ((w,h),a,b): MIN_AREA < w*h < MAX_AREA and MIN_ROT < a and a < MAX_ROT and h < MAX_WIDTH, ar_a)
  median = np.median(map(lambda (ar,a,b): ar,ar_a))
  std = np.std(map(lambda (ar,a,b): ar, l))
  mean = np.mean(map(lambda (ar,a,b): ar,l))
  ls = filter(lambda ((w,h),a,b): mean - FINE_FILTER * std < h*w < mean + FINE_FILTER * std,l)
  return map(lambda (ar,a,b):b,l)

def sort(ells,f):
  sort = sorted(ells, cmp = lambda x,y: epsilon(x,y,10), key = f)
  return sort

def whiteDensity(image,ellipse):
  ((x,y),(h,w),a) = ellipse
  xi,xf = np.uint16(x - w/2), np.uint16(x + w/2)
  yi,yf = np.uint16(y - h/2), np.uint16(y + h/2)

  m = np.asarray(image[yi:yf,xi:xf])
  m = m.astype(np.bool)
  return np.divide(np.sum(m), (w * h))

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

def isIn(elem, l, comp,e):
  count = 0
  for item in l:
    if comp(elem,item,e) == 0:
      return (True,count)
    count += 1
  return (False,None)

def removeDup(l,fc,fa,cmp,e):
  unique = []
  for elem in l:
    (b,i) = isIn(fc(elem),map(fc,unique),cmp,e)
    if not b:
      unique.append(elem)
    else:
      if fa(elem) > fa(unique[i]):
        unique[i] = elem
  return unique

def stats(img,ellg):
  l = map(lambda x: whiteDensity(img,x), list(ellg))
  mean = np.mean(l)
  std = np.std(l)
  (i,mini) = min(((i,w) for i,w in enumerate(l)), key = lambda (i,w): w)
  return (mean,std,mini,i)

def averageSize(ells):
  h_w = map(lambda (c,(h,w),a): (h,w), ells)
  ave = reduce(lambda (h1,w1),(h2,w2): ((h1 + h2) / 2,(w1 + w2) / 2), h_w, (0,0))
  return ave

def patch(ells,ints):
  centers = map(lambda (c,foo,bar): c, ells)
  ells_patched = []
  (h,w) = averageSize(ells)
  count = 0
  for i in ints:
    (b,idx) = isIn(i,centers,epsilon,10)
    if not(b):
      ells_patched.insert(count,(i,(h,w),90))
    else:
      (h1,w1) = centers[idx]
      if abs(h*w - h1*w1) < 20:
        ells_patched.insert(count,ells[idx])
      else:
        ells_patched.insert(count,(i,(h,w),90))
  return ells_patched

def linesFilter(group,correct):
  if len(group) < 4:
    return []
  elif len(group) == correct:
    return group
  else:
    l = len(group)
    return group[l - correct:]

def lines(ells,f,b,correct):
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
  f = filter(lambda x: len(x) > 2, cols)

  if not b:
    return f
  else:
    colValues = map(lambda l: (np.mean(map(lambda c: c[0],l)),l), f)
    colGroups = [[]]
    counter = 0
    colGroups[0].append(colValues[0][1])
    for j in range(1,len(colValues)):
      (xc,cur) = colValues[j]
      (xp,prev) = colValues[j - 1]
      if abs(xc - xp) < 80:
        colGroups[counter].append(cur)
      else:
        counter += 1
        colGroups.append([])
        colGroups[counter].append(cur)
    print counter
    final = map(lambda g: linesFilter(g,correct), colGroups)
    final = filter(lambda x: x,final)
    for g in final:
      print "\x1b[32m" + str(g) + "\x1b[0m" + "\n\n"
    return [item for sublist in final for item in sublist]
 

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

def grouper(iterable, n, fillvalue=((100,100),(50,50),90)):
  args = [iter(iterable)] * n
  return it.izip_longest(*args,fillvalue=fillvalue)

def display(img_c,image,contours,img_o,COL_W,filename,window):
  f_d = contours
  sort_c = sort(f_d, f = lambda x: x[0])
  sort_r = sort(f_d, f = lambda x: x[0][::-1])
  

  h, w = image.shape[:2]

  for points in lines(sort_c,f=lambda ((x,y),fo,ba),((xp,yp),foo,bar): (abs(x - xp),(x,y)) ,b = True,correct = COL_W):

    vx, vy, cx, cy = cv2.fitLine(np.asarray(points),cv2.cv.CV_DIST_L2,0,0.01,0.01)
    cv2.line(image,(int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)),(0,255,0),2)

  for points in lines(sort_r,f=lambda ((x,y),fo,ba),((xp,yp),foo,bar): (abs(y - yp),(x,y)) ,b = False,correct = COL_W):
    vx, vy, cx, cy = cv2.fitLine(np.asarray(points),cv2.cv.CV_DIST_L2,0,0.01,0.01)
    cv2.line(image,(int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)),(0,0,255),2)

#  if len(contours) not in ELL_NUMBER:
 #   raise Exception("Unexpected image!")

  groups = grouper(contours,COL_W)

  count = 1

  filled = []
  for i,g in enumerate(groups):
    (mean,std,mini,i) = stats(img_o,g)
    if mini < mean - std * 1.25 and mini < 0.4:
      filled.append((g[i],mini))
      #cv2.ellipse(image,g[i],(0,255,0),-1,cv2.CV_AA)
      

    j = 0
    for j in range(len(g)):
      (x,y) = g[j][False]
      cv2.putText(image,str(count),(int(x) - 15,int(y) + 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (255,0,0))
      cv2.ellipse(image,g[j],(0,255,0))
    count += 1

  densities = map (lambda (foo,d): d, filled)
  mean = np.mean(densities)
  std = np.std(densities)
  confident = filter(lambda (foo,d): d < mean + std, filled)
  diff = list(set(filled) - set(confident)) 
  probably = filter(lambda (foo,d) : d < 0.3, diff)

  for c in confident:
    cv2.ellipse(image,c[0],(0,255,0),-1,cv2.CV_AA)

  for c in diff:
    cv2.ellipse(image,c[0],(255,0,0),-1,cv2.CV_AA)

  for c in probably:
    cv2.ellipse(image,c[0],(0,0,255),-1,cv2.CV_AA)
  cv2.imshow(window, image)

def finalEllipse(ells,COL_L_CUR,COL_W):
  f_d = removeDup(filterEllipse(ells), fc=(lambda (c,foo,bar): c), fa=(lambda (foo,(h,w),bar): h*w),cmp = epsilon,e=10)
  sort_c = sort(f_d, f = lambda x: x[0])
  sort_r = sort(f_d, f = lambda x: x[0][::-1])

  lines_v = lineFit(lines(sort_c,f=lambda ((x,y),fo,ba),((xp,yp),foo,bar): (abs(x - xp),(x,y)), b=True,correct = COL_W))
  lines_h = lineFit(lines(sort_r,f=lambda ((x,y),fo,ba),((xp,yp),foo,bar): (abs(y - yp),(x,y)),b=False,correct = COL_W))
  intersections = map(lambda h: map(lambda v: intersection(v,h),lines_v), lines_h)
  flattened = [item for sublist in intersections for item in sublist]

  final = sortByCol(sort(patch(sort_c,flattened),f = lambda x: x[0]))[:COL_L_CUR]

  return final

def answers(final,COL_W,img_o):
  groups = grouper(final,COL_W)
  ans = []
  filled = []
  for idx,g in enumerate(groups):
    (mean,std,mini,i) = stats(img_o,g)
    if mini < mean - 1.25 * std and mini < 0.4:
      filled.append((i,mini,"CONF"))
    else:
      filled.append((i,mini,"NONE"))

  densities = map (lambda (foo,d,s): d, filled)
  mean = np.mean(densities)
  std = np.std(densities)
  confidence = map(lambda (foo,d,s): (foo,d,"CONF") if d < mean + std else ((foo,d,"PROB") if d < 0.3 else ((foo,d,"UNSURE")if s == "CONF" else (foo,d,s))), filled)
  for idx,c in enumerate(confidence):
    (i,d,s) = c
    if s == "CONF":
      a = str(unichr(i + 65 + 5 * (idx % 2) + int(idx % 2 == 1 and i == 3)))
      ans.append((idx + 1,a))
    else:
      ans.append((idx + 1,s))
  
  return ans
