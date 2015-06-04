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

def make_image():
    img = np.zeros((500, 500), np.uint8)
    black, white = 0, 255
    for i in xrange(6):
        dx = (i%2)*250 - 30
        dy = (i/2)*150

        if i == 0:
            for j in xrange(11):
                angle = (j+5)*np.pi/21
                c, s = np.cos(angle), np.sin(angle)
                x1, y1 = np.int32([dx+100+j*10-80*c, dy+100-90*s])
                x2, y2 = np.int32([dx+100+j*10-30*c, dy+100-30*s])
                cv2.line(img, (x1, y1), (x2, y2), white)

        cv2.ellipse( img, (dx+150, dy+100), (100,70), 0, 0, 360, white, -1 )
        cv2.ellipse( img, (dx+115, dy+70), (30,20), 0, 0, 360, black, -1 )
        cv2.ellipse( img, (dx+185, dy+70), (30,20), 0, 0, 360, black, -1 )
        cv2.ellipse( img, (dx+115, dy+70), (15,15), 0, 0, 360, white, -1 )
        cv2.ellipse( img, (dx+185, dy+70), (15,15), 0, 0, 360, white, -1 )
        cv2.ellipse( img, (dx+115, dy+70), (5,5), 0, 0, 360, black, -1 )
        cv2.ellipse( img, (dx+185, dy+70), (5,5), 0, 0, 360, black, -1 )
        cv2.ellipse( img, (dx+150, dy+100), (10,5), 0, 0, 360, black, -1 )
        cv2.ellipse( img, (dx+150, dy+150), (40,10), 0, 0, 360, black, -1 )
        cv2.ellipse( img, (dx+27, dy+100), (20,35), 0, 0, 360, white, -1 )
        cv2.ellipse( img, (dx+273, dy+100), (20,35), 0, 0, 360, white, -1 )
    return img

if __name__ == '__main__':
    print __doc__

    img = cv2.imread('test1.png',cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # img = cv2.resize(img,(500,500))
    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    h, w = img.shape[:2]

    contours0, hierarchy = cv2.findContours( img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    print contours0[1], contours0[0], contours0[2]
    contours = [cv2.approxPolyDP(cnt, 1, True) for cnt in contours0]
    # contours = [cv2.fitEllipse(cnt) for cnt in contours0]

    def update(levels):
        vis = np.zeros((h, w, 3), np.uint8)
        levels = levels - 3
        cv2.drawContours( vis, contours, (-1, 3)[levels <= 0], (128,255,255),
            1, cv2.CV_AA, hierarchy, abs(levels) )
        cv2.resize(vis,(0,0),fx=0.5,fy=0.5)
        cv2.imshow('contours', vis)
    update(3)
    cv2.createTrackbar( "levels+3", "contours", 3, 7, update )
    cv2.resize(img,(0,0),fx=0.5,fy=0.5)
    cv2.imshow('image', img)
    0xFF & cv2.waitKey()
    cv2.destroyAllWindows()
