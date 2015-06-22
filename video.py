import datetime as dt
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

SCREEN_H = 900
SCREEN_W = 1440

def guideLines(image):
  cv2.rectangle(image,(int(SCREEN_W/2 - 509.09/2),SCREEN_H),(int(SCREEN_W/2 - 509.09/2 + 50),SCREEN_H - 50),(0,255,0),2)
  cv2.rectangle(image,(int(SCREEN_W/2 + 509.09/2),SCREEN_H),(int(SCREEN_W/2 + 509.09/2 + 50),SCREEN_H - 50),(0,255,0),2)
  cv2.rectangle(image,(int(SCREEN_W/2 - 509.09/2),0),(int(SCREEN_W/2 - 509.09/2 + 50),50),(0,255,0),2)
  cv2.rectangle(image,(int(SCREEN_W/2 + 509.09/2),0),(int(SCREEN_W/2 + 509.09/2 + 50),50),(0,255,0),2)
  return image
while(True):
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Our operations on the frame come here
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray = cv2.flip(gray,1)
  element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
  gray = cv2.morphologyEx(gray,cv2.MORPH_RECT,element)
  # Display the resulting frame
  cv2.imshow('frame',gray)
  ch = cv2.waitKey(1) & 0xFF 
  if ch == ord('q'):
    break
  elif ch == ord('n'):
    s = str(dt.datetime.now())
    cv2.imwrite(s+".png",gray)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
