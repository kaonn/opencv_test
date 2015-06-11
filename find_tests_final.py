# import the necessary packages
#import four_point_transform function from transform_new.py file
from four import *
from crop_test import *
import numpy as np
import cv2

################################################################################
## The following code is strongly derived from the following link: http://www.
## pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example

## code performs edge detection to find contour of object, then uses contour
## to find four cornors of object, and then calls four_point_transform function
## to get rid of excess background behind object and display straigtened object
################################################################################

# test_filled1 = "images/test31.png"
# test_filled2 = "images/test32.png"
left_angle = "images/test33.png"
right_angle = "images/test34.png"
left_view = "images/test36.png"
test_straight1 = "images/test37.png"
test_straight2 = "images/test38.png"
test_straight3 = "images/test39.png"
straight = "images/straight.png"

test_filled2 = "images/test_filled2.png"
test_filled2_1 = "images/test_filled2_1.png"
test_filled3 = "images/test_filled3.png"
test_filled3_1 = "images/test_filled3_1.png"
test_filled4 = "images/test_filled4.png"
test_filled4_1 = "images/test_filled4_1.png"
blank_ACT = "images/book_blank_ACT.png"
noob = "images/noob.png"
book1 = "images/book1.png"
book2 = "images/book2.png"



"""
image123 = cv2.imread(filename)
ttttt = resize(image123, height = 600)
cv2.imshow(filename, ttttt)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_copy = image123.copy()
# image = resize(image, height = 500)
"""

def find_sorted_contours(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 75, 200)
    (foo,threshold_image) = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

    
    zxc = resize(edged, height = 600)
    cv2.imshow("edged", zxc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    qwertyuiop = resize(threshold_image, height = 600)
    cv2.imshow("threshold_image", qwertyuiop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (cnts, _) = cv2.findContours(edged.copy(), 
        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    return cnts

def find_long_box_contour(contours, width, below_first_long_box):
    long_boxes_contours = []

    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
             
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        rightmostpoint = tuple(approx[approx[:,:,0].argmax()][0])
        leftmostpoint = tuple(approx[approx[:,:,0].argmin()][0])
        if len(approx)==2 and leftmostpoint[0]<width/4 and rightmostpoint[0]>3*width/4 and rightmostpoint[1]>below_first_long_box:  #and rightmostpoint[1]<height/2:
            long_box = approx
            long_boxes_contours.append(long_box)
            
    return long_boxes_contours

#sort contours by max y values (from min to max)
def sort_long_box_contours_key(contours):
    return contours[contours[:,:,1].argmax()][0][1]

def display_image_with_line(img, p1,p2):
    line_image = img
    ((a,b),(c,d)) = (p1,p2)
    cv2.line(line_image, (a,b),(c,d), [0,255,0], thickness=10)
    # cv2.circle(test, [actbox], -1, (0, 255, 0), 2)
    line_image = resize(line_image, height = 600)


def display_image_with_line_show(img, p1,p2):
    line_image = img
    ((a,b),(c,d)) = (p1,p2)
    cv2.line(line_image, (a,b),(c,d), [0,255,0], thickness=10)
    # cv2.circle(test, [actbox], -1, (0, 255, 0), 2)
    line_image = resize(line_image, height = 600)
    cv2.imshow("line_image", line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_left_right_points(contour):
    return (tuple(contour[contour[:,:,0].argmin()][0]),tuple(contour[contour[:,:,0].argmax()][0]))


def find_test_border(sorted_long_boxes_contours, previous_border):
    for x in sorted_long_boxes_contours:
        rightmostpoint = tuple(x[x[:,:,0].argmax()][0])
        leftmostpoint = tuple(x[x[:,:,0].argmin()][0])
        if leftmostpoint[1]>previous_border + 100: 
            test_border = x
            break
    return test_border

def find_tests(filename):
    image = cv2.imread(filename)
    (height, width) = image.shape[:2]
    cnts = find_sorted_contours(image)
    long_boxes_contours= find_long_box_contour(cnts, width, int(height*0.14))
    sorted_long_boxes_contours = sorted(long_boxes_contours, key = sort_long_box_contours_key)# reverse = True)#[:5]



    test1_border = sorted_long_boxes_contours[0]
    (left_t1_point, right_t1_point) = find_left_right_points(test1_border)

    # display_image_with_line(image, left_t1_point, right_t1_point)



    test2_border = find_test_border(sorted_long_boxes_contours, left_t1_point[1])
    (left_t2_point, right_t2_point) = find_left_right_points(test2_border)

        
    test3_border = find_test_border(sorted_long_boxes_contours, left_t2_point[1])
    (left_t3_point, right_t3_point) = find_left_right_points(test3_border)

        
    test4_border = find_test_border(sorted_long_boxes_contours, left_t3_point[1])
    (left_t4_point, right_t4_point) = find_left_right_points(test4_border)

        
    test5_border = find_test_border(sorted_long_boxes_contours, left_t4_point[1])
    (left_t5_point, right_t5_point) = find_left_right_points(test5_border)



    #####################################################
    ###                 TEST 1 Section                ###
    #####################################################
    t1_points = [left_t1_point, right_t1_point, left_t2_point , right_t2_point]
    test1section = four_point_transform(image, t1_points)


    #####################################################
    ###                 TEST 2 Section                ###
    #####################################################
    t2_points = [left_t2_point, right_t2_point, left_t3_point , right_t3_point]
    test2section = four_point_transform(image, t2_points)


    #####################################################
    ###                 TEST 3 Section                ###
    #####################################################
    t3_points = [left_t3_point, right_t3_point, left_t4_point, right_t4_point]
    test3section = four_point_transform(image, t3_points)


    #####################################################
    ###                 TEST 4 Section                ###
    #####################################################
    t4_points = [left_t4_point, right_t4_point, left_t5_point , right_t5_point]
    test4section = four_point_transform(image, t4_points)

    return [test1section,test2section,test3section,test4section]

















"""

###################################
### draw ACT_box + student box  ###
###################################
image123 = cv2.imread(filename)
(height, width) = image123.shape[:2]
cnts = find_sorted_contours(image123)
long_boxes_contours= find_long_box_contour(cnts, width, int(height*0.14))
sorted_long_boxes_contours = sorted(long_boxes_contours, key = sort_long_box_contours_key)# reverse = True)#[:5]

# show edge detected image123
print "STEP 1: Edge Detection"
# cv2.imshow("image123", image123)
gray = cv2.cvtColor(image123, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 75, 200)
edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

print len(sorted_long_boxes_contours)
test = edged
for xxxxx in long_boxes_contours: #sorted_long_boxes_contours:
    # if cv2.contourArea(xxxxx) > 10000:
    cv2.drawContours(test, [xxxxx], -1, (0, 255, 0), 2)
    # ttttt = resize(test, height = 750)
    # print xxxxx[xxxxx[:,:,1].argmin()][0][1]
    # cv2.imshow("Edged", ttttt)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
ttttt = resize(test, height = 750)
print xxxxx[xxxxx[:,:,1].argmin()][0][1]
cv2.imshow("Edged", ttttt)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
