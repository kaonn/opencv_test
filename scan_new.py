# import the necessary packages
#import four_point_transform function from transform_new.py file
from transform_new import four_point_transform
import numpy as np
import cv2

################################################################################
## The following code is strongly derived from the following link: http://www.
## pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example

## code performs edge detection to find contour of object, then uses contour
## to find four cornors of object, and then calls four_point_transform function
## to get rid of excess background behind object and display straigtened object
################################################################################

filename = "!snapshot.png"#"!snapshot.png"#"hand.jpg"#"bubblecolor.jpg"#"test.jpg"


# resize function directly from pyimagesearch.com
def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

"""
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
# filename = "test.jpg"#"sample.jpg"  
image = cv2.imread(filename)
# image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(image, height = 500)


# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
# edged = cv2.adaptiveThreshold(gray,255,\
#     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,75,10)
 

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
(cnts, _) = cv2.findContours(edged.copy(), 
	cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break
 
print screenCnt

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
warped_resize = resize(warped, height = 650)

# cv2.circle(warped,(50, 50),5,(150,100,255),3)


# show the original image and the edge detected image
print "STEP 1: Edge Detection"
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# show the contour (outline) of the piece of paper
print "STEP 2: Find contours of paper"
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# show the original and scanned images
print "STEP 3: Apply perspective transform"
cv2.imshow("Original", resize(orig, height = 650))
cv2.imshow("Scanned", warped_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



def reposition_object(filename):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    """# filename = "test.jpg"#"sample.jpg" """
    image = cv2.imread(filename)
    # image = cv2.imread(args["image"])
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = resize(image, height = 500)


    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    edged = cv2.adaptiveThreshold(gray,255,\
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,75,10)
     

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (cnts, _) = cv2.findContours(edged.copy(), 
        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    print "##"
    # loop over the contours
    for c in cnts:
        screenCnt = None
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
     
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt == None: 
        print "no contour detected"
        orig_resize = resize(orig, height = 650)
        new_filename = "!"+str(filename)
        cv2.imwrite(new_filename, orig_resize)
        return image

     

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    warped_resize = resize(warped, height = 650)
    #save file locally
    new_filename = "!"+str(filename)
    cv2.imwrite(new_filename, warped_resize)
    print "Saved"


"""
# reposition_object(filename)

# image_with_function = reposition_object(filename)
image = cv2.imread(filename)
# print image.shape
image = resize(image, height = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
edged = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,75,10)
cv2.imshow("image", image)
cv2.imshow("Scanned", edged)
final_image = cv2.imread("!"+str(filename))
cv2.imshow("final", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# cv2.imwrite("!!!.jpg", image_with_function)
