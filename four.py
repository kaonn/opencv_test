# import the necessary packages
import numpy as np
import cv2
import math

################################################################################
## The following functions have been written by me but the approach is derived
## from the following link: http://www.pyimagesearch.com/2014/08/25/4-point-
## opencv-getperspective-transform-example/
################################################################################

#from www.pysearchimage.com
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


# take in array of 4 coordinates and order coordinates so first coordinate
# is top-left corner and rest are in order moving clockwise
def order_points(points):
    #create empty list of 4 coordinates 
    numOfPoints = 4
    valuesPerPoint = 2
    ordered_points = np.zeros((numOfPoints,valuesPerPoint), dtype= "float32")

    # add x and y componenets of each coordinate
    sumOfCoordinates = np.sum(points, axis = 1)
    #find difference of x and y components of each coordinate
    differenceOfCoordinates = np.diff(points, axis=1)

    # find smallest sum and difference of coordinates
    smallestSumIndex = np.argmin(sumOfCoordinates)
    smallestDifferenceIndex = np.argmin(differenceOfCoordinates)
    # find largest sum and difference of coordinates
    largestSumIndex = np.argmax(sumOfCoordinates)
    largestDifferenceIndex = np.argmax(differenceOfCoordinates)

    # top-left coordinate has smallest coordinate sum
    ordered_points[0] = points[smallestSumIndex]
    # top-left coordinate has smallest coordinate difference
    ordered_points[1] = points[smallestDifferenceIndex]
    # top-left coordinate has largest coordinate sum
    ordered_points[2] = points[largestSumIndex]
    # top-left coordinate has largest coordinate difference
    ordered_points[3] = points[largestDifferenceIndex]

    return ordered_points

def distance_between_points(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# takes in image and list of 4 coordinates for each corner of object in image, 
# and returns image of object in vertical position with excess background 
# removed
def four_point_transform(image, points):
    # unpack points in order
    ordered_points = order_points(points)
    top_left, top_right = ordered_points[0], ordered_points[1]
    bottom_right, bottom_left = ordered_points[2], ordered_points[3]

    # find the max width of the object in the image
    topWidth = int(distance_between_points(top_left, top_right))
    bottomWidth = int(distance_between_points(bottom_left, bottom_right))
    maxWidth = max(topWidth, bottomWidth)

    # find the max height of the object in the image
    topHeight = int(distance_between_points(top_left, bottom_left))
    bottomHeight = int(distance_between_points(top_right, bottom_right))
    maxHeight = max(topHeight, bottomHeight)

    # create array of corner points for final image
    new_top_left = [0, 0]
    new_top_right = [maxWidth - 1, 0]
    new_bottom_right = [maxWidth - 1, maxHeight - 1]
    new_bottom_left = [0, maxHeight - 1]
    new_coordinates = np.array([new_top_left, new_top_right, new_bottom_right,
     new_bottom_left], dtype = "float32")

    # calculate 3x3 matrix of a perspective transform
    M = cv2.getPerspectiveTransform(ordered_points, new_coordinates)
    # apply perspective transform matrix to image
    transformed_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the transformed image
    return transformed_image





# filename = "card.jpg" 
# image = cv2.imread(filename)
# # coordinates = [(20, 64), (132, 47), (155, 202), (40, 215)]
# coordinates = [(40, 215) ,(132, 47), (20, 64), (155, 202)]
# # coordinates = [[40, 215] ,[132, 47], [20, 64], [155, 202]]


# #array of coordinates of corners of object in image
# points = np.array(coordinates, dtype = "float32")


# # apply the four point tranformation to image
# transformedImage = four_point_transform(image, points)
 
# # show the original and warped images
# cv2.imshow("Original", image)
# cv2.imshow("Warped", transformedImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

