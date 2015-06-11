import cv2

def crop_test(test, cv2, image, width, height, t1_sec1, t1_sec2, t1_sec3, t1_sec4, t1_sec5, display=False):
    yyy = image[0:height, 0:t1_sec1]
    cv2.imwrite("saved_images/sections/t"+str(test)+"s1.png", yyy)
    if display == True:
        cv2.imshow("yyy", yyy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    yyy = image[0:height, t1_sec1:t1_sec2]
    cv2.imwrite("saved_images/sections/t"+str(test)+"s2.png", yyy)
    if display == True:
        cv2.imshow("yyy", yyy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    yyy = image[0:height, t1_sec2:t1_sec3]
    cv2.imwrite("saved_images/sections/t"+str(test)+"s3.png", yyy)
    if display == True:
        cv2.imshow("yyy", yyy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    yyy = image[0:height, t1_sec3:t1_sec4]
    cv2.imwrite("saved_images/sections/t"+str(test)+"s4.png", yyy)
    if display == True:
        cv2.imshow("yyy", yyy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    yyy = image[0:height, t1_sec4:t1_sec5]
    cv2.imwrite("saved_images/sections/t"+str(test)+"s5.png", yyy)
    if display == True:
        cv2.imshow("yyy", yyy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    yyy = image[0:height, t1_sec5:width]
    cv2.imwrite("saved_images/sections/t"+str(test)+"s6.png", yyy)
    if display == True:
        cv2.imshow("yyy", yyy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


