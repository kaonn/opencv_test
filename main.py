from find_sections import getSections, getImage, getContours as gC
from find_tests_final import *
from ellipse import *
import os.path

COL_W = [4,5,4,4]
COL_L = [300, 300, 160, 160]
COL_L_CUR = 300
contour_threshold = 100

fn = ""
if __name__ == '__main__':
    print __doc__

    print __doc__
    try:
        fn = sys.argv[1]
    except:
        fn = "test_filled4.jpeg"

if not(os.path.isfile(fn)):
  raise Exception("No such file")

t = 100
points = []
while len(points) != 5 and t < 200:
  points = gC(fn,t)
  t += 1

if len(points) != 5:
  raise Exception("Can't parse sections")

tests = getSections(getImage(fn),points)
# tests = find_tests(fn)

cv2.namedWindow('GRADER') 

for i in range(4):
    (img_t,img_c,img_o) = getImages(tests[i],100)
    final = finalEllipse(getContours(img_c,img_t.copy()),COL_L[i],COL_W[i])
    print "#####TEST %s#####" %(i+1)
    print answers(final,COL_W[i],img_o)
    display(img_c,img_t,final,img_o,COL_W[i],"present %i.png" % (i + 1),'GRADER')
    
    ch = 0xFF & cv2.waitKey()
    if ch == ord('q'):
      break




