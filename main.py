from find_tests_final import *
from ellipse import *

COL_W = [4,5,4,4]
COL_L = [300, 300, 160, 160]
COL_L_CUR = 300
    
if __name__ == '__main__':
    print __doc__


    print __doc__
    try:
        fn = sys.argv[1]
    except:
        fn = "test_filled4.jpeg"

print fn
tests = find_tests(fn)

imgs = map(getImages, tests)
final = [finalEllipse(getContours(img_c),COL_L[i]) for (i,(foo,img_c,bar)) in enumerate(imgs)]
for i in range(4):
  display(imgs[i][0],final[i],imgs[i][2],COL_W[i],"present %i.png" % (i + 1))
  print "#####TEST %s#####" %(i+1)
  print answers(final[i],COL_W[i],imgs[i][2])



