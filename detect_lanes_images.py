import cv2
import glob
import lane

image_files = glob.glob('./test_images/*.jpg')

detector = lane.LaneDetector("calibration.p", debug=False)

cv2.namedWindow('image')

for fname in image_files:
    print("loading ", fname)
    img = cv2.imread(fname)

    processed = detector.process(img)
    cv2.imshow('image', processed)
    cv2.waitKey(500)

cv2.destroyAllWindows()
