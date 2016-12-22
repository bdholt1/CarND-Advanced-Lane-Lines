import cv2
import glob
import lane

image_files = glob.glob('./test_images/*.jpg')

cv2.namedWindow('image')

for fname in image_files:
    print("loading ", fname)
    img = cv2.imread(fname)

    # when running lane detection on complete different frames
    # use a newly initialised detector each time
    # because previous values will not make sense
    detector = lane.LaneDetector("calibration.p", debug=True)

    processed = detector.process(img)
    cv2.imshow('image', processed)
    cv2.waitKey(500)

cv2.destroyAllWindows()
