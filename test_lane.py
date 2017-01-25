import cv2
import glob
import lane

image_files = glob.glob('./test_images/*.jpg')

cv2.namedWindow('image')

for fname in image_files:
    print("loading ", fname)
    img = cv2.imread(fname)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # when running lane detection on complete different frames
    # use a newly initialised detector each time
    # because previous values will not make sense
    try:
        detector = lane.LaneDetector("calibration.p", debug=True, debug_file=fname)

        processed = detector.process(rgb)
    except:
        pass

cv2.destroyAllWindows()
