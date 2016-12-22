#!/usr/bin/python

import numpy as np
import cv2
import lane
import sys
import skvideo.io


filename = sys.argv[1]

# here you can set keys and values for parameters in ffmpeg
inputparameters = {}
outputparameters = {}
reader = skvideo.io.FFmpegReader(filename,
                inputdict=inputparameters,
                outputdict=outputparameters)

# when running lane detection on complete different frames
# use a newly initialised detector each time
# because previous values will not make sense
detector = lane.LaneDetector("calibration.p", debug=False)

for frame in reader.nextFrame():

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed = detector.process(bgr)

    # Display the resulting frame
    cv2.imshow('frame', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
