import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

import thresholds

def nothing(x):
    pass

cv2.namedWindow('GradX')
cv2.createTrackbar('thresh_min','GradX', 20, 255, nothing)
cv2.createTrackbar('thresh_max','GradX', 255, 255, nothing)

cv2.namedWindow('GradY')
cv2.createTrackbar('thresh_min','GradY', 20, 255, nothing)
cv2.createTrackbar('thresh_max','GradY', 255, 255, nothing)

cv2.namedWindow('GradMag')
cv2.createTrackbar('thresh_min','GradMag', 30, 255, nothing)
cv2.createTrackbar('thresh_max','GradMag', 255, 255, nothing)

cv2.namedWindow('GradDir')
cv2.createTrackbar('thresh_min','GradDir', 75, 180, nothing)
cv2.createTrackbar('thresh_max','GradDir', 150, 180, nothing)

cv2.namedWindow('Hue')
cv2.createTrackbar('thresh_min','Hue', 18, 180, nothing)
cv2.createTrackbar('thresh_max','Hue', 25, 180, nothing)

cv2.namedWindow('Saturation')
cv2.createTrackbar('thresh_min','Saturation', 43, 255, nothing)
cv2.createTrackbar('thresh_max','Saturation', 255, 255, nothing)

cv2.namedWindow('Value')
cv2.createTrackbar('thresh_min','Value', 220, 255, nothing)
cv2.createTrackbar('thresh_max','Value', 255, 255, nothing)

cv2.namedWindow('Combined')

if len(sys.argv) < 2:
    print("Usage: python test_thresholds.py filename")

image = cv2.imread(sys.argv[1])

img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
#img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

# convert the YUV image back to RGB format
image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

ksize = 21 # Choose a larger odd number to smooth gradient measurements

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (ksize, ksize), 1)

sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

while True:

    # Apply each of the thresholding functions
    gradx = thresholds.abs_sobel_thresh(sobelx,
                                       cv2.getTrackbarPos('thresh_min','GradX'),
                                       cv2.getTrackbarPos('thresh_max','GradX'))
    grady = thresholds.abs_sobel_thresh(sobely,
                                       cv2.getTrackbarPos('thresh_min','GradY'),
                                       cv2.getTrackbarPos('thresh_max','GradY'))
    mag_binary = thresholds.mag_thresh(sobelx, sobely,
                                      cv2.getTrackbarPos('thresh_min','GradMag'),
                                      cv2.getTrackbarPos('thresh_max','GradMag'))
    kernel = np.ones((5,5),np.uint8)
    mag_binary = cv2.morphologyEx(mag_binary, cv2.MORPH_CLOSE, kernel)
    mag_binary = cv2.morphologyEx(mag_binary, cv2.MORPH_OPEN, kernel)
    dir_binary = thresholds.dir_threshold(sobelx, sobely,
                               cv2.getTrackbarPos('thresh_min','GradDir') / 180. * np.pi / 2,
                               cv2.getTrackbarPos('thresh_max','GradDir') / 180. * np.pi / 2)
    kernel = np.ones((3,3),np.uint8)
    dir_binary = cv2.morphologyEx(dir_binary, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5,5),np.uint8)
    dir_binary = cv2.morphologyEx(dir_binary, cv2.MORPH_CLOSE, kernel)

    hue_binary = thresholds.threshold(h,
                                     cv2.getTrackbarPos('thresh_min','Hue'),
                                     cv2.getTrackbarPos('thresh_max','Hue'))

    saturation_binary = thresholds.threshold(s,
                                             cv2.getTrackbarPos('thresh_min','Saturation'),
                                             cv2.getTrackbarPos('thresh_max','Saturation'))

    value_binary = thresholds.threshold(v,
                                        cv2.getTrackbarPos('thresh_min','Value'),
                                        cv2.getTrackbarPos('thresh_max','Value'))

    combined = np.zeros_like(dir_binary)
    ## strong gradients in X and Y OR strong gradients between 90 and 120 degrees
    ## AND hue between 15 and 25 (the yellow slice in the hue cake also includes white) OR white (value 215 to 255 gives strong white )
    ## AND NOT saturation between 43 and 255 (kills everything off the road)
    combined[#  (((gradx == 1) & (grady == 1)) |
                ((mag_binary == 1) & (dir_binary == 1))
              & ((hue_binary == 1) | (value_binary == 1))
            # & (saturation_binary == 0)
            ] = 1

    cv2.imshow('Original', image)

    cv2.imshow('GradX', gradx)
    cv2.imshow('GradY', grady)
    cv2.imshow('GradMag', mag_binary)
    cv2.imshow('GradDir', dir_binary)

    cv2.imshow('Hue', hue_binary)
    cv2.imshow('Saturation', saturation_binary)
    cv2.imshow('Value', value_binary)

    cv2.imshow('Combined', combined)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(combined, cmap='gray')
ax2.set_title('Combined', fontsize=50)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()
