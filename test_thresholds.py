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

cv2.namedWindow('L')
cv2.createTrackbar('thresh_min','L', 210, 255, nothing)
cv2.createTrackbar('thresh_max','L', 255, 255, nothing)

cv2.namedWindow('U')
cv2.createTrackbar('thresh_min','U', 110, 255, nothing)
cv2.createTrackbar('thresh_max','U', 130, 255, nothing)

cv2.namedWindow('V')
cv2.createTrackbar('thresh_min','V', 160, 255, nothing)
cv2.createTrackbar('thresh_max','V', 190, 255, nothing)

cv2.namedWindow('Combined')

if len(sys.argv) < 2:
    print("Usage: python test_thresholds.py filename")

image = cv2.imread(sys.argv[1])

luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
l = luv[:,:,0]
u = luv[:,:,1]
v = luv[:,:,2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ksize = 21 # Choose a larger odd number to smooth gradient measurements

blur = cv2.GaussianBlur(gray, (ksize, ksize), 1)

sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize)

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

    l_binary = thresholds.threshold(l,
                                     cv2.getTrackbarPos('thresh_min','L'),
                                     cv2.getTrackbarPos('thresh_max','L'))

    u_binary = thresholds.threshold(u,
                                    cv2.getTrackbarPos('thresh_min','U'),
                                    cv2.getTrackbarPos('thresh_max','U'))

    v_binary = thresholds.threshold(v,
                                    cv2.getTrackbarPos('thresh_min','V'),
                                    cv2.getTrackbarPos('thresh_max','V'))

    combined = np.zeros_like(dir_binary)
    ## strong gradients in X and Y AND strong gradients between 90 and 120 degrees
    # AND high lightness in LUV (good for white)
    # AND U and V thresholded (good for yellow)
    combined[#  (((gradx == 1) & (grady == 1)) |
              ((mag_binary == 1) & (dir_binary == 1)) | # gradients
               (l_binary == 1) | # white
              ((u_binary == 1) & (v_binary == 1)) # yellow
            ] = 1

    cv2.imshow('Original', image)
    cv2.imshow('Original L', l)
    cv2.imshow('Original U', u)
    cv2.imshow('Original V', v)

    cv2.imshow('GradX', gradx)
    cv2.imshow('GradY', grady)
    cv2.imshow('GradMag', mag_binary)
    cv2.imshow('GradDir', dir_binary)

    cv2.imshow('L', l_binary)
    cv2.imshow('U', u_binary)
    cv2.imshow('V', v_binary)

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
