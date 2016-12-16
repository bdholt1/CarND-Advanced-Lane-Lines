import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import thresholds

def nothing(x):
    pass

cv2.namedWindow('GradX')
cv2.createTrackbar('thresh_min','GradX', 20, 255, nothing)
cv2.createTrackbar('thresh_max','GradX', 100, 255, nothing)

cv2.namedWindow('GradY')
cv2.createTrackbar('thresh_min','GradY', 40, 255, nothing)
cv2.createTrackbar('thresh_max','GradY', 100, 255, nothing)

cv2.namedWindow('GradMag')
cv2.createTrackbar('thresh_min','GradMag', 30, 255, nothing)
cv2.createTrackbar('thresh_max','GradMag', 130, 255, nothing)

cv2.namedWindow('GradDir')
cv2.createTrackbar('thresh_min','GradDir', 70, 180, nothing)
cv2.createTrackbar('thresh_max','GradDir', 130, 180, nothing)

cv2.namedWindow('Combined')

image = mpimg.imread('signs_vehicles_xygrad.png')

ksize = 7 # Choose a larger odd number to smooth gradient measurements

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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
    dir_binary = thresholds.dir_threshold(sobelx, sobely,
                               cv2.getTrackbarPos('thresh_min','GradDir') / 180. * np.pi / 2,
                               cv2.getTrackbarPos('thresh_max','GradDir') / 180. * np.pi / 2)


    # Try different combinations and see what you get. For example, here is a selection
    # for pixels where both the x and y gradients meet the threshold criteria, or the
    # gradient magnitude and direction are both within their threshold values.

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    cv2.imshow('GradX', gradx)
    cv2.imshow('GradY', grady)
    cv2.imshow('GradMag', mag_binary)
    cv2.imshow('GradDir', dir_binary)
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
