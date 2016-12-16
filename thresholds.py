import numpy as np
import cv2

def threshold(image, thresh_min, thresh_max):
    # general thresholding function
    binary_output = np.zeros(image.shape, dtype=np.float32)
    binary_output[(image >= thresh_min) & (image <= thresh_max)] = 1
    return binary_output

def abs_sobel_thresh(sobel, thresh_min=0, thresh_max=255):
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    return threshold(scaled_sobel, thresh_min, thresh_max)

def mag_thresh(sobelx, sobely, thresh_min=0, thresh_max=255):
    mag_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
    return threshold(scaled_sobel, thresh_min, thresh_max)

def dir_threshold(sobelx, sobely, thresh_min=0, thresh_max=np.pi/2):    
    dir_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    return threshold(dir_sobel, thresh_min, thresh_max)
