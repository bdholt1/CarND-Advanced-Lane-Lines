# This task is very similar to the OpenCV tutorial on camera calibration:
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
# Code has been used from there but modified to suit this project.

import numpy as np
import cv2
import glob
import pickle
import os

# termination criteria for subpixel refining
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
imgpoints = [] # 2d points in image plane.
objpoints = [] # 3d point in real world space

image_files = glob.glob('./camera_cal/*.jpg')


def find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # try to find chessboard points starting from 9x6 and working down to 3x3
    for ny in range(6, 3, -1):
        for nx in range(9, 3, -1):

            #print("finding ", nx, "x", ny, " corners for ", fname)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((nx*ny,3), np.float32)
            objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

            # Try to find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                print("found ", nx, "x", ny, " corners for ", fname)

                # use cornerSubPix to refine the corner locations (improves the undistortion)
                cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

                # return when chess corners are found
                return objp, corners

for fname in image_files:
    print("loading ", fname)
    img = cv2.imread(fname)

    objp, corners = find_corners(img)
    objpoints.append(objp)
    imgpoints.append(corners)

cv2.destroyAllWindows()

print("calibrating camera")
# calibrate the camera using the detected corners
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)

# pickle the camera matrix and distortion coefficients for subsequent use
calibration = { "mtx": mtx, "dist": dist}
pickle.dump(calibration, open( "calibration.p", "wb" ) )

# calculate the mean reprojection error
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    tot_error += error
print("mean reprojection error: ", tot_error/len(objpoints))

# undistort the calibration images to check that it's all working
for fname in image_files:
    img = cv2.imread(fname)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    savefile = "./camera_cal_undistorted/" + os.path.basename(fname)
    print("saving ", savefile)
    cv2.imwrite(savefile, undistorted)
    cv2.imshow('undistorted', undistorted)
    cv2.waitKey(500)

cv2.destroyAllWindows()
