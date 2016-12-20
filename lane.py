import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import scipy.signal
import thresholds

def smooth(x, k=5):
    w = np.ones(k,'d')
    return np.convolve(x, w, mode='valid')

def calculate_lane_curvature(y, fit):
    curverad = ((1 + (2*fit[0]*y + fit[1])**2)**1.5) \
               / np.absolute(2*fit[0])
    return curverad

class LaneDetector:

    def __init__(self, calibration_file, debug=False):

        self.cols = 1280
        self.rows = 720

        # load the calibration file
        calibration = pickle.load( open(calibration_file , "rb" ) )
        self.mtx = calibration["mtx"]
        self.dist = calibration["dist"]

        # store whether to debug
        self.debug = debug

        # Set up the perspective transform to bird's-eye view
        # in order: top-left, top-right, bottom-right, bottom-left
        src = np.float32([(585, 462),
                          (703, 462),
                          (1033, 680),
                          (260, 680)])

        dst = np.float32([[self.cols/4, 0],
                          [3*self.cols/4, 0],
                          [3*self.cols/4, self.rows],
                          [self.cols/4, self.rows]])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)


    def find_candidate_lane_pixels(self, undistorted):
        ksize = 5
        gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)

        blur = cv2.GaussianBlur(gray, (ksize, ksize), 1)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize)

        gradx = thresholds.abs_sobel_thresh(sobelx, 20, 255)
        grady = thresholds.abs_sobel_thresh(sobely, 40, 255)
        sobels_binary = np.zeros(gray.shape, np.float32)
        sobels_binary[ ((gradx == 1) & (grady == 1)) ] = 1

        mags = thresholds.mag_thresh(sobelx, sobely, 30, 130)
        dirs = thresholds.dir_threshold(sobelx, sobely,
                                        70 / 180. * np.pi / 2,
                                        130 / 180. * np.pi / 2)
        dir_binary = np.zeros(gray.shape, np.float32)
        dir_binary[ ((mags == 1) & (dirs == 1)) ] = 1

        # Combine image masks into a lane detector mask
        gradient_binary = np.dstack((np.zeros_like(gray), sobels_binary, dir_binary))

        hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
        h = hls[:,:,0]
        l = hls[:,:,1]
        s = hls[:,:,2]
        h_binary = thresholds.threshold(h, 0, 100)
        s_binary = thresholds.threshold(s, 200, 255)

        color_binary = np.dstack(( np.zeros_like(h_binary), h_binary, s_binary))

        # Combine image masks into a lane detector mask
        combined_binary = np.zeros(gray.shape, np.float32)
        combined_binary[ ((gradx == 1) & (grady == 1))
                         | ((mags == 1) & (dirs == 1))
                         | (s_binary == 1)] = 1

        if self.debug:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.set_title('Stacked thresholds')
            ax1.imshow(color_binary)

            ax2.set_title('Combined S channel and gradient thresholds')
            ax2.imshow(combined_binary, cmap='gray')
            plt.show()

        return combined_binary


    def locate_lane(self, birdseye, starting_centre):
        lane_mask = np.zeros_like(birdseye)

        row_start = birdseye.shape[0]//10 * 9
        row_end = birdseye.shape[0]//10 * 10
        prev_centre = starting_centre
        col_start = prev_centre - 50
        col_end = prev_centre + 50
        lane_mask[row_start:row_end, col_start:col_end] = 1

        for i in range(8, -1, -1):
            row_start = birdseye.shape[0]//10 * i
            row_end = birdseye.shape[0]//10 * (i+1)

            best_col_centre = prev_centre
            best = 0
            for j in range(-50, 50):
                col_start = prev_centre - 50 + j
                col_end = prev_centre + 50 + j
                s = np.sum(birdseye[row_start:row_end, col_start:col_end])
                if s > best:
                    best = s
                    best_col_centre = prev_centre + j

            prev_centre = best_col_centre
            col_start = prev_centre - 50
            col_end = prev_centre + 50
            lane_mask[row_start:row_end, col_start:col_end] = 1

        return lane_mask


    def process(self, img):

        if (img.shape[0] != 1280 or img.shape[1] != 720):
            img = cv2.resize(img, (1280, 720))

        # Undistort the image
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        combined = self.find_candidate_lane_pixels(undist)

        birdseye = cv2.warpPerspective(combined, self.M, (self.cols, self.rows))

        histogram = smooth(np.sum(birdseye[self.rows//2:,:], axis=0), k=51)

        # get the locations of the peaks as the starting points
        # use a very wide window (200) to ensure only the highest peaks
        starting_lanes = scipy.signal.find_peaks_cwt(histogram, np.array([200]))

        left_mask = self.locate_lane(birdseye, starting_lanes[0])
        left_lane = (left_mask==1) & (birdseye==1)

        right_mask = self.locate_lane(birdseye, starting_lanes[1])
        right_lane = (right_mask==1) & (birdseye==1)

        lanes = np.dstack((np.zeros_like(birdseye), left_lane, right_lane))

        if self.debug:
            cv2.imshow('lanes', lanes)

        # Fit a second order polynomial to each fake lane line
        yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image
        left_lane_pixels = np.nonzero(left_lane)
        leftx = left_lane_pixels[1] # * xm_per_pix
        lefty = left_lane_pixels[0] # * ym_per_pix
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]

        right_lane_pixels = np.nonzero(right_lane)
        rightx = right_lane_pixels[1] # * xm_per_pix
        righty = right_lane_pixels[0] # * ym_per_pix
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

        if self.debug:
            f = plt.figure()
            plt.plot(leftx, lefty, 'o', color='red')
            plt.plot(rightx, righty, 'o', color='blue')
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
            plt.plot(left_fitx, yvals, color='green', linewidth=3)
            plt.plot(right_fitx, yvals, color='green', linewidth=3)
            plt.gca().invert_yaxis() # to visualize as we do the images
            plt.show()

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

        # calculate lane curvature at the bottom of the image
        left_lane_curvature = calculate_lane_curvature(720, left_fit_cr)
        right_lane_curvature = calculate_lane_curvature(720, right_fit_cr)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(birdseye).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (self.cols, self.rows))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, str(left_lane_curvature), (10,650), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(result, str(right_lane_curvature), (1000,650), font, 1, (255,255,255), 2, cv2.LINE_AA)

        return result

