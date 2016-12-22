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

class Line():
    def __init__(self, is_left):
        # is this the left or the right lane?
        self.is_left = is_left
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension

        self.yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image

    def _locate_lane(self, birdseye, starting_centre):
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

    def new_birdseye(self, birdseye):

        if not self.detected:
            print("detection lost")
            histogram = smooth(np.sum(birdseye[birdseye.shape[0]//2:,:], axis=0), k=51)

            # get the locations of the peaks as the starting points
            # use a very wide window (200) to ensure only the highest peaks
            starting_points = scipy.signal.find_peaks_cwt(histogram, np.array([200]))

            if self.is_left:
                if starting_points[0] < starting_points[1]:
                    start = starting_points[0]
                else:
                    start = starting_points[1]
            else:
                if starting_points[0] < starting_points[1]:
                    start = starting_points[1]
                else:
                    start = starting_points[0]
        else:
            start = self.fitx[-1]

        mask = self._locate_lane(birdseye, start)
        lane = (mask==1) & (birdseye==1)

        # Fit a second order polynomial to each lane line
        lane_pixels = np.nonzero(lane)
        self.allx = lane_pixels[1] # * xm_per_pix
        self.ally = lane_pixels[0] # * ym_per_pix
        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        self.fitx = self.current_fit[0]*self.yvals**2 + self.current_fit[1]*self.yvals + self.current_fit[2]

        curvature_fit = np.polyfit(self.ally * self.ym_per_pix, self.allx * self.xm_per_pix, 2)

        # calculate lane curvature at the bottom of the image
        self.radius_of_curvature = calculate_lane_curvature(720, curvature_fit)

        self.detected = True

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

        self.left = Line(is_left=True)
        self.right = Line(is_left=False)


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

    def sanity_check(self):
        return True

    def process(self, img):

        if (img.shape[0] != 1280 or img.shape[1] != 720):
            img = cv2.resize(img, (1280, 720))

        # Undistort the image
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        combined = self.find_candidate_lane_pixels(undist)

        birdseye = cv2.warpPerspective(combined, self.M, (self.cols, self.rows))

        self.left.new_birdseye(birdseye)
        self.right.new_birdseye(birdseye)

        if not self.sanity_check():
            # use previous values
            # mark the lanes as not detected
            pass

        if self.debug:
            f = plt.figure()
            plt.plot(self.left.allx, self.left.ally, 'o', color='red')
            plt.plot(self.right.allx, self.right.ally, 'o', color='blue')
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
            plt.plot(self.left.fitx, self.left.yvals, color='green', linewidth=3)
            plt.plot(self.right.fitx, self.right.yvals, color='green', linewidth=3)
            plt.gca().invert_yaxis() # to visualize as we do the images
            plt.show()


        # Create an image to draw the lines on
        warp_zero = np.zeros_like(birdseye).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left.fitx, self.left.yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right.fitx, self.right.yvals])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (self.cols, self.rows))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, str(self.left.radius_of_curvature), (10,650), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(result, str(self.right.radius_of_curvature), (1000,650), font, 1, (255,255,255), 2, cv2.LINE_AA)

        return result

