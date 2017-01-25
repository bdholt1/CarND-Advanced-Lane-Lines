import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import ransac
import scipy.signal
import os
import thresholds

def smooth(x, k=5):
    w = np.ones(k,'d')
    return np.convolve(x, w, mode='valid')

def calculate_lane_curvature(y, fit):
    curverad = ((1 + (2*fit[0]*y + fit[1])**2)**1.5) \
               / np.absolute(2*fit[0])
    return curverad

# Define conversions in x and y from pixels space to meters
ym_per_pixel = 30/720 # meters per pixel in y dimension
xm_per_pixel = 3.7/700 # meters per pixel in x dimension

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

        self.yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image

    def _locate_lane(self, birdseye, starting_centre):
        lane_mask = np.zeros_like(birdseye)

        row_start = int(birdseye.shape[0]/10 * 9)
        row_end = int(birdseye.shape[0]/10 * 10)
        prev_centre = int(starting_centre)
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

    def find_lane_from_birdseye(self, birdseye):

        if not self.detected:
            print("detection lost")
            histogram = smooth(np.sum(birdseye[birdseye.shape[0]//2:,:], axis=0), k=51)

            # get the locations of the peaks as the starting points
            # use a very wide window (200) to ensure only the highest peaks
            starting_points = scipy.signal.find_peaks_cwt(histogram, np.array([200]))

            if len(starting_points) < 2:
                print("Error: Could not detect 2 peaks in histogram")
                self.detected = False
                return False

            if self.is_left:
                start = min(starting_points[0], starting_points[1])
            else:
                start = max(starting_points[0], starting_points[1])
        else:
            start = self.fitx[-1]

        mask = self._locate_lane(birdseye, start)
        lane = (mask==1) & (birdseye==1)

        # Fit a second order polynomial to each lane line
        lane_pixels = np.nonzero(lane)
        self.allx = lane_pixels[1] # * xm_per_pixel
        self.ally = lane_pixels[0] # * ym_per_pixel

        if len(self.allx) < 10:
            self.detected = False
            return

        """
        # code to fit a polynomial using RANSAC
        model = ransac.PolynomialModel(degree=2, debug=False)

        all_data = np.hstack( (self.ally[:, np.newaxis], self.allx[:, np.newaxis]) ) # fit the model x = Ay^2 + By + C
        # run RANSAC algorithm
        n_samples = all_data.shape[0]
        ransac_fit, ransac_data = ransac.ransac(all_data,        #data - a set of observed data points
                                                model,           #model - a model that can be fitted to data points
                                                n_samples//10,   #n - the minimum number of data values required to fit the model
                                                1000,            #k - the maximum number of iterations allowed in the algorithm
                                                7e3,             #t - a threshold value for determining when a data point fits a model
                                                n_samples//2,    #d - the number of close data values required to assert that a model fits well to data
                                                debug=False,     #
                                                return_all=True) #
        self.current_fit = ransac_fit
        """

        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        self.fitx = np.polyval(self.current_fit, self.yvals)

        curvature_fit = np.polyfit(self.ally * ym_per_pixel, self.allx * xm_per_pixel, 2)

        # calculate lane curvature at the bottom of the image
        self.radius_of_curvature = calculate_lane_curvature(720, curvature_fit)

        # calcuate the number of meters the lane is from the centre of the image
        self.line_base_pos = (1280/2 - np.polyval(self.current_fit, 720)) * xm_per_pixel

        self.detected = True

class LaneDetector:

    def __init__(self, calibration_file, debug=False, debug_file=None):

        self.cols = 1280
        self.rows = 720

        # load the calibration file
        calibration = pickle.load( open(calibration_file , "rb" ) )
        self.mtx = calibration["mtx"]
        self.dist = calibration["dist"]

        # store whether to debug
        self.debug = debug
        self.debug_file = os.path.basename(debug_file)

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

        # calculate the perspective transform once at the beginning
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        self.left = Line(is_left=True)
        self.right = Line(is_left=False)

        # Use the average of the last 10 detections for display
        self.N = 10
        self.weights = 0.5 * np.exp(-0.5 * np.arange(self.N))
        self.weights /= np.sum(self.weights)

    def find_candidate_lane_pixels(self, undistorted):
        ksize = 21
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (ksize, ksize), 1)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize)

        gradx = thresholds.abs_sobel_thresh(sobelx, 20, 255)
        grady = thresholds.abs_sobel_thresh(sobely, 40, 255)
        sobels_binary = np.zeros(gray.shape, np.float32)
        sobels_binary[ ((gradx == 1) & (grady == 1)) ] = 1

        mag_binary = thresholds.mag_thresh(sobelx, sobely, 30, 255)
        kernel = np.ones((5,5),np.uint8)
        mag_binary = cv2.morphologyEx(mag_binary, cv2.MORPH_CLOSE, kernel)
        mag_binary = cv2.morphologyEx(mag_binary, cv2.MORPH_OPEN, kernel)

        dir_binary = thresholds.dir_threshold(sobelx, sobely,
                                              70 / 180. * np.pi / 2,
                                              130 / 180. * np.pi / 2)
        kernel = np.ones((3,3),np.uint8)
        dir_binary = cv2.morphologyEx(dir_binary, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((5,5),np.uint8)
        dir_binary = cv2.morphologyEx(dir_binary, cv2.MORPH_CLOSE, kernel)

        luv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2LUV)
        l = luv[:,:,0]
        u = luv[:,:,1]
        v = luv[:,:,2]
        l_binary = thresholds.threshold(l, 210, 255)
        u_binary = thresholds.threshold(u, 110, 130)
        v_binary = thresholds.threshold(v, 160, 190)

        combined = np.zeros_like(dir_binary)
        ## strong gradients in X and Y AND strong gradients between 90 and 120 degrees
        # AND high lightness in LUV (good for white)
        # AND U and V thresholded (good for yellow)
        combined[#  (((gradx == 1) & (grady == 1)) |
            ((mag_binary == 1) & (dir_binary == 1)) | # gradients
            (l_binary == 1) | # white
            ((u_binary == 1) & (v_binary == 1)) # yellow
        ] = 1

        if self.debug:
            print("plotting lane pixels")
            plt.clf()
            f, axes = plt.subplots(1, 5, figsize=(200,100))

            rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            axes[0].set_title('original')
            axes[0].imshow(rgb)

            axes[1].set_title('gradients')
            axes[1].imshow(((mag_binary == 1) & (dir_binary == 1)), cmap='gray')

            axes[2].set_title('white')
            axes[2].imshow((l_binary == 1), cmap='gray')

            axes[3].set_title('yellow')
            axes[3].imshow(((u_binary == 1) & (v_binary == 1)), cmap='gray')

            axes[4].set_title('combined')
            axes[4].imshow(combined, cmap='gray')

            #plt.show()
            plt.savefig("test_images_output/" + self.debug_file + "_lane_pixels.png")

        return combined

    def sanity_check(self):
        # start on the assumption that the lanes are 'sane'
        sane = True

        # Check that the radius of curvature of both of the lanes are 'similar'
        # Because ROC varies from 0 to inf, it's hard to find a threshold to use that would
        #   make sense - sample values are 2000 to 3000.
        # Instead, compare 1/ROC for the left and right lanes.
        curvature_similar = (np.abs(1./self.left.radius_of_curvature - 1./self.right.radius_of_curvature) < 0.1)
        if not curvature_similar:
            print("Curvatures are not similar")
        sane &= curvature_similar

        # Check that the lanes are more or less parallel, which I'm
        # assuming means that the std dev of the distances between the x values
        # of the lines is within 10cm
        dist_apart = []
        for i in range(720, 360, -1):
            x_left = np.polyval(self.left.current_fit, i) *  xm_per_pixel
            x_right = np.polyval(self.right.current_fit, i) * xm_per_pixel
            dist = x_right - x_left
            dist_apart.append(dist)

        lines_parallel = (np.std(dist_apart) < 0.1)
        if not lines_parallel:
            print("Lines are not parallel")
        sane &= lines_parallel

        # Check that the lines are roughly the right distance apart, which I'm
        # assuming to be 1m either side of 3.7m
        lines_correct_distance =  (np.abs(3.7 - np.mean(dist_apart)) < 1)
        if not lines_correct_distance:
            print("Lines are not within 1m of 3.7m")
        sane &= lines_correct_distance

        return sane

    def process(self, img):

        if (img.shape[0] != 1280 or img.shape[1] != 720):
            img = cv2.resize(img, (1280, 720))

        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Undistort the image
        undist = cv2.undistort(bgr, self.mtx, self.dist, None, self.mtx)

        combined = self.find_candidate_lane_pixels(undist)

        birdseye = cv2.warpPerspective(combined, self.M, (self.cols, self.rows))

        if self.debug:
            print("plotting birdseye")
            plt.clf()
            f, axes = plt.subplots(1, 3, figsize=(200,100))

            rgb = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
            axes[0].set_title('original')
            axes[0].imshow(rgb)

            axes[1].set_title('lane pixels')
            axes[1].imshow(combined, cmap='gray')

            axes[2].set_title('birdseye')
            axes[2].imshow(birdseye, cmap='gray')

            #plt.show()
            plt.savefig("test_images_output/" + self.debug_file + "_birdseye.png")

        self.left.find_lane_from_birdseye(birdseye)
        self.right.find_lane_from_birdseye(birdseye)

        if self.debug:
            print("plotting curve")
            plt.clf()
            f = plt.figure()
            if self.left.detected:
                plt.plot(self.left.allx, self.left.ally, 'o', color='red')
                plt.plot(self.left.fitx, self.left.yvals, color='green', linewidth=3)
            if self.right.detected:
                plt.plot(self.right.allx, self.right.ally, 'o', color='blue')
                plt.plot(self.right.fitx, self.right.yvals, color='green', linewidth=3)
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
            plt.gca().invert_yaxis() # to visualize as we do the images
            #plt.show()
            plt.savefig("test_images_output/" + self.debug_file + "_curve_fitting.png")

        if not self.sanity_check():
            print("Lane detection failed sanity check!")

            self.left.detected = False
            self.right.detected = False
        else:
            # Since the lane detection is sane, append the x values to the appropriate list
            self.left.recent_xfitted.append(self.left.fitx)
            self.right.recent_xfitted.append(self.right.fitx)

            if len(self.left.recent_xfitted) < self.N:
                # for the first few frames use the mean of the recent fits
                self.left.bestx = np.mean(np.array(self.left.recent_xfitted[-self.N:]), axis=0)
                self.right.bestx = np.mean(np.array(self.right.recent_xfitted[-self.N:]), axis=0)
            else:
                # when we have enough frames, use a weighted average
                self.left.bestx = np.average(np.array(self.left.recent_xfitted[-self.N:]), axis=0, weights=self.weights)
                self.right.bestx = np.average(np.array(self.right.recent_xfitted[-self.N:]), axis=0, weights=self.weights)


        # Create an image to draw the lines on
        warp_zero = np.zeros_like(birdseye).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        if self.left.bestx is not None:
            pts_left = np.array([np.transpose(np.vstack([self.left.bestx, self.left.yvals]))])
            pts_middle = np.transpose(np.vstack([(self.left.bestx + self.right.bestx)/2.0,
                                                 (self.left.yvals + self.right.yvals)/2.0]))
            pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right.bestx, self.right.yvals])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
            cv2.polylines(color_warp, np.int_([pts_middle]), False, (255,0,0), thickness=3)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (self.cols, self.rows))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, "{0:.2f}".format(self.left.radius_of_curvature), (10,650), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(result, "{0:.2f}".format(self.right.radius_of_curvature), (1100,650), font, 1, (255,255,255), 2, cv2.LINE_AA)
        position_off_centre = (self.left.line_base_pos + self.right.line_base_pos) / 2.0
        cv2.putText(result, "{0:.3f}".format(position_off_centre) + "m", (600,650), font, 1, (255,255,255), 2, cv2.LINE_AA)

        rgb_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return rgb_result

