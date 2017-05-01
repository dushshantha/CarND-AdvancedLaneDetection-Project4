import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from moviepy.editor import VideoFileClip
from collections import deque

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=10)
        #average x values of the fitted line over the last n iterations
        self.bestx = np.zeros(720)
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = np.zeros(3)
        # Polinomial coeffidients of the last n iterations
        self.recent_fit = deque(maxlen=10)
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
        self.smoothen_nframes = 10
        # first frame
        self.first_frame = True

    def add_best_xfit(self, xfit):
        if xfit.size > 0:
            self.recent_xfitted.append(xfit)
        #Update best fit
        self.bestx = np.mean(self.recent_xfitted, axis=0)

    def add_best_fit(self,nfit):
        self.recent_fit.append(nfit)
        self.best_fit = np.mean(self.recent_fit, axis=0)
        #print(self.best_fit)

global Left
global Right

Left  = Line()
Right = Line()

def read_image(filename):
    # Read in an image
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def convertHLS(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls

def getSLayer(hls):
    return hls[:,:,2]

def getLLayer(hls):
    return hls[:,:,1]



def print_Image(img, heading):
    plt.figure(figsize=(5, 5))
    plt.title(heading)
    plt.imshow(img, cmap='gray')
    plt.show()


def saveImage(img, filename):
    plt.imshow(img)
    plt.savefig(filename)
    #os.listdir("test_images/")


def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by \"vertices\" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero\n",
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def find_corners(img, nx, ny):
    # Find the chessboard corners
    image =  np.copy(img)
    gray = grayscale(image)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
        print_Image(image)

    return ret, corners


def calibration_points(filename, len_points, high_points):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((len_points*high_points,3), np.float32)
    objp[:,:2] = np.mgrid[0:len_points, 0:high_points].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(filename)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (len_points,high_points), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (len_points,high_points), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

    #cv2.destroyAllWindows()
    return objpoints, imgpoints


def calibrate_cam(img_name, objpoints, imgpoints, filename):
    img = cv2.imread(img_name)
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(filename, "wb"))

    return ret, mtx, dist, rvecs, tvecs


def load_calibrations(filename):
    calibrations = pickle.load(open(filename, "rb"))
    #print(calibrations)
    mtx = calibrations['mtx']
    dist = calibrations['dist']
    return mtx, dist


def undistortImage(img,   mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def perspective_trans(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size)
    return warped

def apply_binary_threshold(img, thresh=(0, 255)):
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Apply threshold
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    abs_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_mag = np.uint8(255 * abs_mag / np.max(abs_mag))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_mag)
    mag_binary[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1

    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_grad = np.arctan2(abs_sobel_y, abs_sobel_x)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(dir_grad)
    dir_binary[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    return dir_binary

def get_transform_points(img):
    imshape = img.shape
    '''
    src = np.float32([[490, 482], [810, 482],
                      [1250, 720], [0, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                      [1250, 720], [40, 720]])
    '''
    src = np.float32([[int(imshape[1] * 0.10) - 20, imshape[0]], [int(imshape[1] * 0.50) - 55, int(imshape[0] * 0.6)+ 20],
                      [int(imshape[1] * 0.55) + 45 , int(imshape[0] * 0.6) + 20 ], [imshape[1] * .95 + 20, imshape[0]]])
    dst = np.float32([[int(imshape[1] * 0.20), imshape[0]], [int(imshape[1] * 0.20), 0],
                      [int(imshape[1] * 0.75), 0], [int(imshape[1] * 0.75), imshape[0]]])

    return src, dst

def hist(img):
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    plt.plot(histogram)
    plt.show()


def return_Curvature(left_fitx, right_fitx, ploty, image_size):
    y_val = np.max(ploty)
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    left_curvature = ((1 + (2 * left_fit_cr[0] * y_val * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curvature = ((1 + (2 * right_fit_cr[0] * y_val * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    scene_height = image_size[0] * ym_per_pix
    scene_width = image_size[1] * xm_per_pix

    left_intercept = left_fit_cr[0] * scene_height ** 2 + left_fit_cr[1] * scene_height + left_fit_cr[2]
    right_intercept = right_fit_cr[0] * scene_height ** 2 + right_fit_cr[1] * scene_height + right_fit_cr[2]
    calculated_center = (left_intercept + right_intercept) / 2.0

    lane_deviation = (calculated_center - scene_width / 2.0)
    return left_curvature, right_curvature, lane_deviation


def sliding_windows(binary_warped):

    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    Left.current_fit = left_fit
    Left.add_best_fit(left_fit)
    right_fit = np.polyfit(righty, rightx, 2)
    Right.current_fit = right_fit
    Right.add_best_fit(right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    l_curv, r_curv, dev = return_Curvature(left_fitx, right_fitx, ploty, binary_warped.shape)
    print(l_curv)
    print(r_curv)
    return left_fitx, right_fitx, ploty, left_fit, right_fit, l_curv, r_curv, dev



def  skip_sliding_windows(binary_warped):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    imshape = binary_warped.shape
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_fit = Left.current_fit
    right_fit = Right.current_fit
    #print(right_fit)
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each

    if leftx.size > 5 and lefty.size > 5:
        left_fit = np.polyfit(lefty, leftx, 2)
        Left.current_fit = left_fit
        Left.add_best_fit(left_fit)
        Left.detected = True
        left_fit = Left.best_fit
        right_fit = np.polyfit(righty, rightx, 2)
        Right.current_fit = right_fit
        Right.add_best_fit(right_fit)
        Right.detected = True
        right_fit = Right.best_fit
    else:
        left_fit = Left.best_fit
        Left.detected = False
        right_fit = Right.best_fit
        Right.detected = False

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fitx, right_fitx, ploty


def draw_on_original_image(orig_undist, left_fitx, right_fitx, ploty, s, d):
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])

    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    line_pts = np.hstack((left_line, right_line))

    #Perspective Transform
    transformed = perspective_trans(orig_undist, s, d)
    new_image = np.zeros_like(transformed)

    cv2.fillPoly(new_image, np.int_([line_pts]), (0, 255, 0))
    unwarped = perspective_trans(new_image, d, s)

    result = cv2.addWeighted(orig_undist, 1, unwarped, 0.3, 0)
    return result


def process_image(image):
    ksize = 7
    src, dst = get_transform_points(image)
    vertices = np.array([[(int(src[0][0]) - 20 , int(src[0][1])), (int(src[1][0]) - 15 , int(src[1][1])),
                          (int(src[2][0]), int(src[2][1]) + 15 ), (int(src[3][0]) + 20, int(src[3][1]))]],
                        dtype=np.int32)

    # Step 1 : Undistort images
    # Load the calibration data
    mtx, dist = load_calibrations('wide_dist_pickle.p')
    undist = undistortImage(image, mtx, dist)

    # Step 2: Convert to Gray scale and HLS
    gray = grayscale(undist)
    s = getSLayer(convertHLS(undist))

    # Step 3: Apply Gradient Thresholding and S Change
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    gradx = region_of_interest(gradx, vertices)
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    grady = region_of_interest(grady, vertices)
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(20, 100))
    mag_binary = region_of_interest(mag_binary, vertices)
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, np.pi / 2))
    dir_binary = region_of_interest(dir_binary, vertices)

    s_binary = apply_binary_threshold(s, (175, 255))

    s_binary = region_of_interest(s_binary, vertices)

    l = getLLayer(convertHLS(undist))
    l_binary = apply_binary_threshold(l, (175, 255))
    l_binary = region_of_interest(l_binary, vertices)

    # Combining the color and thresholds
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) ) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    color_combined = np.zeros_like(s_binary)
    color_combined[(combined == 1) | ((s_binary == 1) & (l_binary == 1))] = 1

    # Step 4: Perspective Transform
    transformed = perspective_trans(color_combined, src, dst)

    # Step 5: Apply Sliding Window
    # Here, the sliding Window will be called only first time and the the results will be used to find the next set of
    # lane segmants using the results of that.
    #global left_fit, right_fit
    #left_fitx, right_fitx, ploty, window_img = sliding_windows(transformed)
    left_fitx, right_fitx, ploty, left_fit, right_fit, l, r, d = sliding_windows(transformed)
    '''
    if Left.detected != True and Right.detected != True:
        left_fitx, right_fitx, ploty, window_img = sliding_windows(transformed)
    else:
        left_fitx, right_fitx, ploty = skip_sliding_windows(transformed)
    '''
    #left_fitx, right_fitx, ploty, radius_of_curvature, line_base_pos = for_sliding_window(transformed)

    final = draw_on_original_image(undist, left_fitx, right_fitx, ploty, src, dst)
    return final


def draw_lines(img, src):

    # print(src)
    copy = np.copy(img)
    cv2.line(copy, tuple(src[0]), tuple(src[1]),color=[255, 0, 0], thickness = 3)
    cv2.line(copy, tuple(src[1]), tuple(src[2]),color=[255, 0, 0], thickness = 3)
    cv2.line(copy, tuple(src[2]), tuple(src[3]),color=[255, 0, 0], thickness = 3)
    cv2.line(copy, tuple(src[3]), tuple(src[0]),color=[255, 0, 0], thickness = 3)
    #print_Image(copy)






if __name__ == "__main__":


    '''
    # Calibration of the camera
    # Load the Calibration images and get the object ppoints and image points
    objpoints, imgpoints = calibration_points('camera_cal/calibration*.jpg', 9, 6)
    #print(objpoints)
    
    # Calibrate the camera and getting the image matrix and Distortion matrix ans also, Save the matrix for later usage 
    # without having to run though the whole calibration process again.  
    ret, mtx, dist, rvecs, tvecs = calibrate_cam('camera_cal/calibration1.jpg', objpoints, imgpoints, 'wide_dist_pickle.p')
    #print(mtx) '''

    ksize = 15

    # Load the calibration data
    mtx, dist = load_calibrations('wide_dist_pickle.p')
    print(mtx)

    # Undistort the image
    img = read_image('camera_cal/calibration1.jpg')
    saveImage(img,'images/image1.png')
    undist = undistortImage(img, mtx, dist)
    # print(undist.shape)
    saveImage(undist, 'images/image2.png')

    '''
    # Undistort the image
    img = read_image('test_images/test6.jpg')
    undist = undistortImage(img, mtx, dist)
    print_Image(undist, 'Original')
    #print(undist.shape)
    #saveImage(undist, 'undistorted_chess.jpg')

    src, dst = get_transform_points(undist)
    vertices = np.array([[(int(src[0][0]) - 20 , int(src[0][1])), (int(src[1][0]) - 15, int(src[1][1])),
                          (int(src[2][0]), int(src[2][1]) + 15 ), (int(src[3][0]) + 20, int(src[3][1]))]],
                        dtype=np.int32)
    #masked_image = region_of_interest(undist, vertices)
    #print_Image(undist)

    gray = grayscale(undist)
    s = getSLayer(convertHLS(undist))
    print_Image(s, 'S Layer')

    l = getLLayer(convertHLS(undist))
    l_binary = apply_binary_threshold(l, (175,255))
    l_binary = region_of_interest(l_binary, vertices)
    print_Image(l_binary, 'L Layere Binary')

    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    gradx = region_of_interest(gradx, vertices)
    print_Image(gradx, 'Grad X')
    grady = abs_sobel_thresh(l, orient='y', sobel_kernel=ksize, thresh=(10, 150))
    grady = region_of_interest(grady, vertices)
    print_Image(grady, 'Grad Y')
    mag_binary = mag_thresh(l, sobel_kernel=ksize, mag_thresh=(20, 100))
    mag_binary = region_of_interest(mag_binary, vertices)
    print_Image(mag_binary, 'Mag Binary')
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, np.pi / 2))
    dir_binary = region_of_interest(dir_binary, vertices)
    print_Image(dir_binary, 'Dir BInary')

    s_binary = apply_binary_threshold(s, (175,255))
    s_binary = region_of_interest(s_binary, vertices)
    print_Image(s_binary, 'S Binary')

    #Combining the color and thresholds
    combined = np.zeros_like(s_binary)
    combined[((gradx == 1)) | ((mag_binary == 1))] = 1

    #print_Image(combined, 'Combined')

    color_combined = np.zeros_like(s_binary)
    color_combined[(gradx == 1) | ((s_binary == 1))] = 1
    print_Image(color_combined, 'Color Combined')

    #draw_lines(undist, src)
    #tr_img = perspective_trans(undist, src, dst)
    #draw_lines(tr_img, dst)


    transformed = perspective_trans(color_combined, src, dst)
    #print_Image(combined)
    print_Image(transformed, 'Trans')
    #hist(transformed)
    #global left_fit, right_fit
    left_fitx, right_fitx, ploty, left_fit, right_fit, l, r = find_lane_lines(transformed)
    print(left_fitx)
    final = draw_on_original_image(undist, left_fitx, right_fitx, ploty, src, dst)
    print_Image(final,' Final')

    '''
    '''
    output = 'output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    # clip1 = VideoFileClip("challenge.mp4")
    clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    clip.write_videofile(output, audio=False)
    '''







