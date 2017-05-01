
## **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/imge1.png "Distorted"
[image2]: ./images/imge2.png "Undistorted"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README  

This README.md file will walk you through the process I followed in completing this project. The code for this project is in the form of a python script. This document will walk you throuhg the file AdvancedLaneDetection.py 

### Camera Calibration

I used OpenCV's findChessboardCorners and calibrateCamera functions for the Camera calibration in this project. I used the Chess Board images [provided](https://github.com/udacity/CarND-Advanced-Lane-Lines/tree/master/camera_cal) in the course to perform the calibration. 

I created 2 functions to achieve this task. The first function calibration_points(filename, len_points, high_points) (lines 124 - 157 in AdvancedLaneDetection.py) till read the images from the given location and return 2 arrays for 3d points in real world space (object points) and 2d points in image plane ( image points). The function accepts the image location and number of inner points in the chess board used for the images. In our case, Its a 9 x 6 chess board. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  TThen I use cv2.findChessboardCorners function to detect the image points. This function returns the distorted points on each image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients in calibrate_cam(img_name, objpoints, imgpoints, filename) functin (lines 160 - 171 in AdvancedLaneDetection.py). This function will accept a image of a distorted chess board, obj and img points returned from the previous function and a file name to save the calibration matrix for future use. This function uses the cv2.calibrateCamera() function to calibrate the camera.  

The function undistortImage(img, mtx, dist) (Lines 182 - 184 in AdvancedLaneDetection.py) uses cv2.undistort() to use the calibration matrix returned by the above mentioned function to apply the undistortion to the image. Below examples show a chess board image before and after the undistortion applied using the calibration matrix. 

![alt text][image1] ![alt text][image2]

The Calibration data is saved in wide_dist_pickle.p file and this data can be loaded to undistort the images in the pipeline without having to recalibrate in every step. 

### Pipeline (single images)

#### 1. Undistort the image

In the first step pf the pipeline, I use the Camera Calibration data the I originally saved to undistprt the image. I use the function undistortImage(img, mtx, dist) (Lines 182 - 184 in AdvancedLaneDetection.py) to apply the undistortion. 
Below is a example of this function applied to one of the test images. 

Original
![alt text][image3]

Undistorted
![alt text][image4]

#### 2. Covert the image to Grayscale and HLS

In this step, I converted the undistorted image from step 1 to Grayscale and HLS using the helper function grayscale ( Lines 61 - 63 in AdvancedLaneDetection.py) and convertHLS ( Lines 65 - 67 in AdvancedLaneDetection.py). I then used the helpter funtion getSLayer (Lines 69 - 70 in AdvancedLaneDetection.py) to extract the S layer from the HLS converted image. 

Below are these convertions applied to the test image above.

Grayscale
![alt text][image5]

HLS 
![alt text][image6]

S Layer of HLS
![alt text][image7]

#### 3. Apply Gradient thresholds

In the 3rd step I experiented with many different combinations of Gradient thresholds applied to the images from the last step. 

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
