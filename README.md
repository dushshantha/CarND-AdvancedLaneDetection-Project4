
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

[image1]: ./images/image1.png "Distorted"
[image2]: ./images/image2.png "Undistorted"
[image3]: ./images/image3.png "Original"
[image4]: ./images/image4.png "Undistorted"
[image5]: ./images/image5.png "Grayscale"
[image6]: ./images/image6.png "HLS"
[image7]: ./images/image7.png "S Layer"
[image8]: ./images/image8.png "S Layer"
[image9]: ./images/image9.png "S Layer"
[image10]: ./images/image10.png "S Layer"
[image11]: ./images/image11.png "S Layer"
[image12]: ./images/image12.png "S Layer"
[image13]: ./images/image13.png "S Layer"
[image14]: ./images/image14.png "S Layer"
[image15]: ./images/image15.png "S Layer"
[image16]: ./images/image16.png "S Layer"
[image17]: ./images/image17.png "S Layer"

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

Distorted                     |  Undistorted
:----------------------------:|:-------------------------:
![alt text][image1]           |  ![alt text][image2]

 

The Calibration data is saved in wide_dist_pickle.p file and this data can be loaded to undistort the images in the pipeline without having to recalibrate in every step. 

### Pipeline (single images)

#### 1. Undistort the image

In the first step pf the pipeline, I use the Camera Calibration data the I originally saved to undistprt the image. I use the function undistortImage(img, mtx, dist) (Lines 182 - 184 in AdvancedLaneDetection.py) to apply the undistortion. 
Below is a example of this function applied to one of the test images. 

Original                     |  Undistorted
:---------------------------:|:-------------------------:
![alt text][image3]          |  ![alt text][image4]


#### 2. Covert the image to Grayscale and HLS

In this step, I converted the undistorted image from step 1 to Grayscale and HLS using the helper function grayscale ( Lines 61 - 63 in AdvancedLaneDetection.py) and convertHLS ( Lines 65 - 67 in AdvancedLaneDetection.py). I then used the helpter funtion getSLayer (Lines 69 - 70 in AdvancedLaneDetection.py) to extract the S layer from the HLS converted image. 

Below are these convertions applied to the test image above.

Grayscale                     |HLS                        |S Layer of HLS
:----------------------------:|:-------------------------:|:-------------------------:
![alt text][image5]           |![alt text][image6]        | ![alt text][image7]


#### 3. Apply Gradient thresholds

In the 3rd step I experiented with many different combinations of color and Gradient thresholds applied to the images from the last step. 

* X directional Gradient on Grayscale

In this step, I used X directional Sobel gradient on the Grascale image from the previous step. I then coverted the resulting image to a binary image with thethreshold 20 - 100. The function abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)) ( Lines 198 -213 in AdvancedLaneDetection.py) accepts the grascale image, orienation, kernel size and threshold values and return a binary image with the gradient threashold applied. In my pipeline I used the kernel size 15. 
I also applied a region mask to the resulting image to eliminate the distractions. For this, I used 2 functions. get_transform_points(img) (Lines 243 - 256 in  AdvancedLaneDetection.py) takes the image and returns source and destination points for perspective transformation. This function will be explained in detail later. I used the source points returned from this function to create 4 vertices for the mask. Then I used the function region_of_interest(img, vertices) (Lines 90 -106 in AdvancedLaneDetection.py) to return a masked image with only the pixels belonging to themasked area. I added extra padding to the original Source points when I created the vertices to include the entire lane lines without cutting it off. Below python code shows the creation of Vertices.

```python
vertices = np.array([[(int(src[0][0]) - 20 , int(src[0][1])), (int(src[1][0]) - 15 , int(src[1][1])),
                          (int(src[2][0]), int(src[2][1]) + 15 ), (int(src[3][0]) + 20, int(src[3][1]))]],
                        dtype=np.int32)
```
 Example images below shows the original Grayscale image and the gradient thresholded image with the mask.
 
 Grayscale                   |  Gradient thresholded
:---------------------------:|:-------------------------:
![alt text][image8]          |  ![alt text][image9]


* Color Threashold on S layer

In this 2nd step of thresholding, I apply a threashold of 175 - 255 on the S layer image from step 2 above. Here I use a function called apply_binary_threshold(s, thresh=(0, 255))) ( Lines 193 - 196 in AdvancedLaneDetection.py). This function will get an image and apply a threshold and return the binary image. I also apply the same mask for the resulting image in this step as well. Example images below shows the S layer and the threshoded image.

S Layer                      |  Color Thresholded
:---------------------------:|:-------------------------:
![alt text][image10]         |  ![alt text][image11]

* Combine the 2 thresholds

As the final stage of this step, I combine the color and gradient thresholds to get the final thresholded image. 

Below example shows the result of this step.

![alt text][image12]


#### 4. Perspective Transformation

The next step in the pipeline is to perform the perspective transformation on the binary image from the previos step. The goal of this transformation is to get a bird's eye view of the lanes so we can identify any curvature of the lane lines. 

To perform this, I first find 4 source points of the source image that roughly covers the lanes. I use get_transform_points(img) function ( Lines 243 - 256 in  AdvancedLaneDetection.py). This function takes the image in to get the dimentions and calculate 4 points where the lane lines can most likely be located. Below lines of code shows how I come up with th points.

```python
src = np.float32([[int(imshape[1] * 0.10) - 20, imshape[0]], 
                  [int(imshape[1] * 0.50) - 55, int(imshape[0] * 0.6)+ 20],
                  [int(imshape[1] * 0.55) + 45 , int(imshape[0] * 0.6) + 20 ],
                  [imshape[1] * .95 + 20, imshape[0]]])
    dst = np.float32([[int(imshape[1] * 0.20), imshape[0]], 
                     [int(imshape[1] * 0.20), 0],
                     [int(imshape[1] * 0.75), 0],
                     [int(imshape[1] * 0.75), imshape[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 108, 720      | 256, 720      | 
| 585, 452      | 256, 0        |
| 749, 452      | 960, 0        |
| 1236, 720     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Original                     |  Tansformed
:---------------------------:|:-------------------------:
![alt text][image13]         |  ![alt text][image14]


Then I use perspective_trans(img, src, dst) function (lines 186 - 191 in AdvancedLaneDetection.py) to perform the perspective Transformation.  Below exaple images show the before and after the transformation. 

Before                       |  After
:---------------------------:|:-------------------------:
![alt text][image15]         |  ![alt text][image16]


#### 5. Sliding Window, Curvature and deviation from the center

I used the Sliding Window method described in  the lessons to identify the lane lines. This is implemented in the function sliding_windows(binary_warped) (Lines 287 - 371 in AdvancedLaneDetection.py). This takes the warped binary image from previous step and look for the lane lines using highest points in a histogram. this generates a series of X coordinated for left and riht lines. This function also uses return_Curvature(left_fitx, right_fitx, ploty, image_size) function( Lines 24 - 284) that uses the x ccordinate points to calculate the curvature and the deviation of the vehicle from the center of the road. 

I also uses a class called Line to save the data from above function for each image so i can skip the blind search where there is a lane detected in the previous frame. In this case I use skip_sliding_windows(binary_warped) function ( Lines 375 - 428) to look at the 100 pixel area from the last line detcted. This increase the performance of the algorythm significantly. 


#### 6. Draw the lane lines on the original image

In thislast step of the pipelne, I use an copy of the original undistorted image, apply perspectie transformation, draw the lines using the X coordinate points obtained from the previous step and reverse the perspective transformation to the original. I do this using the function draw_on_original_image(orig_undist, left_fitx, right_fitx, ploty, s, d) ( Lines 431 - 447) 

Below example shows the resulting image with the highlighted lane. 

![alt text][image17]

---

### Pipeline (video)

The process_image(image) function ( Lines 450 - 509) organizes all the steps i mentioned in the previous section in to a function and used to process every frame of the video. Below is the link to the resulting Video.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
