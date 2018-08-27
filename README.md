## README
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Term 1
---

**Advanced Lane Finding Project**

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

[image1]: ./camera_cal/calibration1.jpg "Original" 
[image1_1]: ./output_images/calib1_output.jpg "Undistorted" 
[image2]: ./test_images/test1.jpg "Road Transformed Original"
[image2_1]: ./output_images/test6_output.jpg "Road Transformed Undistorted"
[image3]: ./output_images/test1_binary_output.jpg "Binary Example"
[image4]: ./output_images/straight_lines1_warped_img.jpg "Warp Example"
[image5]: ./output_images/straight_lines1_final_img.jpg "Fit Visual"
[image6]: ./output_images/project_video_output_Moment.jpg "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 
You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `util.camera_calibration.py`.  I call the function `prepareCalibrationInputs()` to read images from `camera_cal` directory. I convert into grayscale.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I saved object and image points in pickle.
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original
![alt Original][image1] 

Undistorted
![alt Undistorted][image1_1] 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I call function `get_undistorted_image(img)` located in `util.Utils.py` file. I convert the image to grayscale. I get the object and image points stored in pickle from a global function. I call `cv2.calibrateCamera()` passing the obj and img objects. I get cameraMatrix and distCoeffs which I use to call `cv2.undistort()`.
The result is :

Undistorted test Image
![alt text][image2_1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in function `get_threshold_binary_image()` located in `util.Utils.py`).  Here's an example of my output for this step.  
![alt text][image3]
In `util.Utils.py` the combination is coded in function `get_threshold_binary_image()`. I call `apply_abs_sobel_thresh()` twice, once for each orientation along with sobel_kernel=3, thresh=(20, 100). Gradientx and Gradienty are computed. Magnitude of the gradient is calculated with values sobel_kernel=9, mag_thresh=(30, 100).
Direction of the gradient is next computed with sobel_kernel=15, thresh=(0.7, 1.3). Color transform is applied with  s_thresh=(170, 255), sx_thresh=(20, 100). Next, all these thresholds are combined. 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `apply_perspective()` in `util.Utils.py`. The `apply_perspective()` function takes as inputs an image (`img`), the `src` and `dst` values are retrieved from global variables.  I chose to hard-code the source and destination points in the following manner:

```python
    src = np.float32([[200, img.shape[0]],[600, 450], [700, 450], [1150, img.shape[0]]])
    dst = np.float32([[350, img.shape[0] ], [350, 0], [950, 0], [950, img.shape[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 350, 720      | 
| 600, 450      | 350, 0        |
| 700, 450      | 950, 0        |
| 1150, 720     | 950, 720      |

I verified that my perspective transform was working as expected. The lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In `util.sliding_window.py` the function `find_lane_pixels()` along with `fit_polynomial()` identifies lane-line pixels and fit their position with a 2nd order polynomial,
kinda like this:

![alt text][image5]

Additionally, search_around_poly() function in `util.prev_poly.py`  is implemented as a look ahead filter.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature in function` measure_curvature_real()` of `radius_curve.py` Offset is calculated in function `calculate_offset()` located in `computer_vision.computer_vision_lane_lines.py` 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `plot_back_to_orig()` function  in `util.Utils.py` .  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
There are 3 major areas of improvement. 

Sanity Check
    Slope checks need to be tightened. Radius of curvature checks need to be added. Min and Max distances can be based on curvature and slopes.  

Iterations to average over
    Sweet spot between predominantly straight and overly curvy roads need to be identified.
    
Playing with thresholds. 
    Color and gradient thresholds need to be tweaked. 
