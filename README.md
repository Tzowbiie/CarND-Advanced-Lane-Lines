
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

[image1]: ./camera_cal/calibration3.jpg "Original Image"
[image2]: ./camera_cal/corners_found3.jpg "Image with corners"
[image3]: ./camera_cal/test_undist.jpg "Undistorted Image"

[image4]: ./test_images/test1.jpg "Raw Image"
[image5]: ./output_images/undistort_image.jpg "Undistorted Image "
[image6]: ./output_images/thresholded_image.jpg "Thresholded Image"
[image7]: ./output_images/warped_image.jpg "Warped Image"
[image8]: ./output_images/withlanes_image.jpg "Warped Image with Polynomial Fit"
[image9]: ./output_images/laneswarped_image.jpg "Rewarped Image with Polynomial Fit"
[image10]: ./output_images/final_image.jpg "Final Image"

[image11]: ./output_images/withlanes_image1.jpg "Warped Image with Polynomial Fit"
[image12]: ./output_images/laneswarped_image1.jpg "Rewarped Image with Polynomial Fit"
[image13]: ./output_images/final_image1.jpg "Final Image"

[video1]: ./video_out/project_video_out_REWORK.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./P2.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
![alt text][image1]
![alt text][image2]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][image3]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image4]
To distort the image I use the same camera parameters as I found before with the method described above.
![alt text][image5]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color threshold and a Lacplace operator to generate a binary image (thresholding steps at lines 38 through 69 in `P2.ipynb` Function imgSobel(img, s_thresh, sx_thresh, v_thresh):).  Here's an example of my output for this step:

![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transformImage(img)`, which appears in lines 111 through 118 in the file `P2.ipynb`. The function takes as inputs an image (`img`). Source (`src`) and destination (`dst`) points depend on the image size in the function:

```python
'''Transforms image of bird eye view to road'''
        x_fac=int(img.shape[0]/20)
        y_fac=int(img.shape[1]/20)
        '''Src are the source points in the original image, dst are the same points but on the road viewed from above'''
        dst = np.float32([[0,18*x_fac],[8.6*y_fac,450],[11.4*y_fac,450],[20*y_fac,18*x_fac]])
        src = np.float32([[0,20*x_fac],[0,0],[20*y_fac,0],[20*y_fac,20*x_fac]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]
![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify lane-line pixels I use a 2-step approach. 
1. If no lanes are detected yet, `fit_polynomial()` and `find_lanepixels()` funcion is used:
   Finding lane pixels with maximum of histogram. Following from there with the sliding boxes methode to detect lane pixels.
    2. Once lanes are detected, function `search_around_poly()` uses the polynomial coefficients from the last frame to search for new lane pixels.

A second degree polynomial is fitted through the lane-line pixels afterwards. The polynomial coefficients are stored in the right_lane and left_lane object.

Sliding box method:
![alt text][image8]
Search is based on polynomial coefficients of last frame:
![alt text][image11]

This Image is transformed back to the normal view:
![alt text][image9]
![alt text][image12]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 393 through 468 in my code in `measure_curvature_pixels()`. Pixel values are calculated into real world units (meter) as well. For this the x and y pixel values of the lanes are calculated into real world coordinates with 2 factors: 

    ym_per_pix = 150/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
The real world coordinates are fed to a polyfit funtion to calculate the real world parameters of a second degree polynomial.
With these parameters the Radius of both lanes is defined.

After that I perform a sanity check of both lane candidates with the function `sanity_check_lines()`. The idea is to avoid big changes of lanes between 2 frames. I want to reduce noise between the single frames:
    1. check wether polynomial coefficients of the lanes of older frames (smoothing the last 8 frames) are similar to new candidate of the actual frame.
       IF yes -> left_lane.best_fit is updated with new lane parameters
With the second approach I want to check if both lanes are somehow parallel (or have the same curvature):
    2. Check whether poly-coefficients of both lanes are similar compared to each other and
    3. Check whether right lane and left lane are around 3.7 m away from each other.
       IF yes -> both lanes are found = TRUE

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `transformImageBack()`.  Here is an example of my result on a test image:

![alt text][image10]
![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Big changes in lighting conditions still lead to failures. A better failure exception handling could improve robustness in that case. If no new lane is detected in a the acutal frame, use the information from the last valid detection.

The implementation of polynomial functions of degree 3 or even fitting of a spline could improve the line fitting and so the processing of lane position, curvature and position of the car relativly to the lanes.
