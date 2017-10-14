## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image10]: ./output_images/Histogram.png "Histogram"
[image11]: ./output_images/ChessBoardCorners.png "ChessBoardCorners"
[image12]: ./output_images/Combined_Color_Gradient_Threshold.png "Combined_Color_Gradient_Threshold"
[image13]: ./output_images/Combined_Gradient.png "Combined_Gradient"
[image14]: ./output_images/Directional_Gradient.png "Directional_Gradient"
[image15]: ./output_images/H-L-S_Channels.png "H-L-S Channels"
[image16]: ./output_images/InverseTransformation_AllTestImages.png "InverseTransformation_AllTestImages"
[image17]: ./output_images/L-a-b_Channels.png "L-a-b Channels"
[image18]: ./output_images/Lab-B_Threshold.png "Lab-B Threshold"
[image19]: ./output_images/Lanefitment_lanepixels_windows_polyfill.png "Lanefitment_lanepixels_windows_polyfill"
[image20]: ./output_images/LaneFitment_LanePixels_Windows.png "LaneFitment_LanePixels_Windows"
[image21]: ./output_images/Magnitude_Gradient.png "Magnitude_Gradient"
[image22]: ./output_images/Perspective_Transformation.png "Perspective_Transformation"
[image23]: ./output_images/R-Channel_Threshold.png "R-Channel Threshold"
[image24]: ./output_images/R-G-B_Channels_of_GrayScale.png "R-G-B Channels of GrayScale"
[image25]: ./output_images/S-Channel_Threshold.png "S-Channel Threshold"
[image26]: ./output_images/Undistorted_ChessBoard.png "Undistorted_ChessBoard"
[image26]: ./output_images/Undistorted_Highway.png "Undistorted_Highway"
[image27]: ./output_images/X_Gradient_Threshold.png "X_Gradient_Threshold"
[image28]: ./output_images/Y_Gradient_Threshold.png "Y_Gradient_Threshold"

[video1]: ./project_video_output_text.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first two code cells of the IPython notebook located in "./project.ipynb" (IN-5 and IN-2).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

The OpenCV functions findChessboardCorners and calibrateCamera are the backbone of the image calibration. A number of images of a chessboard, taken from different angles with the same camera, comprise the input. Arrays of object points, corresponding to the location (essentially indices) of internal corners of a chessboard, and image points, the pixel locations of the internal chessboard corners determined by findChessboardCorners, are fed to calibrateCamera which returns camera calibration and distortion coefficients. These can then be used by the OpenCV undistort function to undo the effects of distortion on any image produced by the same camera. Generally, these coefficients will not change for a given camera (and lens). The below image depicts the corners drawn onto twenty chessboard images using the OpenCV function drawChessboardCorners:

![alt text][image11]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Function `cv2.calibrateCamera()` applied on chess board images returned  distorting coefficients required to undistort a camera image. 
I have passed these coefficents to `cv2.undistort()` function to demonstrate undistorting for a test image.
![alt text][image26]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. For my pipe line i settled with the following combination though i experiemented with various other combinations. 

    a. Gradient - Sobel(x) with a threshold of (50,255)
    b. Gradient - Sobel(y) with a threshold of (50,255)
    c. Magnitude Gradient - with a threshold of (50,100)
    d. Directional Gradient - with a threshold of (0.5, 1.3)

Then I combined the Gradient Thresholds above using the following method.

    `combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1`

Below is the output of this combination of Gradient Threshold

![alt text][image13]

After that I explored various color spaces like HLS and Lab and experimented with them. After multiple experiements i settled down with 'Saturation Channel' of HLS color space. The S-Channel with a threshold of (200,255) ensured the lane lines are detected properly. Here is one example where lane lines detected in 'S' channel stand out compared to 'L' and 'H' channels of HLS color space

![alt text][image15]

Finally i combined the binary images obtained from  Gradient Thresholding and color thresholding in the following way
    `color_combined_binary[(s_binary == 1) | (combined == 1)] = 1`

Here is how the combined output of the Gradient and Color thresholded image looked like

![alt text][image12]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `img_transform()`, which appears in section IN 22 of file `project.ipynb`.  The `img_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32([296,686],[1096,686],[560,482],[761,482])
dst = np.float32([[296,686],[1096,686],[296,0],[1096,0]])
```


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The below image demonstrates the perspective transform.

![alt text][image22]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Functions `search_window()` (found in section IN 27 of ./project.ipynb)  and `search_prior()` (found in secion IN 28 of ./project.pynb)  does the job of identifying lane line pixels. 

The former takes in a perspective transformed binary image as input. It then sums up the pixels along veritical direction of the image to possibly identify the right and left lanes of the image. The peaks shown on the below histogram represent summation of the pixels along y-axis and will tell where to search for lane pixels. The function defines  windows around the left and right peak positions obtained from the histogram at the bottom of the image. It then searches for a minimum of 50 pixels in those windows. If found, it adjusts the window position around the mean of the identified pixels, records the position of the pixels and then refits the windows in the next section of the image from the bottom. The search for pixels continues again in this new section of the image with in those windows. This process continues until the entire image is scanned from bottom to top.  At the end , the function identifies all left lane and right lane pixels and returns them back to calling process along with the coefficients required to fit a line through them

The below visual demonstrates the plotted histogram

![alt text][image10]


Once the left and right lane pixels are identified, lines are plotted through them using a second degree polynomial fit.
Below visual demonstrates the fitted lines and the identified pixels.

![alt text][image20]


Now that i have done a blind search on the very first frame/image, i will then pass the average 'Y' position of just identified lane lines to function `search_prior()` to process the next frame/image. This function instead of performing a blind search it would rather look for pixels around the average 'Y' position passed to it. This process continues for the rest of the frames/images by looking for pixels around the average 'Y' position from the prior frames until it breaks a certain threshold(in my case , its 50 pixels). Once the threshold is broken, the pipeline forces to perform a blind search again to ensure it does not lose track of lane lines and build confidence again.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Code for computing the radius of curvature can be found in the last section of ./project.ipynb. 

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
 
 For converting the pixels to real world measurements, i have assumed the width of lane line to be 3.7 meters (700 pixels on the image transformed binary image) and lane height to be 30 meters (720 pixels). As a first step , i fitted polynomial along the left and right lane pixels by multiplying the pixel values with appropriate conversion factor. Then I obtained second order derviative to obtain left and right lane curvature. At the end , i have taken the average of left and right lane curvature to arrive at the curvature of the lane.
 
 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Once i plotted the lane lines successfully, i have warped the plot back to the original undistored image/frame (2D view ) and filled the area between the identified lane lines with a color to visualize it. The below code in the last section of ./project.ipynb achieves this.

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = np.linalg.inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (ccwarped.shape[1], ccwarped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(udst, 1, newwarp, 0.3, 0)  
    
Here is an example of my result on a test images:

![alt text][image16]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output_text.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

On my first attempt of running the pipeline against the project video, it was clearly evident that the pipeline did not do well on frames with poor lighting and shadows. To correct this, i explored and tried various color spaces along with different combinations of gradients.
After many tries,  I felf that  'S-Channel' of HLS color space did a better job accurately identifying the lane line pixels. Hence fort , I used Saturation channel of HLS color space for my pipeline along with some combination of gradient.

Though the issue with low lighting and shadows is resolved, i am now faced with a new challeng - Radius of curvature was way too high.
It was reading about 10 to 15 kms all along and i felt some thing must be off. The only reason that i can think of is - My identified lane lines must be linear on every frame. Upon analysis, i found out that perspective transformation may be the culprit. I was being very conservative when transforming the image to birds eye view. My perspective transformed image is zoomed in and lane lines are so much linear. I have then adjusted the SRC and DST points that i used to transform the image. This change immediately improved the radius of curvature. The average radius through out the vide was between 1 to 2 kms.

Finally, i tuned the pipeline to imrpove the overall performance and reduce the time taken to process the video input.

Though my pipeline worked well on the project video, it did not work as expected on challenge videos. 
The problem lies with accurately identifying the lane lines. Obviosuly the pipeline i designed will not work if there any lane like features in the image (which are not actually lane lines). The pipeline can be more robust my having a better gradient and color thresholding process that can accurately identify lane lines on any kind of image/frame.
