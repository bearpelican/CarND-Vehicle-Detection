## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_sample_imgs.png
[image2]: ./output_images/visualize_colorspace_features.png
[image3]: ./output_images/gradient_visualization_HSV
[image4]: ./output_images/sliding_window_img.png
[image5]: ./output_images/color_coded_search_boxes.png
[image6]: ./output_images/color_search_5.png
[image7]: ./output_images/color_search_2.png
[image8]: ./output_images/color_search_3.png
[image9]: ./output_images/heat_map.png
[image10]: ./output_images/label_frame_12.png
[image11]: ./output_images/label_frame_18.png
[image12]: ./output_images/label_frame_21.png
[image13]: ./output_images/label_frame_31.png
[image14]: ./output_images/label_frame_38.png
[image15]: ./output_images/label_frame_50.35.png
[video1]: ./output_video/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  
Images were loaded and saved to a pickle file in `Cell 3` `1-Data-Preparation.ipynb`. 
* # of Vehicle images: 8792  
* # of Non-vehicle images: 8968  
* Size: (64, 64, 3)
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

#### Colorspace `2-Color-Spaces.ipynb`

I  explored different color spaces in this notebook. I plotted the normalized features for each colorspace to get a better understanding of the differences.  
Mean normalization of all car images VS non-car images `Cell 20`:

![alt text][image2]

Here, it looks like HSV and LUV colorspaces had the most differentiation between the car and non-car histograms.
I also ran an SVM on the colorspaces alone and found that LUV had the best accuracy.


#### HOG `3-HOG.ipynb`
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters through manual tuning and discovered different tradeoffs.  
Here are the values I arrived at:  
|Parameter|Value|
|:--------|----:|
|Color Space|HSV|
|HOG Orient|8|
|HOG Pixels per cell|16|
|HOG Cell per block|2|
|HOG Channels|All|
|Spatial bin size| (16,16)|
|Histogram bins|32|
|Histogram range|(0,256)|
|Classifier|LinearSVC|
|Scaler|StandardScaler|

Several of the accuracies were recorded in the spreadsheet: `ParameterClassificationAccuracy.xls`  
* Channels - Using all 3 color channels consitently has ~2% better accuracy  `3-HOG.ipynb - Cell 11`
* Orientation - Sweet spot in accuracy seems to be at 8 `3-HOG.ipynb - Cell 12`
* Pixels per cell - 8 has slighty better accuracy, but 16 trained 2-3x faster `3-HOG.ipynb - Cell 13`


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the above parameters in the notebook `4-SVM-Classify.ipynb`. I was able to achieve a test accuracy of 99.3%

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I first used the sliding windows implementation as a sanity test to make sure I could locate cars - `5-Sliding-Window.ipynb - Cell 3`.
A search window of 96x96 px seemed captured the cars nicely. To limit false positives, I limited the starting y position search to 400px.

![alt text][image4]

I then combined the sliding widnow search with HOG sub-sampling to speed up image search in the notebook `5-HOG-Subsampling.ipynb`.  

I color coded the search boxes to get a better idea of what we were searching (Scale of 2):

![alt text][image5]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 5 scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  
|Box|Scale|ystart|ystop|
|:---|---|---|---:|
|0|1|370|520|
|1|1|400|550|
|2|1.5|390|550|
|3|2|380|650|
|4|2.5|410|680|
|5|3|400|700|

Here are some example images:  

![alt text][image6]

![alt text][image7]

![alt text][image8]
---

The performance of this pipeline is pretty slow. To speed it up:  
HOG-subsampling as mentioned earlier.  
16 pixels per cell sped up processing time


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap in and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  `7-False-Positives.ipynb - Cell 1`

### Here are six frames and their corresponding bounding boxes, heatmaps and labels:  

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The processing pipeline still runs pretty slow. It currently takes a second to process each frame on my laptop. More testing would need to be done to evalutate the tradeoffs between speed and accuracy. Some ideas include -  
Using 2 color channels instead of all 3 and speed up processing  
Use an ensemble of different parameters and colorspaces can maybe achieve higher accuracy  
Keeping track of objects that have appeared on the screen to help eliminate false positives  


