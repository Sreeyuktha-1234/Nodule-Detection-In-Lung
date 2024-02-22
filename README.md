# Nodule-Detection-In-Lung

Features
CT Scan Images of lung with nodule
Kaggle Dataset (Lung Nodules Detection Dataset Annotations)
The dataset is organized into two folders, train and val, with each Samples Image file in Formart.txt having an annotation relating to it. The data contains labels and images.
Image dimensions: 256 x 256 Annotation: the x and y coordinates of the lung nodule's location.
The label folder has the coordinates of every image that has been gathered and labeled with X and Y.
The image folder consists of 239 files in the train folder whereas the val folder consists of 41 files.
Drawbacks of the kaggle dataset
Only a few CT scan images consist of the nodule in it.

Approaches towards lung nodule detection
The process of a computer-aided diagnosis  system is explained in this part by outlining various "feature engineering"-based methods for nodule identification. Pre-processing, lung segmentation, nodule identification, and classification are the four primary processes in this conventional framework. Taking into account the relevant works which have been found so far, each stage has been precisely detailed.

Image Preprocessing
The objective of this stage is to apply multiple filters on the given lung CT image in order to reduce noise and accentuate nodule-like structures.
We have used three main filters they are
Median Filter
It is a processing method that keeps the edges of an image intact while reducing noise in image processing and other applications. This filter is nonlinear in nature, substituting the median value of the nearby pixel values for each pixel value.While salt-and-pepper noise manifests itself in an image as sporadic bright and dark pixels, the median filter is very good at eliminating it. Unlike linear filters and mean filters, the median filter is less susceptible to extreme results (outliers) since it takes into account the statistical midpoint of the local pixel values.However, in comparison to other filters made for these kinds of noise patterns, the median filter might not be as good at eliminating Gaussian or evenly distributed noise. Furthermore, the median filter may not be appropriate for cases when edge preservation is critical because it can smooth off image edges.

Dot enhancement filter
By using this filter we can enhance the dots in the image. By this method we can clearly enhance the nodule from the CT scan Image of the lung.

Log filter
The Gaussian filter is used to smooth the image before applying the Laplacian filter. The Laplacian filter draws attention to areas of the image, like edges, where there is a fast change in intensity. It works particularly well for identifying small features and edges.A mask with corner elements set to 0 and the center element set to a negative value is used by the positive laplacian operator. This filter extracts an image's outer edges.Utilizing the negative Laplacian operator, the inward boundaries of the image can be determined. It makes use of a typical mask where all other elements are set to -1, the corners to 0 and the center element to positive.

Image segmentation
Lung segmentation is the second stage of the nodule finding system. This step extracts only the lung areas in an attempt to narrow the search space. The thorax extraction and the parenchyma structure/lung extraction sub-steps are the two main components of this step.

Lung Extraction
We have written a python script opencv library code to perform lung extraction from the CT scan images that we have taken from kaggle.
Lung extraction process 
Read the image in grayscale.
Applies Gaussian blur to the original image.
Uses Otsu's method to threshold the blurred image.
Applies morphological operations (closing and opening) to the binary image.
Finds contours in the processed image and selects the largest contour.
Creates a mask based on the largest contour.
Applies the mask to the original image using bitwise AND, resulting in the extracted lung region.

Nodule Detection
Using various image processing and pattern recognition approaches, the nodule detection module seeks to distinguish actual nodule candidates from fake positives (FPs), such as blood vessels, hues, bronchial vessels,bifurcation points, and ribs. 

Nodule detection using YOLO v5 model
A neck network , detection head, and a backbone network are the typical components of YOLOv5. The boundary box dimensions with ratings of confidence for each class are output by the detecting head.The model developed can be applied to new, unknown lung scans for inference after training.
To find lung nodules, apply a trained YOLOv5 model to the input lung pictures.In order to refine and filter the bounding boxes that were detected, post-process the model's output.A computer vision model in the YOLO family is called YOLOv5. YOLOv5 is frequently utilized for object detection. YOLOv5 is available in four primary variants, with increasing accuracy rates: small , medium , large , and extra large . The training time for each version varies as well.

Nodule detection using thresholding based methods
A threshold-based method for nodule detection involves setting a pixel intensity threshold to segment potential nodules in medical images.Out of all the image segmentation techniques, this is the easiest. The image is divided using the image histogram. The intensity level between two peaks, referred to as the "threshold," is chosen to divide desired classes, with each peak on the histogram representing a distinct region. Next, all pixel intensities are grouped according to the threshold value, completing the segmentation process.
Afterwards, the 3-D skeletonization technique was used to remove tubular structures. In their investigation, scientists have shown that the average depth variation with respect to the medial axis is minimal in tubular structures, such as vessels. Conversely, nodules exhibit a sudden rise in values further from the nodule boundary. Nodules have therefore been removed from vessels by selecting the proper threshold value using the skeletonized segmentation technique.

For implementing this we have written a python code where we have imported all the required libraries from OpenCV for image processing numpy for numeric operations matplotlib for plotting.We have a nodule detection function which takes the image path and threshold value as a input parameters it reads the grayscale image and applies the binary threshold to segment nodule. Where it finds counters in the binary image and draws the counters. We have written a simple nodule detection method based on a fixed threshold.



Nodule detection using region based methods 
This method divides an image into distinct sections based on the satisfaction of certain homogeneity requirements. The homogeneity criteria are based on gray-level pixel values. Several methods, such as split and merge, region splitting, and region merging, are used in region-based segmentation. "Split and Merge" is a hybrid strategy that combines the best features of both approaches. This method groups pixels according to a set of similarity criteria. A key component of precise segmentation is the use of similarity criterion.

To implement this we have used a pre-trained mask R-CNN model from the torchvision library. We have imported pytorch deep learning library for image preprocessing matplot for visualization. We have loaded the image and passed through the pre-trained mask R-CNN model to obtain predictions including bounding box coordinates, confidence score segmentation masks. The predictions are extracted and converted to numpy arrays for further preprocessing.Detections with confidence scores and specified threshold are filtered out.
