# Detecting Rice Seeds with OpenCV (Python)
This repository contains a reduced version of some source codes of an original research project. The task was to find and crop segments in which a single rice grain is present to form a dataset. Also empty slots and broken seeds must be omitted in the final set.
The original dataset and source code is confidential until final results are published. The goal of this repository is for educational purposes.

Raw images came in two different variations:
 - Rice seeds implanted in slots with **Blue** background
 - Rice seeds scattered over a **Black** plate

Directories: 
 - [input](input/) : two image samples of original dataset
 - [output](output/) : step by step outputs of black and blue backgrounded images 
  

### Rice seeds on blue background

* Step 1: Load original image
![](images/blue01.jpg)
* Step 2: Create a binary mask
![](images/blue02.jpg)
* Step 3: Run canny edge detector _(click on the image to see in detail)_
![](images/blue03.jpg)
* Step 4: use detected edges to find contours
![](images/blue04.jpg)
* Step 5: Generate bounding box and circles from contours
![](images/blue05.jpg)
* Step 6: Sample crops of image containing rice seeds
![](images/blue06.jpg)

#### _For more detailed images open the output directory_