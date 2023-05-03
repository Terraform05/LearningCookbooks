# Load library
import pandas as pd
import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend to avoid displaying plots

from matplotlib import pyplot as plt

print('=======================|load images|=======================')

# Load image as grayscale
image = cv2.imread("MachineLearning/images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# Show image
plt.imshow(image, cmap="gray"), plt.axis("off")

# SHOW IMAGE
plt.show()

# Show data type
print(type(image))

# Show image data
print(image)

# Show dimensions
print('image resolution and mat dimensions: ', image.shape)

# Show first pixel
print('first pixel: ', image[0, 0])

# Load image in color
image_bgr = cv2.imread("MachineLearning/images/plane.jpg", cv2.IMREAD_COLOR)

# Show pixel
print(image_bgr[0, 0])

# OpenCV default is BGR, but Matplotlib and others use RGB. convert bgr to rgb
# Convert to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Show image
plt.imshow(image_rgb), plt.axis("off")
plt.show()

print('=======================|save image for preprocessing|=======================')

# Load libraries

# Load image as grayscale (read filepath)
image = cv2.imread("MachineLearning/images/plane.jpg", cv2.IMREAD_GRAYSCALE)

# Save new image (write to filepath)
print(cv2.imwrite("MachineLearning/images/plane_new.jpg", image))

print('=======================|resize images|=======================')

# Load image

# Load image as grayscale
image = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Resize image to 50 pixels by 50 pixels
image_50x50 = cv2.resize(image, (50, 50))

# View image
plt.imshow(image_50x50, cmap="gray"), plt.axis("off")
plt.show()

print('=======================|crop images|=======================')

# Load image

# Load image in grayscale
image = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Select first half of the columns and all rows
image_cropped = image[:, :128]

# Show image
plt.imshow(image_cropped, cmap="gray"), plt.axis("off")
plt.show()

print('=======================|blur images|=======================')

# Load libraries

# Load image as grayscale
image = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Blur image (averages 5x5 radius around kernel)
image_blurry = cv2.blur(image, (5, 5))

print('====================================================')
print('100x100 kernel blur')

# Show image
plt.imshow(image_blurry, cmap="gray"), plt.axis("off")
plt.show()

# Blur image
image_very_blurry = cv2.blur(image, (100, 100))

# Show image
plt.imshow(image_very_blurry, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

print('====================================================')

# Create kernel
kernel = np.ones((5, 5)) / 25.0

# Show kernel
print(kernel)

# Apply kernel
image_kernel = cv2.filter2D(image, -1, kernel)

# Show image
plt.imshow(image_kernel, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

print('=======================|sharpen image|=======================')

# Load libraries

# Load image as grayscale
image = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Create kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Sharpen image
image_sharp = cv2.filter2D(image, -1, kernel)

# Show image
plt.imshow(image_sharp, cmap="gray"), plt.axis("off")
plt.show()

print('=======================|enhance contrast gray|=======================')

# Load libraries

# Load image
image = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Enhance image
image_enhanced = cv2.equalizeHist(image)

# Show image
plt.imshow(image_enhanced, cmap="gray"), plt.axis("off")
plt.show()

print('=======================|enhance contrast color convert to YUV|=======================')

# Load image
image_bgr = cv2.imread("MachineLearning/images/plane.jpg")

# Convert to YUV
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)  # or COLOR_RGB2YUV

# Apply histogram equalization
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

# Convert to RGB
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

# Show image
plt.imshow(image_rgb), plt.axis("off")
plt.show()

print('=======================|isolating colors color convert to HSV|=======================')

# Load libraries

# Load image
image_bgr = cv2.imread('MachineLearning/images/plane_256x256.jpg')

# Convert BGR to HSV (hue saturation value)
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Define range of blue values in HSV
lower_blue = np.array([50, 100, 50])
upper_blue = np.array([130, 255, 255])

# Create mask
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

# Mask image
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)

# Show image
plt.imshow(image_rgb), plt.axis("off")
plt.show()

# APPLY MASK
# Show image
plt.imshow(mask, cmap='gray'), plt.axis("off")
plt.show()

print('=======================|binarize image|=======================')

# Load libraries

# Load image as grayscale
image_grey = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Apply adaptive thresholding
max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10
image_binarized = cv2.adaptiveThreshold(image_grey,
                                        max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        neighborhood_size,
                                        subtract_from_mean)

# Show image
plt.imshow(image_binarized, cmap="gray"), plt.axis("off")
plt.show()

# Apply cv2.ADAPTIVE_THRESH_MEAN_C
image_mean_threshold = cv2.adaptiveThreshold(image_grey,
                                             max_output_value,
                                             cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY,
                                             neighborhood_size,
                                             subtract_from_mean)

# Show image
plt.imshow(image_mean_threshold, cmap="gray"), plt.axis("off")
plt.show()

print('=======================|remove backgrounds|=======================')
# Load library

# Load image and convert to RGB
image_bgr = cv2.imread('MachineLearning/images/plane_256x256.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Rectangle values: start x, start y, width, height
rectangle = (0, 56, 256, 150)

# Create initial mask
mask = np.zeros(image_rgb.shape[:2], np.uint8)

# Create temporary arrays used by grabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Run grabCut
cv2.grabCut(image_rgb,  # Our image
            mask,  # The Mask
            rectangle,  # Our rectangle
            bgdModel,  # Temporary array for background
            fgdModel,  # Temporary array for background
            5,  # Number of iterations
            cv2.GC_INIT_WITH_RECT)  # Initiative using our rectangle

# Create mask where sure and likely backgrounds set to 0, otherwise 1
mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Multiply image with new mask to subtract background
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]

# Show image
plt.imshow(image_rgb_nobg), plt.axis("off")
plt.show()

print('====================================================')

# Show mask
plt.imshow(mask, cmap='gray'), plt.axis("off")
plt.show()

print("""black region is the area outside our rectangle that is assumed to be definitely background. The gray area is what GrabCut considered likely background, while the white area is likely foreground.""")

print('=======================|edge detection|=======================')

# Load library

# Load image as grayscale
image_gray = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Calculate median intensity
median_intensity = np.median(image_gray)

# Set thresholds to be one standard deviation above and below median intensity
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

# Apply canny edge detector
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)

# Show image
plt.imshow(image_canny, cmap="gray"), plt.axis("off")
plt.show()

print('=======================|corner detection|=======================')
# Load libraries

# Load image as grayscale
image_bgr = cv2.imread("MachineLearning/images/plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

# Set corner detector parameters
block_size = 2
aperture = 29
free_parameter = 0.04

# Detect corners
detector_responses = cv2.cornerHarris(image_gray,
                                      block_size,
                                      aperture,
                                      free_parameter)

# Large corner markers
detector_responses = cv2.dilate(detector_responses, None)

# Only keep detector responses greater than threshold, mark as white
threshold = 0.02
image_bgr[detector_responses >
          threshold *
          detector_responses.max()] = [255, 255, 255]

# Convert to grayscale
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Show image
plt.imshow(image_gray, cmap="gray"), plt.axis("off")
plt.show()

print('=======================|potential corners|=======================')

# Load images
image_bgr = cv2.imread('MachineLearning/images/plane_256x256.jpg')
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Number of corners to detect
corners_to_detect = 10
minimum_quality_score = 0.05
minimum_distance = 25

# Detect corners
corners = cv2.goodFeaturesToTrack(image_gray,
                                  corners_to_detect,
                                  minimum_quality_score,
                                  minimum_distance)
corners = np.float32(corners)

# Draw white circle at each corner
for corner in corners:
    x, y = corner[0]
    cv2.circle(image_bgr, (int(x), int(y)), 10, (255, 255, 255), -1) #cast coordinates to int

# Convert to grayscale
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Show image
plt.imshow(image_rgb, cmap='gray'), plt.axis("off")
plt.show()

print('=======================|machine learning create features|=======================')

# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Resize image to 10 pixels by 10 pixels
image_10x10 = cv2.resize(image, (10, 10))

# Convert image data to one-dimensional vector
image_10x10.flatten()

plt.imshow(image_10x10, cmap="gray"), plt.axis("off")
plt.show()

print(image_10x10.shape)

print(image_10x10.flatten().shape)

# Load image in color
image_color = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Resize image to 10 pixels by 10 pixels
image_color_10x10 = cv2.resize(image_color, (10, 10))

# Convert image data to one-dimensional vector, show dimensions
print(image_color_10x10.flatten().shape)

# Load image in grayscale
image_256x256_gray = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Convert image data to one-dimensional vector, show dimensions
print(image_256x256_gray.flatten().shape)

# Load image in color
image_256x256_color = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Convert image data to one-dimensional vector, show dimensions
print(image_256x256_color.flatten().shape)

print('=======================|encode mean color as feature|=======================')

# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as BGR
image_bgr = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Calculate the mean of each channel
channels = cv2.mean(image_bgr)

# Swap blue and red values (making it RGB, not BGR)
observation = np.array([(channels[2], channels[1], channels[0])])

# Show mean channel values
print(observation)

# Show image
plt.imshow(observation), plt.axis("off")
plt.show()

print('=======================|encode color hist as feature|=======================')

# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
image_bgr = cv2.imread("MachineLearning/images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Convert to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Create a list for feature values
features = []

# Calculate the histogram for each color channel
colors = ("r","g","b")

# For each channel: calculate histogram and add to feature value list
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], # Image
                        [i], # Index of channel
                        None, # No mask
                        [256], # Histogram size
                        [0,256]) # Range
    features.extend(histogram)

# Create a vector for an observation's feature values
observation = np.array(features).flatten()

# Show the observation's value for the first five features
print(observation[0:5])

# Show RGB channel values
print(image_rgb[0,0])

# Import pandas
import pandas as pd

# Create some data
data = pd.Series([1, 1, 2, 2, 3, 3, 3, 4, 5])

# Show the histogram
data.hist(grid=False)
plt.show()

# Calculate the histogram for each color channel
colors = ("r","g","b")

# For each channel: calculate histogram, make plot
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], # Image
                        [i], # Index of channel
                        None, # No mask
                        [256], # Histogram size
                        [0,256]) # Range
    plt.plot(histogram, color = channel)
    plt.xlim([0,256])

# Show plot
plt.show()