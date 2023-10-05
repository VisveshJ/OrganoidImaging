#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("C://Users//visve//Downloads//video1_000022.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.medianBlur(img,5)

ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)','Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
 plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
 plt.title(titles[i])
 plt.xticks([]),plt.yticks([])
plt.show()


# In[25]:


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
img = cv.imread("C://Users//visve//Downloads//video1_000022.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "File could not be read, check with os.path.exists()"

# Apply median blur for initial noise reduction
img = cv.medianBlur(img, 5)

# Apply global thresholding
ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# Apply adaptive thresholding methods
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 1)
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15,1)

# Apply Gaussian blur to thresholded images
blurred_th2 = cv.GaussianBlur(th2, (5, 5), 0)
blurred_th3 = cv.GaussianBlur(th3, (5, 5), 0)

# Define a kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

# Apply morphological operations to thresholded images
morph_th2 = cv.morphologyEx(th2, cv.MORPH_OPEN, kernel)

# Define titles and images for display
titles = ['Original Image', 'Global Thresholding)',
          'Mean Thresholding + Gaussian Blur',
          'Gaussian Thresholding + Gaussian Blur',
          'Mean Thresholding + Morphological Operations']
images = [img, th1, blurred_th2, blurred_th3, morph_th2]

# Display images
for i in range(5):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()


# In[7]:


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("C://Users//visve//Downloads//video1_000022.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.medianBlur(img,5)

ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
 plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
 plt.title(titles[i])
 plt.xticks([]),plt.yticks([])
plt.show()


# In[10]:


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("C://Users//visve//Downloads//video1_000022.jpg")
assert img is not None, "file could not be read, check with os.path.exists()"

blur = cv.blur(img,(5,5))
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

blur = cv.bilateralFilter(img,9,75,75)


# In[6]:


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv.imread('C://Users//visve//Downloads//video1_000022.jpg')
assert img_rgb is not None, "file could not be read, check with os.path.exists()"
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread("C://Users//visve//OneDrive//Pictures//organoidpic.JPG", cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"

w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
plt.imshow(img_rgb)


# In[3]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread("C://Users//visve//Downloads//video1_000022.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
 param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
 # draw the outer circle
 cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
 # draw the center of the circle
 cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
plt.imshow(cimg)
# cv.imshow('detected circles',cimg)
# cv.waitKey(0)
# cv.destroyAllWindows()


# In[ ]:




