#!/usr/bin/env python
# coding: utf-8

# In[49]:


import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
'''
get_hog
    Functionality : Calculates histogram of oriented gradients of input array(image)
    Input: Image
    Output: Feature Vector and HOG image
changelog-- 
    01/01/2019: Initial version
    05/01/2019: Included feature linearization. Returns linearized feature vector
    06/01/2019: Included visualization of HOG vector
    https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
    N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005, pp. 886-893 vol. 1.
doi: 10.1109/CVPR.2005.177
'''
def get_hog(img,csize=8,bin=9,visualize=False):
    sudo_image=np.sqrt(np.sqrt(img / float(np.max(img))))
    sudo_image*=255
    unit_angle=360/bin
    height,width=sudo_image.shape
    gm,ga=get_gradient(img)
    block_gradient= np.zeros((int(height / csize), int(width /csize),bin))# creating an empty array for storing HOG
    for i in range(block_gradient.shape[0]):
        for j in range(block_gradient.shape[1]):
            cm = gm[i * csize:(i + 1) * csize,j * csize:(j + 1) * csize]
            ca = ga[i * csize:(i + 1) * csize,j * csize:(j + 1) * csize]
            block_gradient[i][j] = get_block_gradient(cm, ca,bin,unit_angle)
    fds=[]
    for i in range(block_gradient.shape[0] - 1):
        for j in range(block_gradient.shape[1] - 1):
            block_vector = []
            block_vector.extend(block_gradient[i][j])
            block_vector.extend(block_gradient[i][j + 1])
            block_vector.extend(block_gradient[i + 1][j])
            block_vector.extend(block_gradient[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            fds.append(block_vector)
    if visualize:
        hog_output_image= display_hog(np.zeros([height, width]), block_gradient,unit_angle,csize)
        return np.array(fds).ravel(), hog_output_image
    else:
        return np.array(fds).ravel()
'''
get_gradient--
    Functionality : Calculates Gradient of the whole image array
    Input: Image
    Output: gradient magnitude and gradient angle
changelog-- 
    09/01/2019: Initial version with manual calculation
    12/01/2019: final version with use of sobel operator
    https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
'''
def get_gradient(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=5)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=5)
    gm = cv2.addWeighted(gx, 0.5, gy,0.5, 0)
    ga = cv2.phase(gx, gy, angleInDegrees=True)
    return abs(gm), ga

'''
get_block_gradient--
    Functionality : Calculates Gradient of the whole image array
    Input: cell magnitude/angle vector,bin size,angleunit (angle unit /bin)
    Output: vector with block orientation
changelog-- 
    09/01/2019: Initial version
'''
def get_block_gradient(cm,ca,bin,unit_angle):
    orientation_centers = [0] * bin
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            block_gradient = cm[i][j]
            block_angle = ca[i][j]
            min_angle, max_angle, modulus = get_bins(block_angle,unit_angle,bin)
            orientation_centers[min_angle] += (block_gradient * (1 - (modulus / unit_angle)))
            orientation_centers[max_angle] += (block_gradient * (modulus / unit_angle))
    return orientation_centers
'''
get_bins
    Functionality : Calculates distribution of angle units into bins
    Input: block angle,unit angle,bin size
    Output: bin array with angle distribution
changelog-- 
    09/01/2019: Initial version
'''
def get_bins(ga,unit_angle,bin=9):
    temp_a = int(ga / unit_angle)
    modulus = ga % unit_angle
    if temp_a == bin:
        return temp_a - 1, (temp_a) % bin, modulus
    return temp_a, (temp_a + 1) % bin, modulus
'''
display_hog
    Functionality : Returns HOG feature image for visualization
    Input: array , block_gradient, angle unit, cell size
    Output: HOG feature image 
changelog-- 
    06/01/2019: Initial version
'''
def display_hog(img,block_gradient,unit_angle,csize):
    cell_width = csize / 2
    max_mag = np.array(block_gradient).max()
    for i in range(block_gradient.shape[0]):
        for j in range(block_gradient.shape[1]):
            temp_grad = block_gradient[i][j]
            temp_grad /= max_mag
            angle = 0
            angle_gap = unit_angle
            for magnitude in temp_grad:
                angle_radian = math.radians(angle)
                x1 = int(i * csize + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(j * csize + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(i * csize - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(j * csize - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(img, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                angle += angle_gap
    return img

# a=cv2.imread("1.png",0)
# fds,hog_image=get_hog(a,csize=8,bin=9,visualize=True)
# print(fds.shape)
# plt.figure(1)
# plt.axis('off')
# plt.imshow(hog_image,cmap='gray')
# plt.title("HOG using our function")
# plt.show()

