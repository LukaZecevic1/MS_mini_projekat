import numpy as np
# from PIL import Image # For showing images in the notebook
import cv2 # read/write images, treshold, highlight

# From .tif
le = cv2.imread('LE.tif', 0).astype(np.uint8)
he = cv2.imread('HE.tif', 0).astype(np.uint8)
# From .csv
# le = np.genfromtxt('LE.csv', delimiter=',', skip_header=True)
# he = np.genfromtxt('HE.csv', delimiter=',', skip_header=True)
# le = cv2.normalize(le, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Must normalize because csv contains values [0, 922]
# he = cv2.normalize(he, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Must normalize because csv contains values [0, 592]

# method = cv2.THRESH_BINARY_INV # Black background
method = cv2.THRESH_BINARY # White background

# _, segmented = cv2.threshold(he, 50, 255, cv2.THRESH_BINARY_INV)
_, segmented = cv2.threshold(le, 52, 255, method)

i=0
while segmented[0][i] == 255*method:
    i+=1
segmented[:,:i] = 255 * (not method)
i = segmented.shape[1]-1
while segmented[0][i] == 255*method:
    i-=1
segmented[:,i:] = 255 * (not method)

cv2.imwrite("output.jpg", segmented)
# cv2.imwrite("output.tiff", (segmented*2**8).astype(np.uint16))
# Image.fromarray(segmented)