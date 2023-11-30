import numpy as np
from PIL import Image # For showing images in the notebook
import cv2 # read/write images, treshold, highlight

le = cv2.imread('LE.tif', 0).astype(np.uint8)
he = cv2.imread('HE.tif', 0).astype(np.uint8)

# From .csv
# le = np.genfromtxt('LE.csv', delimiter=',', skip_header=True)
# he = np.genfromtxt('HE.csv', delimiter=',', skip_header=True)
# le = cv2.normalize(le, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Must normalize because csv contains values [0, 922]
# he = cv2.normalize(he, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Must normalize because csv contains values [0, 592]

c = 17

# Crop
# Best results with 13 classes when cropped

# Uncomment section 1 or 2 to crop

######### 1 ################
# le = le[:,150:900]
# he = he[:,150:900]
# c = 13
######### 1 ################


######### 2 ################
# def crop(imgArr):
#     "Crop black space on the left and right side of the image"
#     i=0
#     while imgArr[0][i] == 0:
#         i+=1
#     imgArr = imgArr[:,i:]
#     i = imgArr.shape[1]-1
#     while imgArr[0][i] == 0:
#         i-=1
#     return imgArr[:,:i]

# le = crop(le)
# he = crop(he)
# c = 13
######### 2 ################

# Initialise gray levels [0,255]
G = np.arange(256)
# Number of gray levels
n = 256
DIS = (np.tile(G, (256, 1)).transpose() - G)**2 # Euclidian distances between gray levels dis(x,y) := (G[x] - G[y])^2

# Calc histogram of he and le gray values
H = np.histogram(le.flatten(), 256)[0] + np.histogram(he.flatten(), 256)[0]

# Initializing the neural network

V = np.zeros((G.size,c))
# Random network start only shuffles the final gray-level values
# V[:,0] = 1
# for row in V:
#     np.random.shuffle(row)

# Network functions

def net(x, i):
    """
    Net input to the (x,i)th neuron
    
    for y in range(n):
    
        nu += d(x,y)*H[y]*V[y][i]

        de += H[y]*V[y][i]

    return -nu.sum()/de.sum()
    """
    nu = H*V[:,i] # numerator
    de = nu.sum() # denominator
    nu *= DIS[x,:] # Same as nu *= (G[x]-G[:])**2 # G[X] is turned into an array [G[x],G[x], ... ,G[x]] with length G
    return -nu.sum()/de

def wta(x, active):
    """
    'Winner take all'. Biggest value in row x is set to 1, all other 0
    
    Returns True if the active neuron changed
    """

    m = V[x].argmax()

    V[x][:] = 0
    V[x][m] = 1

    if(active != m):
        return True
    return False

# Running the network

def run_once():
    "Update all neurons row by row. Returns True if any rows changed"
    changed = False
    for x in range(n):
        active = V[x].argmax()
        for i in range(c):
            V[x][i] = net(x, i)
        changed |= wta(x, active)
    
    return changed
def run():
    "Run till limit or until there are no changing neurons"
    # Show progress bar with tqdm
    limit = 150
    i = 0
    changed = True
    while changed and i<limit:
        changed = run_once() # stop early if there are no changes in network
        # run_once(bar) # ignore early stop
        i+=1
    # print("Iterations done:", i)

run()

# Show segmented image

classValue = np.arange(0,2**8-1,(2**8-1)/c).astype(np.uint8)

# index - gray-level input
# value - gray-level representing the class (lighter to darker)
classified = (V*classValue).max(axis=1).astype(np.uint8)

# for each x in img returns classified[x]
imgArr = np.take(classified, le.flatten()).reshape(le.shape)

# Threshold

# method = cv2.THRESH_BINARY_INV # Black background
method = cv2.THRESH_BINARY # White background

_, segmented = cv2.threshold(imgArr, classValue[1], 255, method)

# if the image is not cropped fills white/black margins
# Same method as crop
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