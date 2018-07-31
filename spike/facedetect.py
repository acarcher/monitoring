#!/usr/bin/python3

import cv2
import numpy as np


# https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python


def HOGColorDescriptor(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):

    if (img.shape[0] != img.shape[1]) or img.shape[0] != 256:
        print("Error: Must be 256x256")
        exit(-1)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)       # Horizontal gradients
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)       # Vertical gradients
    magnitude, angle = cv2.cartToPolar(gx, gy)  # Magnitude (max across 3 channels) and direction (of max) of gradients
    num_bins = orientations                     # Number of bins .. written as 16 by default
    bin = np.int32(num_bins * angle / (2 * np.pi))    # approximate proper bin for each gradient (array)

    bin_cells = []
    mag_cells = []

    cellx = pixels_per_cell[1]
    celly = pixels_per_cell[0]

    num_cells_x = int(np.ceil(img.shape[1] / cellx))
    num_cells_y = int(np.ceil(img.shape[0] / celly))

    for i in range(0, num_cells_y):             # 1024 cells total for 256x256, size 8x8
        for j in range(0, num_cells_x):
            bin_cells.append(bin[i * celly: i * celly + celly, j * cellx: j * cellx + cellx])
            mag_cells.append(magnitude[i * celly: i * celly + celly, j * cellx: j * cellx + cellx])

    # zip: create tuple of same index values; (8x8,8x8) tuple 1 cell each
    # ravel: flatten array (64x1, 64x1)
    # bincount: (input array of [0->orientations for each pixel], 
    # array of weights [magnitude], 
    # min number bins == max number of bins [num_bins = floor(data)+1])
    # for each cell, bincount allocates the magnitude of each pixel's gradient into its directional bin
    # result is list of 9x1 arrays length 1024
    hists = [np.bincount(b.ravel(), m.ravel(), num_bins) for b, m in zip(bin_cells, mag_cells)]

    # hstack concatenates columnwise (2nd dim)
    # hist = 9216 (1024x9) len list
    hist = np.hstack(hists)

    # transform to Hellinger kernel (probabilistic analog of the Euclidean distance)
    eps = 1e-7
    hist /= hist.sum() + eps  # transforms frequency to discrete probability function
    hist = np.sqrt(hist)
    hist /= np.linalg.norm(hist) + eps

    return hist

# im = cv2.imread("data/video_1_59.jpg") #8x8? cell size
# HOGDescriptor(im)