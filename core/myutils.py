#!/usr/bin/python3

import os
import imghdr
import cv2
import glob
import numpy as np


def cvt2JPG(inDir, outDir):
    # Convert input directory images to jpg in the output directory

    cvtformats = [".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp",
                  ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif"]

    paths = [os.path.join(inDir, filename) for filename in os.listdir(inDir)]

    os.makedirs(outDir, exist_ok=True)

    print("Converting to .jpg...")

    for path in paths:
        filetype = imghdr.what(path)
        filename, fileext = os.path.splitext(os.path.split(path)[1])

        replaceext = False

        if fileext in cvtformats or fileext == ".gif":
            replaceext = True

        if filetype == "gif":
            success, img = cv2.VideoCapture(path).read()

            if not success:
                print("Error: VideoCapture for {} image failed".format(filetype))
                exit(-1)
        else:
            img = cv2.imread(path)

        if replaceext:
            cv2.imwrite(outDir + '/' + filename + ".jpg", img)
        else:
            cv2.imwrite(outDir + '/' + filename + fileext + ".jpg", img)

    print("Done converting to .jpg")


def npmerge(directory, dimensions, sampletype, variety=False):
    # positive = list of files with positive samples in them
    # negative = ""

    print("Merging together features...")
    paths = glob.glob(directory + "/{}*.npy".format(sampletype))

    merged = np.empty(dimensions)
    max_rows = merged.shape[0]
    index = 0

    if variety:
        variety_rows, extra_rows = divmod(merged.shape[0], len(paths))

    for path in paths: # Need to handle the case where the feature vector is smaller than the specified number of samples

        partial_sample = np.load(path)

        if variety and variety_rows <= partial_sample.shape[0]:
            num_rows = variety_rows
        else:
            num_rows = partial_sample.shape[0]

        if index + num_rows > max_rows:
            num_rows = max_rows - index - 1

        merged[list(range(index, index + num_rows)), :] = partial_sample[list(range(0, num_rows)), :]
        index = index + num_rows if index != 0 else index + num_rows - 1

        #print(index)
        #print(dimensions[0])

        if variety and paths.index(path) + 1 == len(paths) and \
           num_rows + extra_rows < partial_sample.shape[0]: # last item, time to correct juicy rounding errors
            
            #print("hi")
            merged[list(range(index, index + extra_rows)), :] = \
                partial_sample[list(range(num_rows, num_rows + extra_rows)), :]
            #print(index)
            #print(merged.shape)

        if (index + 1) % dimensions[0] == 0:
            #np.save(directory + '/' + "{}_merged".format(sampletype), merged)
            #print("hello")
            print("Done merging features")
            return merged

    return merged
