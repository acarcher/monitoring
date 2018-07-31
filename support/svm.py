#!/usr/bin/python3

import cv2
import os
import myutils
import imghdr
import glob
from skimage.feature import hog
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from myutils import npmerge


def extractframes(inDir, outDir):
    """Takes in a video directory and destination directory,
    outputs the frames of the video to the destination in jpg format"""

    os.makedirs(outDir, exist_ok=True)

    paths = [os.path.join(inDir, filename) for filename in os.listdir(inDir)]

    print("Extracting frames...")

    for path in paths:
        _, tail = os.path.split(path)
        fileName = os.path.splitext(tail)[0]

        currentFrame = 0
        cap = cv2.VideoCapture(path)
        success = True

        while success:

            success, frame = cap.read()
            if not success:
                break

            name = outDir + "/" + fileName + "_" + str(currentFrame) + ".jpg"

            cv2.imwrite(name, frame)

            currentFrame += 1

        cap.release()
        cv2.destroyAllWindows()
    print("Done!")


def isolatefaces(inDir, outDir):
    """Uses a Haar Cascade to isolate faces, saves into new image, will convert to jpg if necessary"""
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    print("Isolating faces...")

    if imghdr.what(os.path.join(inDir, os.listdir(inDir)[0])) is not ".jpg":
        myutils.cvt2JPG(inDir, inDir + "_jpg")
        inDir = inDir + "_jpg"

    os.makedirs(outDir, exist_ok=True)

    #paths = [os.path.join(inDir, filename) for filename in os.listdir(inDir)]
    paths = glob.glob(inDir + "/**/*.jpg", recursive=True)

    images = []
    filenames = []
    maxsize = 0

    for path in paths:

        _, filename = os.path.split(path)
        filetype = imghdr.what(path)

        if filetype is "gif":
            success, img = cv2.VideoCapture(path).read()
            if not success:
                print("Error: VideoCapture for gif image failed")
                exit(-1)
        else:
            img = cv2.imread(path)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(img, 1.2, 5)

        for (x, y, xoffset, yoffset) in faces:
            images.append(img[y:y + yoffset, x:x + xoffset])
            filenames.append(filename)

    for image in images:
        if image.shape[0] > maxsize:        # assumes square
            maxsize = image.shape[0]

    for image, filename in zip(images, filenames):  # reshapes to max similar size
        image = cv2.resize(image, (maxsize, maxsize), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(outDir + "/" + filename, image)

    print("Done isolating faces")

def gensamples(inDir, outDir, exampletype, startpoint=0):
    """Generates HOG descriptor examples from the files in inDir and saves the numpy output to outDir"""

    """	HOG feature length, N, is based on the image size and the function parameter values.
        N = prod([BlocksPerImage, BlockSize, NumBins])
        BlocksPerImage = floor((size(I)./CellSize – BlockSize)./(BlockSize – BlockOverlap) + 1)"""

    if exampletype != "positive" and exampletype != "negative":
        print("Error: exampletype must be 'positive' or 'negative'")
        exit(-1)

    os.makedirs(outDir, exist_ok=True)

    #paths = [os.path.join(inDir, filename) for filename in os.listdir(inDir)]
    paths = glob.glob(inDir + "/**/*.jpg", recursive=True)

    samples = np.empty([1000, 14400])  # need to fix when sub 1k samples

    print("Generating {} samples...".format(exampletype))
    index = 0
    round = int(startpoint / 1000)
    paths = paths[startpoint:]
    print("paths len: {}".format(len(paths)))
    print("round: {}".format(round))

    for path in paths:

        print("index: {}".format(index))
        print("path: {}".format(path))

        im = cv2.imread(path)
        # very large images break things
        if im.shape[0] >= 3000 or im.shape[1] >= 3000:
            continue

        cheightrem = im.shape[0] % 21
        #cheightpad = math.ceil(cheightrem / (cheightrem + 1)) * 21 - cheightrem
        cheightpad = 21 - cheightrem if cheightrem > 0 else 0

        cwidthrem = im.shape[1] % 21
        #cwidthpad = math.ceil(cwidthrem / (cwidthrem + 1)) * 21 - cwidthrem
        cwidthpad = 21 - cwidthrem if cwidthrem > 0 else 0

        im = cv2.copyMakeBorder(im, 0, cheightpad, 0, cwidthpad, cv2.BORDER_REPLICATE)

        cheight = (im.shape[0] + cheightpad) // 21
        cwidth = (im.shape[1] + cwidthpad) // 21

        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        descriptor = hog(im, orientations=9, pixels_per_cell=(cwidth, cheight), cells_per_block=(2, 2), block_norm="L2")

        samples[index, :] = np.array([descriptor])

        # if we are on 1000th iteration OR we're at the length of paths ... LOGIC IS FLAWED WHEN SKIPPING ... usde round to sub .. 
        # wow that's a terrible statement
        if (index + 1) % 1000 == 0 or (index + 1) % (len(paths) - (round - int(startpoint / 1000)) * 1000) == 0:
            round += 1

            if (index + 1) % (len(paths) - (round - 1 - int(startpoint / 1000)) * 1000) == 0:
                samples = np.resize(samples, (index + 1, 14400))
                np.save(outDir + '/' + "{}_samples_".format(exampletype) + str(round), samples)
            else:
                np.save(outDir + '/' + "{}_samples_".format(exampletype) + str(round), samples)

            print("Writing out {}/{}_samples_".format(outDir, exampletype) + str(round))
            index = 0
        else:
            index += 1
            print("post index: {}".format(index))

    print("Done generating {} samples".format(exampletype))


def trainsvm(positive, negative, n_positive, n_negative):
    # positive is 
    # n_positive/negative is number of samples

    # number per entry in directory
    # create a single feature vector from positive and negative samples
    # generate a vector of class labels
    # train model
    # save model

    print("Training SVM...")

    selected_positive = positive[list(range(0, n_positive)), :]
    selected_negative = negative[list(range(0, n_negative)), :]

    X = np.vstack((selected_positive, selected_negative))

    y = ["face"] * n_positive + ["no face"] * n_negative

    clf = SVC()
    clf.fit(X, y)

    joblib.dump(clf, 'classifier.pkl')

    print("Done training SVM")