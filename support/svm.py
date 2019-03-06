#!/usr/bin/python3

import cv2
from os import path, listdir, makedirs
import myutils
import imghdr
import glob
from skimage.feature import hog
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib


class SVM():

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

