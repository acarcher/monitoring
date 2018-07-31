from myutils import npmerge
import svm
import preprocessing as pp
import numpy as np


if __name__ == "__main__":

    #pp.extractframes("videos/video_1.hevc", "data/video_1")
    #pp.extractframes("videos/video_2.hevc", "data/video_2")
    #pp.extractframes("videos/video_3.hevc", "data/video_3")
    #pp.extractframes("videos/video_4.hevc", "data/video_4")

    #pp.isolatefaces('svmdata/yalefaces', 'svmdata/yalefaces_isolated')

    #svm.gensamples("svmdata/yalefaces_isolated", "svmdata/samples", "positive")
    #svm.gensamples("svmdata/256_ObjectCategories", "svmdata/samples", "negative")

    negative = npmerge("svmdata/samples", (5000, 14400), "negative", True)
    positive = np.load("svmdata/samples/positive_samples.npy")
    svm.trainsvm(positive, negative, 164, 5000)