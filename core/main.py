#!/usr/bin/python3

from support.preprocessing import Preprocess

VIDEO_IN = "../data/input/videos/"
FACE_DATASET_DIRTY = ""
FACE_DATASET_CLEAN = ""


def run():



    #pp.extractFrames("videos/video_1.hevc", "data/video_1")
    #pp.extractFrames("videos/video_2.hevc", "data/video_2")
    #pp.extractFrames("videos/video_3.hevc", "data/video_3")
    #pp.extractFrames("videos/video_4.hevc", "data/video_4")

    #pp.isolatefaces('svmdata/yalefaces', 'svmdata/yalefaces_isolated')

    #svm.gensamples("svmdata/yalefaces_isolated", "svmdata/samples", "positive")
    #svm.gensamples("svmdata/256_ObjectCategories", "svmdata/samples", "negative")

    #negative = npmerge("svmdata/samples", (5000, 14400), "negative", True)
    #positive = np.load("svmdata/samples/positive_samples.npy")
    #svm.trainsvm(positive, negative, 164, 5000)


    #
    # run tests
    # 
    #

    preprocessing = Preprocess()
    preprocessing.videosToFrames(VIDEO_IN)
    preprocessing.imagesToFaces(FACE_DATASET_DIRTY)

