#!/usr/bin/python3


class imgutil():

    # https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
    def imgPyramid(img, scale=1.5, minSize=(30, 30)): #in our case, we want min size to be 75x75
        yield img
