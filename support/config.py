#!/usr/bin/python3

CASCADE = "data/haarcascade_frontalface_default.xml"
PREPROCESS = "data/intermediate/preprocessing/"

V2F_OUT = "data/intermediate/videosToFrames/"
I2F_OUT = "data/intermediate/imagesToFaces/"

V2F_CONFIG = {
    'FRAME_OUT': PREPROCESS + "_extractFrames/"
}

I2F_CONFIG = {
    'CONVERT_OUT': PREPROCESS + "_convertToJPG/",
    'GRAY_OUT': PREPROCESS + "_grayScale/",
    'SHRINK_OUT': PREPROCESS + "_shrinkImage/",
    'ISOLATE_OUT': PREPROCESS + "_isolateFaces/",
    'CROP_OUT': PREPROCESS + "_crop/",
    'DIM_OUT': PREPROCESS + "_maxDim/",
    'RESIZE_OUT': PREPROCESS + "_resize/"
}
