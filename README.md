Solution writeup to the comma.ai Driver Monitoring Challenge
======

##### Input

* 4x 60s 20hz video

##### Ouput

* Annotated face tracking video
* Head pose feature vector

## Dependencies

* numpy
* sklearn
* skimage
* cv2

## Layout

1. Core

    * Frame preprocessing
    * Facial detection
    * Facial landmark identification
    * Geometric orientation
    * Rendering
    * Main

1. Support

    * Configuration
    * SVM preprocessing
    * SVM training
    * File utilities
    * Numpy utilities
    * Image utilities

1. Data

    * Trained SVM
    * Input
        * Video files
    * Intermediate
        * Preprocessing (optional)
        * imagesToFaces
        * videosToFrames
    * Output
        * Annotated video
        * Head pose estimation feature vectors
    * Haar cascade classifier

1. Spike

    * Random excursions

1. Tests

    * TBD

## Method

video -> video preprocess + dataset preprocess -> train svm -> face detection -> retrain svm -> face detection -> find landmarks -> calculate geometry -> render

## Pipeline

read frame -> frame preprocess -> face detection -> find landmarks -> calculate geometry -> render


## SVM Preprocessing

1. HEVC video dataset -> frames
1. Yale faces dataset -> cropped

## SVM Training

1. Cropped yale faces -> positive samples
1. 256 object categories dataset -> negative samples
1. Annotate samples
1. Train linear SVM
1. Save SVM model
1. (After sliding window): Retain SVM  with hard-negative mining
1. Save new SVM model

## Face Detection

1. Sliding window over image pyramid
1. Non-maximum suppression

## Face Alignment and Head Pose

1. Facial landmark alignment
1. 2D-3D point mapping
1. Compute head orientation

## Render Tracking and Pose

1. 
1. 

## Future:

1. Pupil detection
    * CDF
    * Feature Extraction and Normalization
1. Gaze Classification and Decision Pruning





Method:

Using comma ai dataset:
Take in hevc video
Extract frames from 60s of 20hz video (~1200)

Using yale faces dataset:
Convert to jpg and grayscale
Crop the images using builtin haar cascades
uniform resize
write to disk
generate (~165) positive samples for SVM using skimage hog descriptor

Using 256_object_categories dataset:
generate (~30600) negative samples for SVM using skimage hog in batches of 1000 saving to disk

arrange data correctly + add labels
train svm with the positive and negative samples
save trained svm

sliding window
image pyramid
non-maximum suppression
hard negative mining
retrain

find face
find eyes
geometric transformation for facial plane
generate vector