#!/usr/bin/python3
import os
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure


def testhaar(directory):
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]

    for path in paths:

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x,y,xoffset,yoffset) in faces:
            cv2.rectangle(img,(x,y),(x+xoffset,y+yoffset),(255,0,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = img[y:y+h, x:x+w]
            #eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
            #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        winname = "testhaar"
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 40, 30)
        cv2.imshow(winname, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def testskimagehog():
    #im = cv2.imread("svmdata/yalefaces_isolated/subject01.gif.jpg")  # 8x8? cell size
    im = cv2.imread("data/video_1_59.jpg")

    ppc = int(im.shape[0] / 20)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    david, image = hog(im, orientations=9, pixels_per_cell=(ppc, ppc), cells_per_block=(2, 2), block_norm="L2", visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    hog_image_rescaled = exposure.rescale_intensity(image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()

    print(len(david))


def testcv2hog():
    im = cv2.imread("data/video_1_59.jpg")  # 8x8? cell size

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(_winSize=(256, 256), 
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=9
                            )
    features = hog.compute(im)

    print(len(features))


# testskimagehog()
# testcv2hog()
#testhaar('svmdata/yalefaces_jpg')