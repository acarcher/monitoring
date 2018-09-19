#!/usr/bin/python3

import cv2
from os import path, listdir
import imghdr
import numpy as np
import inspect
from .fileutil import fileutil
from .config import CASCADE, V2F_CONFIG, I2F_CONFIG


class Preprocess():

    def videosToFrames(self, directory, delete=True):

        print("videosToFrames")

        method = inspect.stack()[0].function

        util = fileutil()
        #util.createIntermediateDir(method)
        util.createTempIntermediateDirs(method, directory)  # optional argument?

        paths = [path.join(directory, filename) for filename in listdir(directory)]

        for filepath in paths:
            self._extractFrames(filepath)

        util.copyToIntermediateDir(method)

        if delete:
            util.deleteTempIntermediateDirs(method)

    def _extractFrames(self, filepath):

        """Takes in a video directory and destination directory,
        outputs the frames of the video to the destination in jpg format"""

        util = fileutil()
        filename = util.getFileName(filepath)

        print("Extracting frames for %s" % filename)

        #for entry in paths:
        #    _, tail = path.split(entry)
        #    fileName = path.splitext(tail)[0]

        currentFrame = 0
        cap = cv2.VideoCapture(filepath)
        vidCapSuccess = True

        while vidCapSuccess:

            vidCapSuccess, frame = cap.read()

            if not vidCapSuccess:
                break

            filepath = V2F_CONFIG["FRAME_OUT"] + filename + "/" + filename + "_" + str(currentFrame) + ".jpg"

            writeSuccess = cv2.imwrite(filepath, frame)     # test this

            currentFrame += 1

        cap.release()
        cv2.destroyAllWindows()

        print("Done extracting frames for %s" % filename)

    def imagesToFaces(self, directory, delete=True):

        print("imagesToFaces")

        method = inspect.stack()[0].function

        util = fileutil()
        util.createIntermediateDir(method)
        util.createTempIntermediateDirs(method)

        # paths = [path.join(directory, filename) for filename in listdir(directory)]

        for file in listdir(directory):
            self._convertToJPG(directory, file)

        for file in listdir(I2F_CONFIG["CONVERT_OUT"]):
            self._grayScale(I2F_CONFIG["CONVERT_OUT"], file)

        for file in listdir(I2F_CONFIG["GRAY_OUT"]):
            self._shrinkImage(I2F_CONFIG["GRAY_OUT"], file)

        for file in listdir(I2F_CONFIG["SHRINK_OUT"]):
            self._isolateFaces(I2F_CONFIG["SHRINK_OUT"], file)

        for file in listdir(I2F_CONFIG["ISOLATE_OUT"]):
            self._crop(I2F_CONFIG["ISOLATE_OUT"], file)

        #for file in listdir(I2F_CONFIG["CROP_OUT"]):
        self._maxDim(I2F_CONFIG["CROP_OUT"])

        #for file in listdir(I2F_CONFIG["DIM_OUT"]):
        self._resize(I2F_CONFIG["CROP_OUT"])

        # for filepath in paths:
        # for file in listdir(I2F_CONFIG["CONVERT_OUT"]):
            # maybe should make a file
            # self._grayScale(directory, file)  # in "data/support/raw/yalefaces/" (img), out "data/intermediate/preprocessing/_grayScale" (img)
            # self._shrinkImage(I2F_CONFIG["GRAY_OUT"], file)  # in ".../_grayScale" (img), out ".../_shrinkImage" (img)
            # self._convertToJPG(directory, file)  # in ".../_shrinkImage" (img), out ".../_convertToJPG" (img)
            # self._grayScale(I2F_CONFIG["CONVERT_OUT"], file)
            # self._shrinkImage(I2F_CONFIG["GRAY_OUT"], file)
            # self._isolateFaces(I2F_CONFIG["SHRINK_OUT"], file)  # in ".../_convertToJPG" (img), out ".../_isolateFaces" (file)
            # self._crop(I2F_CONFIG["ISOLATE_OUT"], file)  # in ".../_isolateFaces" (file), out ".../_crop" (img)

        # self._maxDim(I2F_CONFIG["CROP_OUT"])  # in ".../_crop" (img), out ".../_maxDim" (file)
        # self._resize(I2F_CONFIG["DIM_OUT"])  # in ".../_maxDim" (file), out "data/support/input/yalefaces/" (img)

        util.copyToIntermediateDir(method)

        if delete:
            util.deleteTempIntermediateDirs(method)

    # something that calls over directory
    # scaledown
    # convert
    # isolate
    # crop
    # resize

    # need new

    #independent function that generates the max dimensions..
        #10 images ... open file x1, open image x10; x2 ... (N+1)*2
        #open file + open image x10  (2N) ??

    #creates all in one.. deletes all in one ... based on the caller

    #return filename

    def _grayScale(self, dirpath, file):
        print("grayscale")
        #print(dirpath + file)
        #print(path.isfile(dirpath+file))

        img = cv2.imread(dirpath + file)
        #print(type(img))

        if img.shape == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(I2F_CONFIG["GRAY_OUT"] + file, img)

    def _shrinkImage(self, dirpath, file):
        print("shrink")

        img = cv2.imread(dirpath + file)
        # row col channel
        y = img.shape[0]
        x = img.shape[1]

        while(y >= 3000 or x >= 3000):
            x = int(x / 2)
            y = int(y / 2)
            img = cv2.resize(img, (x, y), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(I2F_CONFIG["SHRINK_OUT"] + file, img)

    def _convertToJPG(self, dirpath, file):                                     #
        # Convert input directory images to jpg in the output directory

        # formats = [".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp",
        #            ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif"]

        # paths = [os.path.join(inDir, filename) for filename in os.listdir(inDir)]

        print("convertToJPG")

        filetype = imghdr.what(dirpath + file)

        img = cv2.imread(dirpath + file)

        if img is None:
            success, img = cv2.VideoCapture(dirpath + file).read()

            if not success:
                print("%s: image conversion failed" % dirpath + file)

        _, extension = path.splitext(path.split(dirpath + file)[1])

        if filetype == extension[1:] or (filetype == "jpeg" and extension[1:] == "jpg"):
            k = (I2F_CONFIG["CONVERT_OUT"] + file).rfind(".")
            filepath = (I2F_CONFIG["CONVERT_OUT"] + file)[:k] + ".jpg"    # these () might be broken
            cv2.imwrite(filepath, img)
        else:
            filepath = I2F_CONFIG["CONVERT_OUT"] + file + ".jpg"
            cv2.imwrite(filepath, img)

        # for path in paths:
        #     filetype = imghdr.what(path)
        #     filename, fileext = os.path.splitext(os.path.split(path)[1])

        #     replaceext = False

        #     if fileext in formats or fileext == ".gif":
        #         replaceext = True

        #     if filetype == "gif":
        #         success, img = cv2.VideoCapture(path).read()

        #         if not success:
        #             print("Error: VideoCapture for {} image failed".format(filetype))
        #             exit(-1)
        #     else:
        #         img = cv2.imread(path)

        #     if replaceext:
        #         cv2.imwrite(outDir + '/' + filename + ".jpg", img)
        #     else:
        #         cv2.imwrite(outDir + '/' + filename + fileext + ".jpg", img)

        # print("Done converting to .jpg")

    def _isolateFaces(self, dirpath, file):
        """Uses a Haar Cascade to isolate faces, saves into new image, will convert to jpg if necessary"""

        print("isolate")

        face_cascade = cv2.CascadeClassifier(CASCADE)

        # print("Isolating faces...")

        # if imghdr.what(os.path.join(inDir, os.listdir(inDir)[0])) is not ".jpg":
        #     self.cvt2JPG(inDir, inDir + "_jpg")
        #     inDir = inDir + "_jpg"

        # os.makedirs(outDir, exist_ok=True)

        # paths = [os.path.join(inDir, filename) for filename in os.listdir(inDir)]
        # paths = glob.glob(inDir + "/**/*.jpg", recursive=True)

        # images = []
        # filenames = []
        # maxsize = 0

        # for path in paths:

        #     _, filename = os.path.split(path)
        #     filetype = imghdr.what(path)

        #     if filetype is "gif":
        #         success, img = cv2.VideoCapture(path).read()
        #         if not success:
        #             print("Error: VideoCapture for gif image failed")
        #             exit(-1)
        #     else:
        #         img = cv2.imread(path)

        #     if len(img.shape) == 3:
        #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.imread(dirpath + file)

        #(x, y, xoffset, yoffset) = face_cascade.detectMultiScale(img, 1.2, 5)

        if np.any(face_cascade.detectMultiScale(img, 1.2, 5)): # should be all... don't ask me
            np.save(I2F_CONFIG["ISOLATE_OUT"] + file, face_cascade.detectMultiScale(img, 1.2, 5))

        #cv2.imwrite(filepath, img[y:y + yoffset, x:x + xoffset])

            # for (x, y, xoffset, yoffset) in faces:
            #     images.append(img[y:y + yoffset, x:x + xoffset])
            #     filenames.append(filename)

        # for image in images:
        #     if image.shape[0] > maxsize:        # assumes square
        #         maxsize = image.shape[0]

        # for image, filename in zip(images, filenames):  # reshapes to max similar size
        #     image = cv2.resize(image, (maxsize, maxsize), interpolation=cv2.INTER_CUBIC)
        #     cv2.imwrite(outDir + "/" + filename, image)

        #print("Done isolating faces")

    def _crop(self, dirpath, file): # in ".../_isolateFaces" (file), out ".../_crop" (img)

        print("crop")
        #print("coords file: %s" % (dirpath+file))
        faceCoords = np.load(dirpath + file)
        (x, y, xoffset, yoffset) = faceCoords[0]

        file = file[:file.rfind(".")]

        img = cv2.imread(I2F_CONFIG["SHRINK_OUT"] + file)
        #print(I2F_CONFIG["SHRINK_OUT"] + file)
        #print(type(img))
        cv2.imwrite(I2F_CONFIG["CROP_OUT"] + file, img[y:y + yoffset, x:x + xoffset])

    def _maxDim(self, directory):

        print("maxdim")

        maxsize = 0

        paths = [path.join(directory, filename) for filename in listdir(directory)]  # what if there are files that aren't images

        for filepath in paths:
            img = cv2.imread(filepath)
            if img.shape[0] > maxsize:              # assumes square output from
                maxsize = img.shape[0]

        np.save(I2F_CONFIG["DIM_OUT"] + "maxsize.npy", maxsize)

        # for image in images:
        #     if image.shape[0] > maxsize:        # assumes square
        #         maxsize = image.shape[0]

    def _resize(self, directory):

        print("resize")

        util = fileutil()

        maxsize = np.load(I2F_CONFIG["DIM_OUT"] + "maxsize.npy").item()

        paths = [path.join(directory, filename) for filename in listdir(directory)]

        for filepath in paths:
            img = cv2.imread(filepath)
            img = cv2.resize(img, (maxsize, maxsize), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(I2F_CONFIG["RESIZE_OUT"] + util.getFileName(filepath), img)

        # for image, filename in zip(images, filenames):  # reshapes to max similar size
        #     image = cv2.resize(image, (maxsize, maxsize), interpolation=cv2.INTER_CUBIC)
        #     cv2.imwrite(outDir + "/" + filename, image)