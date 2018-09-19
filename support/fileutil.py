#!/usr/bin/python3

from os import path, listdir, makedirs
from shutil import rmtree, copy, copytree
from .config import I2F_CONFIG, V2F_CONFIG, I2F_OUT, V2F_OUT


class fileutil():

    def _createDir(self, directory):
        makedirs(directory, exist_ok=True)

    def createTempIntermediateDirs(self, method, directory=""):

        if method == "videosToFrames":
            for entry in V2F_CONFIG:
                self._createDir(V2F_CONFIG[entry])

            if directory:
                for filename in listdir(directory):
                    self._createDir(V2F_CONFIG["FRAME_OUT"] + filename)

        elif method == "imagesToFaces":
            for entry in I2F_CONFIG:
                self._createDir(I2F_CONFIG[entry])
        else:
            pass

        # caller = inspect.stack()[1].function

        # if caller[0] == "_":
        #     output = PREPROCESSING_PATH
        # else:
        #     output = INTERMEDIATE_PATH

        # #outDirectory = output + caller + filename + "/"
        # outDirectory = output + caller + "/"
        # self._createDir(outDirectory)

        # return outDirectory

    def _deleteDir(self, directory):
        rmtree(directory)

    def deleteTempIntermediateDirs(self, method):

        if method == "videosToFrames":
            for entry in V2F_CONFIG:
                self._deleteDir(V2F_CONFIG[entry])

        elif method == "imagesToFaces":
            for entry in I2F_CONFIG:
                self._deleteDir(I2F_CONFIG[entry])
        else:
            pass

    def createIntermediateDir(self, method):

        if method == "videosToFrames":
            self._createDir(V2F_OUT)

        elif method == "imagesToFaces":
            self._createDir(I2F_OUT)
        else:
            pass

    def copyToIntermediateDir(self, method): # needs to be recursive

        if method == "videosToFrames":

            folders = [path.join(V2F_CONFIG["FRAME_OUT"], foldername) for foldername in listdir(V2F_CONFIG["FRAME_OUT"])]
            # 'FRAME_OUT': PREPROCESS + "_extractFrames/" -> video_1.hevc/ .. etc
            # paths = [path.join(folders, filename) for filename in listdir()]

            for folder in folders:
                #print("copying from %s to %s" % (folder, I2F_OUT + self.getFolderName(folder)))
                copytree(folder, V2F_OUT + self.getFolderName(folder))
                #print("end copy")

            # for filepath in paths:
            #     copy(filepath, V2F_OUT + self.getFileName(filepath))
            #     # copy(filepath, V2F_OUT + self.getFileName(filepath) + "/" + self.getFileName(filepath))

        elif method == "imagesToFaces":
            paths = [path.join(I2F_CONFIG["RESIZE_OUT"], filename) for filename in listdir(I2F_CONFIG["RESIZE_OUT"])]

            for filepath in paths:
                #print("copying from %s to %s" % (filepath, I2F_OUT + self.getFileName(filepath)))
                copy(filepath, I2F_OUT + self.getFileName(filepath))
                #print("end copy")
        else:
            pass

    def getFileName(self, filepath):
        _, filename = path.split(filepath)

        return filename

    def getFolderName(self, directory):
        return path.basename(path.normpath(directory))
