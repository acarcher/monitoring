'''
Uses OpenCV to turn a video into frames
'''

import cv2
import sys
import os.path

def extractframes():
    if len(sys.argv) != 2:
        print("Usage: extractframes.py filepath")
        return
    
    path = sys.argv[1]
    head, tail = os.path.split(path)

    framePath = head + "/" + os.path.splitext(tail)[0] + "_frames"
    
    if not os.path.isfile(path):
        print("Error: {} not found".format(path))
        return

    if not os.path.exists(framePath):
        try:
            os.mkdir(framePath)
        except OSError:
            print("Error: Creating the directory")
            return
    
    currentFrame = 0
    cap = cv2.VideoCapture(path)
    success = True

    while success:

        success, frame = cap.read()
        if not success: break

        name = framePath + "/" + str(currentFrame) + ".jpg"
        print ('Creating... ' + name)
        cv2.imwrite(name, frame)   
        
        currentFrame += 1

    cap.release()
    cv2.destroyAllWindows()

extractframes()