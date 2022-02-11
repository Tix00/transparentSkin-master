import source.skinSegmentation.skinSegmentation as SS
import time
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2

"""
def show_webcam(mirror=False):
    cv2.CAP_PROP_BUFFERSIZE = 0
    cam = cv2.VideoCapture(0)

    while True:
        ret_val, img = cam.read()
        start_time = time.time()
        maskSS = SS.skinSegmentation(img)
        print('Elapsed time : ' + str(time.time() - start_time))
        cv2.imshow('my webcam', maskSS)
        cv2.waitKey(1)
        fps.update()
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
"""


def faster_stream(video_path = ""):
    # import the necessary packages

    # construct the argument parse and parse the arguments
    # open a pointer to the video stream and start the FPS timer

    # se non è stato passato alcun parametro, allora abbiamo chiamato la webcam
    if(video_path == ""):
        stream = cv2.VideoCapture(0)
    else:
        stream = cv2.VideoCapture(video_path)
    fps = FPS().start()
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video file stream
        (grabbed, frame) = stream.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # resize the frame and convert it to grayscale (while still
        # retaining 3 channels)
        # display a piece of text to the frame (so we can benchmark
        # fairly against the fast method)
        # show the frame and update the FPS counter

        #frame = SS.skinSegmentation(frame)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        fps.update()
