#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to run a CNN based face detector using dlib.  The
#   example loads a pretrained model and uses it to find faces in images.  The
#   CNN model is much more accurate than the HOG based model shown in the
#   face_detector.py example, but takes much more computational power to
#   run, and is meant to be executed on a GPU to attain reasonable speed.
#
#   You can download the pre-trained model from:
#       http://dlib.net/files/mmod_human_face_detector.dat.bz2
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

from imutils.video import FileVideoStream
import sys
import dlib
import cv2
import argparse
import time  # because we initially sleep after loading the vdo, we ggive it one sec to stabilize before running this program on it
import face_recognition # for 128 parameterization
import pickle

def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

#if len(sys.argv) < 3:
    #print(
        #"Call this program like this:\n"
        #"   ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg\n"
        #"You can get the mmod_human_face_detector.dat file from:\n"
        #"    http://dlib.net/files/mmod_human_face_detector.dat.bz2")
    #exit()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video")
ap.add_argument("-m", "--model", required=True,
    help="path to the cnn model to be used")
ap.add_argument("-e", "--encodings", required=True,
	help="path to store db of facial encodings from the input video")
ap.add_argument("-d", "--dump", required=True,
	help="path to dump frames with faces from the input video")
args = vars(ap.parse_args())

print("[INFO] starting to read video file...")
fvs = FileVideoStream(args["input"]).start()
time.sleep(1.0)
frameCount = 0

cnn_face_detector = dlib.cnn_face_detection_model_v1(args["model"])  # initializing the face detector
win = dlib.image_window()

#knownEncodings = []
#knownFrames = []

data = []


while fvs.more():  # until more frames are there in the vdo (basically until vdo keeps running)
    frame = fvs.read()
    if frame is None:
        continue
    boxes = []
    frameCount = frameCount + 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #print("Processing file: {}".format(f))
    #img = dlib.load_rgb_image(frame)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = cnn_face_detector(rgb, 1)
    '''
    This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
    These objects can be accessed by simply iterating over the mmod_rectangles object
    The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.
    
    It is also possible to pass a list of images to the detector.
        - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)

    In this case it will return a mmod_rectangless object.
    This object behaves just like a list of lists and can be iterated over.
    '''
    print("Number of faces detected: {}".format(len(dets)))
    if (len(dets) > 0):  # dets is not empty i.e, faces have been detected
        #Dump the frame
        frameName = args["dump"] + "/Frame_" + str(frameCount) + ".jpg"
        cv2.imwrite(frameName, frame)
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))  # note that this is a normal print statement and does not print stuff in the image

    rects = dlib.rectangles()
    rects.extend([d.rect for d in dets])
    for d in dets:
        boxes.append(_rect_to_css(d.rect))
    encodings = face_recognition.face_encodings(rgb, boxes)  # in 1 rgb frame, whatever boxes are detected, each face (box) is converted into 128 parameters

    d = [{"imagePath": frameName, "loc": box, "encoding": enc}  # array of maps is created (the content of the pickle file is created)
		for (box, enc) in zip(boxes, encodings)]
    data.extend(d)

    win.clear_overlay()  # clearing the boxes that were drawn on the earlier frame
    win.set_image(rgb)   # draw the new frame
    win.add_overlay(rects) # overlay the boxes on the frame
    #dlib.hit_enter_to_continue()

fvs.stop()

print("[INFO] serializing all face encodings...")
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
