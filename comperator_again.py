from imutils.video import FileVideoStream
import sys
import dlib
import cv2
import argparse
import time
import face_recognition
import pickle
import numpy as np

def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video")
ap.add_argument("-m", "--model", required=True,
    help="path to the cnn model to be used")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings to be compared against")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading reference face encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

print("[INFO] starting to read input video file...")
fvs = FileVideoStream(args["input"]).start()
time.sleep(1.0)
frameCount = 0

cnn_face_detector = dlib.cnn_face_detection_model_v1(args["model"])
#win = dlib.image_window()

while fvs.more():
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

    #print("Number of faces detected: {}".format(len(dets)))
    #if (len(dets) == 0):
        #Nothing to look here - no face detected...
        #continue

    rects = dlib.rectangles()
    rects.extend([d.rect for d in dets])
    for d in dets:
        boxes.append(_rect_to_css(d.rect))
    encodings = face_recognition.face_encodings(rgb, boxes)

    labels = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        #matches = face_recognition.compare_faces(data["encodings"],
                #encoding, 0.3875)
        matches = face_recognition.compare_faces(data["encodings"],
                encoding, 0.35)
        label = -1
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                label = data["labels"][i]
                counts[label] = counts.get(label, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            #print("[INFO] counts: ", counts)
            label = max(counts, key=counts.get)

        # update the list of labels
        labels.append(label)
        #print("[INFO] labels: ", labels)

    #win.clear_overlay()
    #win.set_image(rgb)
    #for (rect, label) in zip(rects, labels):
        #win.add_overlay(rect)
        #win.add_overlay("" + str(label))
    #dlib.hit_enter_to_continue()
    for ((top, right, bottom, left), label) in zip(boxes, labels):
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 0, 255), 2)
        if (label == -1):
            continue
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, "" + str(label), (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)

    cv2.imshow(args["input"], frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

fvs.stop()