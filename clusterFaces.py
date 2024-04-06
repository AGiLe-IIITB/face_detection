from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-d", "--dump", type=int, default=0,
    help="should the clustered labels be dumped into a pickle file")
ap.add_argument("-o", "--output", 
    help="the pickle file to which the clustered face db will be output to")
ap.add_argument("-j", "--jobs", type=int, default=-1,
    help="# of parallel jobs to run (-1 will use all CPUs)")
args = vars(ap.parse_args())

# load the serialized face encodings + bounding box locations from
# disk, then extract the set of encodings to so we can cluster on
# them
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

# cluster the embeddings
print("[INFO] clustering...")
clt = DBSCAN(eps=0.282, metric="euclidean", n_jobs=args["jobs"], min_samples=10)
#clt = OPTICS(max_eps=0.8, metric="euclidean", n_jobs=args["jobs"])
clt.fit(encodings)
# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

knownFaceLabels = []
knownFaceEncodings = []
labelsIgnored = 0

for labelID in labelIDs:
    # find all indexes into the `data` array that belong to the
    # current label ID, then randomly sample a maximum of 25 indexes
    # from the set
    print("[INFO] faces for face ID: {}".format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]

    randIdxs = np.random.choice(idxs, size=min(25, len(idxs)),
        replace=False)
    # initialize the list of faces to include in the montage
    labelData = []
    labelDataFrames = []
    labelDataFaceLoc = []
    faces = []
    # loop over the sampled indexes
    for i in randIdxs:
        # load the input image and extract the face ROI
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        #print("[INFO] Image Path: ", data[i]["imagePath"])
        #print("[INFO] Top: ", top, " Right: ", right, " Bottom: ", bottom, " Left: ", left)
        face = image[top:bottom, left:right]
        if ((face is None) or
            (top < 0) or (bottom < 0) or (left < 0) or (right < 0)):
            continue
        # force resize the face ROI to 96x96 and then add it to the
        # faces montage list
        face = cv2.resize(face, (96, 96))
        faces.append(face)

    # create a montage using 96x96 "tiles" with 5 rows and 5 columns
    montage = build_montages(faces, (96, 96), (5, 5))[0]
	
    # show the output montage
    title = "Face ID #{}".format(labelID - labelsIgnored)
    title = "Unknown Faces" if labelID == -1 else title
    cv2.imshow(title, montage)
    key = cv2.waitKey(0) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    elif key == ord("n"):
        if (labelID >= 0):
            labelsIgnored = labelsIgnored + 1
        continue
    else:
        if (labelID >= 0):
            # Ignore class label -1 ("unknown")
            labelDataFrames = []
            labelDataFaceLoc = []
            for i in idxs:
                knownFaceLabels.append(labelID - labelsIgnored)
                knownFaceEncodings.append(data[i]["encoding"])
                labelDataFrames.append(data[i]["imagePath"])
                labelDataFaceLoc.append(data[i]["loc"])
            labelData.append({"frame": labelDataFrames, "faceLoc": labelDataFaceLoc})


if (args["dump"] > 0):
    print("[INFO] serializing all clustered face encodings...")
    f = open(args["output"], "wb")
    f.write(pickle.dumps({"encodings": knownFaceEncodings, "labels": knownFaceLabels, "labelledFrameNFace": labelData}))
    f.close()
