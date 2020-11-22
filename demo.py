from mecfTracker import mecfTracker
import cv2
from get_image import get_image
import numpy as np
import sys
sys.path.append('./')
# set your video path
path = ""

cap = get_image(path)

# set init bounding box
bbox = [0, 0, 0, 0]

tracker = mecfTracker()
index = 0

for frame in cap:
    if index == 0:
        tracker.init(frame, bbox)
        index += 1
    else:
        _, bbox = tracker.update(frame)
        bbox = list(map(int, map(np.round, bbox)))
        cv2.rectangle(frame,(bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (255,255,255), 1)
        cv2.imshow("video", frame)
        cv2.waitKey(100)
