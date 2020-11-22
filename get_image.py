import os
import cv2
def get_image(path):
    files = os.listdir(path)
    files.sort()
    for filename in files:
        frame_copy = cv2.imread(os.path.join(path, filename))
        yield frame_copy
    return
                
