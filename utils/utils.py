import numpy as np
import cv2
import random

def create_space(height=30, width=30, custom_space=""):

    w = int(width)
    h = int(height)

    if custom_space == "":

        pixels = np.array([random.randint(0, 255) for i in range(w * h)])/255.
        space = pixels.reshape((h, w))

    else:

        space = cv2.imread(custom_space, 0)
        space = cv2.resize(space, (width, height))/255.


    min = np.min(space)
    min_loc = np.where(space <= min)
    locations = list(zip(min_loc[0], min_loc[1]))
    return space, min, locations
