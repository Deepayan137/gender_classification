import cv2
import numpy as np


def downscale(input, rows=256, cols=256):
    input = np.array(input)
    scale_x = rows/input.shape[0]
    scale_y = rows/input.shape[1]
    final = cv2.resize(input, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
    return final


def upcale(input, rows=256, cols=256):
    input = np.array(input)
    scale_x = rows/input.shape[0]
    scale_y = rows/input.shape[1]
    final = cv2.resize(input, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
    
    return final
