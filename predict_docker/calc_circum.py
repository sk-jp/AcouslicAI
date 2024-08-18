import cv2
import numpy as np

from ellipse_circumference import ellipse_circumference


def calc_circum(mask, mm_pixel):
    mask = mask.astype(np.uint8)
    
    # measure the circum
    # get contours
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # select maximum contour
    max_contour = max(contours, key=lambda x: cv2.contourArea(x))

    # fit ellipse
    (cx, cy), (width, height), angle = cv2.fitEllipse(max_contour)           

    # get arc length
    a = max(width, height)
    b = min(width, height)
    circum = ellipse_circumference(a, b)
    circum = circum * mm_pixel
    
    # flattening
    flattening = 1 - b / a
        
    return circum, flattening
