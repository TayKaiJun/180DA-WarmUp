'''
Baseline Code References: 
- Changing Colorspaces by OpenCV-Python Tutorials (https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)
- Contour Features (https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)

Improvements: 
- Combined both code references and added a minimum bounding box area to filter out small contours that could be noise 

Other References:
- https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
- https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
'''

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(2)

while True:
    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    ### Testing out RGB color space
    # hsv = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # lower_blue = np.array([0, 0, 100])
    # upper_blue = np.array([80, 80, 255])

    mask = cv.inRange(hsv, lower_blue, upper_blue)
    res = cv.bitwise_and(frame, frame, mask=mask)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv.contourArea(contour) > 100:  # only show the bounding box if it is big enough
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
cap.release()
