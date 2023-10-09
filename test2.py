'''
Baseline Code References: 
- Finding Dominant Colour on an Image (https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097)

Improvements:
- Added a rectangle to the video stream to show the area that is being used to find the dominant colors
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv2.VideoCapture(2)

while True:
    _, frame = cap.read()

    rect_size = (500, 400)
    rect_x = (frame.shape[1] - rect_size[0]) // 2
    rect_y = (frame.shape[0] - rect_size[1]) // 2

    central_rect = frame[rect_y : rect_y + rect_size[1], rect_x : rect_x + rect_size[0]]
    central_rect_rgb = cv2.cvtColor(central_rect, cv2.COLOR_BGR2RGB)
    central_rect_rgb_reshaped = central_rect_rgb.reshape(
        (central_rect_rgb.shape[0] * central_rect_rgb.shape[1], 3)
    )

    # Perform KMeans clustering on the central rectangle in HSV color space
    clt = KMeans(n_clusters=3, random_state=0, n_init=10)
    clt.fit(central_rect_rgb_reshaped)

    hist = find_histogram(clt)
    dominant_colors = clt.cluster_centers_
    bar = plot_colors2(hist, dominant_colors)
    bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)

    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_size[0], rect_y + rect_size[1]), (0, 255, 0), 2)
    cv2.imshow("Video Stream", frame)
    cv2.imshow("Dominant Colors", bar)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
