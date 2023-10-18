import os
from datetime import datetime

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from gradio import components
from skimage import measure
from skimage.morphology import skeletonize
from sklearn.neighbors import KDTree

import AutoCheck as Ac
import Config as Cf
import GlobalArgs as Ga
import ImageView as Iv
import IncircleWide as Iw
import Logger as Log
import Reasoning as Re
import imutils

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

image = cv2.imread("C:/Users/86188/Desktop/00000.jpg")
height, width, _ = image.shape
image = cv2.resize(image, (int(width * 0.5), int(height * 0.5)))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("image0", gray)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("image1", gray)
edged = cv2.Canny(gray, 35, 125)
cv2.imshow("image2", edged)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
cv2.imshow("image3", edged)
# find contours in the edge map

cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# cv2.waitKey(0)
max_len = max(len(c) for c in cnts)
for c in cnts:
    # c = np.array(c)
    # c = np.array(c, dtype=np.float32)
    # if the contour is not sufficiently large, ignore it
    # c = c.reshape(-1, 1, 2)
    if cv2.contourArea(c) < 100:
        continue

    # if len(c) != max_len:
    #     continue

    # compute the rotated bounding box of the contour
    orig = image.copy()
    print(c)
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding

  # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)

    cv2.imshow("image4", orig)
cv2.waitKey(0)

# 寻找轮廓
# contours, _ = cv2.findContours(gray, 35, 125)
# max_len = max(len(c) for c in contours)
# # 获取最小矩形
# for c in contours:
#     if len(c) == max_len:
#         rect = cv2.minAreaRect(c)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#
#         # 绘制最小矩形
#         cv2.drawContours(edged, [box], 0, (0, 255, 0), 2)
#
#         # 获取矩形的长和宽
#         width = int(rect[1][0])
#         height = int(rect[1][1])
#
#         # 显示结果
#         cv2.imshow("image", edged)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
