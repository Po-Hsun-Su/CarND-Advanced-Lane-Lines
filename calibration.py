# %%
import cv2
import numpy as np

# %%

fname = "camera_cal/calibration1.jpg"
img = cv2.imread(fname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1 use (9, 5)
# 4 and 5 fail
# others use (9, 6)


cases = [(9,6), (8, 6), (9, 5), (8, 5)] # cases to find corners
for case in cases:
    ret, corners = cv2.findChessboardCorners(gray, case)
    if ret:
        img = cv2.drawChessboardCorners(img, case, corners, ret)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        break
