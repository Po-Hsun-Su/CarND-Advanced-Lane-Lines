import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

img = mpimg.imread("test_images/straight_lines1.jpg")
with open("calibration.pickle", "rb") as handle:
    calibration = pickle.load(handle)
print(calibration)
mtx = calibration["mtx"]
dist = calibration["dist"]
img = cv2.undistort(img, mtx, dist, None, mtx)
plt.figure()
plt.imshow(img)


src = np.float32(
    [[710, 464],
     [1055, 689],
     [248, 689],
     [574, 464]])

for i in range(4):
    plt.plot(src[i, 0], src[i, 1], '.')


dst = np.float32(
    [[1055, 0],
     [1055, 720],
     [248, 720],
     [248, 0]])

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
plt.figure()
plt.imshow(warped)
plt.show()
