import glob
import cv2
import numpy as np
import pdb
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def getCalibration():
    # imageList = glob.glob("camera_cal/*.jpg")
    valid_id = list(range(2, 4)) + list(range(6, 21))
    imageList = []
    for i in valid_id:
        imageList.append("camera_cal/calibration" + str(i) + ".jpg")
    # print(imageList)

    objpoints = []
    imgpoints = []
    nx = 9
    ny = 6
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for filename in imageList:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        if ret == True:
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            imgpoints.append(corners)
            objpoints.append(objp)
            # cv2.imshow(filename, img)
            # cv2.waitKey()

    img = cv2.imread(imageList[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # for filename in imageList:
    #     img = cv2.imread(filename)
    #     dst = cv2.undistort(img, mtx, dist, None, mtx)
    #     cv2.imshow(filename, dst)
    #     cv2.waitKey()
    calibration = {"mtx": mtx, "dist": dist}
    with open("calibration.pickle", 'wb') as handle:
        pickle.dump(calibration, handle)
    return mtx, dist

def getPerspective():
    src = np.float32(
        [[710, 464],
         [1055, 689],
         [248, 689],
         [574, 464]])


    dst = np.float32(
        [[1055, 0],
         [1055, 720],
         [248, 720],
         [248, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    with open("calibration.pickle", 'rb') as handle:
        calibration = pickle.load(handle)
    calibration["M"] = M
    with open("calibration.pickle", 'wb') as handle:
        pickle.dump(calibration, handle)
    return M

def calibrate(img, mtx, dist, M):

    return img

def get_feature(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1



    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title('Stacked thresholds')
    ax1.imshow(color_binary)

    ax2.set_title('Combined S channel and gradient thresholds')
    ax2.imshow(combined_binary, cmap='gray')

    return combined_binary

def fit(binary_warped):
    print(type(binary_warped))
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each


    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    plt.figure()
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = binary_warped.shape[0]
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    overlay_img = np.zeros((out_img.shape[0], out_img.shape[1], 4))
    overlay_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0, 1]
    overlay_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255, 1]
    for y in range(binary_warped.shape[0]):
        overlay_img[int(left_fitx[i]):int(right_fitx[i]),i] = [0, 255, 0, 1]
    plt.figure()
    plt.imshow(over)


    print(left_curverad, 'm', right_curverad, 'm')

    return left_curverad, right_curverad, overlay_img

def main():
    # mtx, dist = getCalibration()
    # M = getPerspective()
    with open('calibration.pickle', 'rb') as handle:
        calibration = pickle.load(handle)
    print(calibration)
    mtx = calibration["mtx"]
    M = calibration["M"]
    dist = calibration["dist"]
    img = mpimg.imread("test_images/test1.jpg")
    img = cv2.undistort(img, mtx, dist, None, mtx)
    feature = get_feature(img)
    feature = cv2.warpPerspective(feature, M, (feature.shape[1], feature.shape[0]))
    left_fit, right_fit, left_curv, right_curv = fit(feature)
    plt.show()

if __name__ == '__main__':
    main()
