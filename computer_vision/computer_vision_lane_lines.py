import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
from util.Utils import get_undistorted_image, get_threshold_binary_image, apply_perspective, find_lane_boundary

#image = cv2.imread('../test_images/test6.jpg')
from util.global_variables import GlobalVar


def pipeline(img):
    # Get undistorted image

    GlobalVar().set_orig_image(img)
    undistorted_img = get_undistorted_image(img)
    thresh_bin_img = get_threshold_binary_image(undistorted_img)
    warped_img = apply_perspective(thresh_bin_img)
    lane_boundary_image, left_fit, right_fit = find_lane_boundary(warped_img)
    from util.radius_curve import measure_curvature_real
    left_curverad, right_curverad = measure_curvature_real(img, left_fit, right_fit)
    text = str(left_curverad) + "m" + str(right_curverad) + "m"
    cv2.putText(lane_boundary_image, text, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (255, 255, 255), 3, lineType=cv2.LINE_AA)
    # print(left_curverad, 'm', right_curverad, 'm', '\n')
    return lane_boundary_image


# performs the camera calibration, image distortion correction and
# returns the undistorted image


#from util.Utils import get_undistorted_image, get_threshold_binary_image, apply_perspective, find_lane_boundary

