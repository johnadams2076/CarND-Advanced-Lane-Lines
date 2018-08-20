import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
from util.Utils import get_undistorted_image, get_threshold_binary_image, apply_perspective, find_lane_boundary
#from computer_vision.sliding_window import fit_polynomial


def pipeline(img):
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load(open("../util/cam_calibration_pickle.p", "rb"))
    obj_points = dist_pickle["objpoints"]
    img_points = dist_pickle["imgpoints"]

    # Get undistorted image
    undistorted_img = get_undistorted_image(img, obj_points, img_points)
    thresh_bin_img = get_threshold_binary_image(undistorted_img)
    warped_img = apply_perspective(thresh_bin_img)
    lane_boundary_image = find_lane_boundary(warped_img)

    return lane_boundary_image

# performs the camera calibration, image distortion correction and
# returns the undistorted image


image = cv2.imread('../test_images/straight_lines1.jpg')
final_img = pipeline(image)
cv2.imwrite("../output_images/straight_lines1_final_img.jpg", final_img)
# Get object and image points for camera calibration input.


plt.imshow(final_img)
plt.show()
