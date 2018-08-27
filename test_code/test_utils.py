import glob

import cv2

from util.global_variables import GlobalVar


def test_get_undistort_image(objpoints, imgpoints):
    # Make a list of calibration images
    #images = glob.glob('../test_images/*.jpg')
    images = glob.glob('../camera_cal/*.jpg')
    # Step through the list and search for chessboard corners
    idx = 1
    for fname in images:
        img = cv2.imread(fname)
        from util.Utils import get_undistorted_image
        undist = get_undistorted_image(img)

        cv2.imwrite("../output_images/calib"+str(idx)+"_output.jpg", undist)
        idx += 1


obj_points, img_points = GlobalVar().ret_calib_points()
test_get_undistort_image(obj_points, img_points)
