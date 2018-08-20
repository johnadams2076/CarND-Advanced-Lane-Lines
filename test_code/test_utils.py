import glob

import cv2


def test_get_undistort_image(objpoints, imgpoints):
    # Make a list of calibration images
    images = glob.glob('../test_images/*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        undist = get_undistort_image(img, objpoints, imgpoints)

        cv2.imwrite(undist, '../output_images/undistorted/', fname)