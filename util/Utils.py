import numpy as np
import cv2

from util.global_variables import GlobalVar


def get_undistorted_image(img):
    # Use cv2.calibrateCamera() and cv2.undistort()
    # Convert to grayscale
    grayscaleimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    obj_points, img_points, = GlobalVar().ret_calib_points()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, grayscaleimage.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def get_threshold_binary_image(img):
    gradx = apply_abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100))
    grady = apply_abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(20, 100))
    mag_binary = apply_mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100))
    dir_binary = apply_dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1) & (grady == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def apply_abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if (orient == 'x'):
        sobelderivative = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobelderivative = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel_dt = np.absolute(sobelderivative)
    scaled_sobel_dt = np.uint8(255 * abs_sobel_dt / np.max(abs_sobel_dt))
    binary_output = np.zeros_like(scaled_sobel_dt)
    binary_output[(scaled_sobel_dt >= thresh[0]) & (scaled_sobel_dt <= thresh[1])] = 1
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def apply_mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx_derivative = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely_derivative = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel_mag = np.sqrt(np.square(sobelx_derivative) + np.square(sobely_derivative))
    scaled_sobel_dt = np.uint8(255 * abs_sobel_mag / np.max(abs_sobel_mag))
    binary_output = np.zeros_like(scaled_sobel_dt)
    binary_output[(scaled_sobel_dt >= mag_thresh[0]) & (scaled_sobel_dt <= mag_thresh[1])] = 1
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def apply_dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx_derivative = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely_derivative = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx= np.sqrt(np.square(sobelx_derivative))
    abs_sobely = np.sqrt(np.square(sobely_derivative))
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(dir_grad)
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    return binary_output


def apply_color_transform(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    return s_binary


def apply_perspective(img):
    global src, dst
    src = np.float32([[200, img.shape[0]],[600, 450], [700, 450], [1150, img.shape[0]]])
    dst = np.float32([[350, img.shape[0] ], [350, 0], [950, 0], [950, img.shape[0]]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
    return warped


def is_lane_detected():
    ret_bool = True

    if ((GlobalVar().get_idx()) % (GlobalVar().line_detected.maxlen)) == 0:
        line_detected = GlobalVar().line_detected
        ret_bool = False
        for line_bool in line_detected:
            ret_bool |= line_bool
    return ret_bool


def find_lane_boundary(img):

    from util.sliding_window import fit_polynomial
    #left_fitx = right_fitx = leftx = lefty = rightx = righty = []
    if (GlobalVar().get_idx() <= 0) | (not is_lane_detected()) | bool(np.array(GlobalVar().get_left_fit()).any()) & bool(np.array(GlobalVar().get_right_fit()).any()):
        left_fit, right_fit, left_fitx, right_fitx , leftx, lefty, rightx, righty = fit_polynomial(img)
        GlobalVar().set_left_fit(left_fit)
        GlobalVar().set_right_fit(right_fit)
    elif GlobalVar().get_left_fit().shape[0] == img.shape[0]:
        from util.prev_poly import search_around_poly
        left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty = search_around_poly(img, GlobalVar().get_left_fit(),  GlobalVar().get_right_fit())
    GlobalVar().set_idx(GlobalVar().get_idx() + 1)

    # win_center_img = find_window_centroids(img)
    return GlobalVar().get_left_fit(),  GlobalVar().get_right_fit(), left_fitx, right_fitx, leftx, lefty, rightx, righty


def get_curve_pos():
    # radius = measure_curvature_real(img, fitx, ploty)
    pass

    # Read in the saved objpoints and imgpoints


def plot_back_to_orig(left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(GlobalVar().get_orig_image()).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    #ploty = np.linspace(0, orig_image.shape[0] - 1, orig_image.shape[0])
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(warp_zero, Minv, (GlobalVar().get_orig_image().shape[1], GlobalVar().get_orig_image().shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(GlobalVar().get_orig_image(), 1, newwarp, 0.3, 0)
    return result
