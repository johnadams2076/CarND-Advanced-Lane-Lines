import math

import numpy as np
import cv2

from util.Line import Line
from util.Utils import get_undistorted_image, get_threshold_binary_image, apply_perspective, find_lane_boundary, \
    plot_back_to_orig

#image = cv2.imread('../test_images/test6.jpg')
from util.global_variables import GlobalVar

skipcount = 0
text = ""


def is_santiycheck_ok(imgshape, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty , left_curverad, right_curverad):
    has_passed = True

    #check values
    has_passed &= bool(np.array(left_fit).any()) & bool(np.array(right_fit).any())
    #check curvature
    #has_passed = has_passed & (np.abs(right_curverad - left_curverad) <= 1000)
    # check distance
    maxdist = (np.nanmax(right_fitx) - np.nanmin(left_fitx))
    has_passed &= (maxdist > 0) & (maxdist <= imgshape[1])
    mindist = (np.nanmin(right_fitx) - np.nanmax(left_fitx))
    has_passed &= (mindist > 0) & (mindist <= 600)

    min_y = imgshape[0] // 2
    max_y = imgshape[0]


    # check slope

    [slope_left, intercept_left] = np.polyfit(leftx, lefty,1)

    [slope_right, intercept_right] = np.polyfit(rightx, righty, 1)

    # minLeftX = math.floor((min_y - intercept_left) / slope_left)
    # maxLeftX = math.floor((max_y - intercept_left) / slope_left)
    #
    # minRightX = math.floor((min_y - intercept_right) / slope_right)
    # maxRightX = math.floor((max_y - intercept_right) / slope_right)

    # has_passed &= (maxRightX - minLeftX) > 0 & (minLeftX - maxRightX) <= np.int(imgshape[1]*(3/4))
    # has_passed &= (minRightX - maxLeftX) > 0 & (minRightX - maxLeftX) <= np.int(imgshape[1] * (1/4))

    has_passed &= (5 >= np.abs(slope_left) >= 0.1)  &  (5 >= np.abs(slope_right) >= 0.1) & (np.abs(slope_left - slope_right) <= 6)
    return has_passed


def get_last_conf_vals():
    average_left_fitx = []
    average_right_fitx = []
    left_curverad = []
    right_curverad = []
    if len(GlobalVar().left_lines) > 0:
        left_lane = GlobalVar().left_lines[len(GlobalVar().left_lines) - 1]
        left_curverad = left_lane.radius_of_curvature
        average_left_fitx = left_lane.bestx
        average_left_fit = left_lane.best_fit

    if len(GlobalVar().right_lines) > 0:
        right_lane = GlobalVar().right_lines[len(GlobalVar().right_lines) - 1]
        right_curverad = right_lane.radius_of_curvature
        average_right_fitx = right_lane.bestx
        average_right_fit = right_lane.best_fit

    offset = GlobalVar().get_offset()

    return average_left_fit, average_right_fit, average_left_fitx, average_right_fitx, left_curverad, right_curverad, offset


def pipeline(img):
    # Get undistorted image
    out_img = img
    GlobalVar().set_orig_image(img)
    undistorted_img = get_undistorted_image(img)
    thresh_bin_img = get_threshold_binary_image(undistorted_img)
    warped_img = apply_perspective(thresh_bin_img)
    left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty = find_lane_boundary(warped_img)
    from util.radius_curve import measure_curvature_real
    left_curverad, right_curverad = measure_curvature_real(warped_img, left_fitx, right_fitx)
    average_left_fitx = left_fitx
    average_right_fitx = right_fitx
    offset = 0.0
    average_left_fit = left_fit
    average_right_fit = right_fit

    if is_santiycheck_ok(img.shape, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty,left_curverad, right_curverad):

        left_line, right_line = initialize_lines(left_fit, left_fitx, leftx, lefty, right_fit, right_fitx, rightx,
                                                 righty)

        offset, offsetLeftLine, offsetRightLine = calculate_offset(img, left_fitx, right_fitx)

        left_line.line_base_pos = offsetLeftLine
        right_line.line_base_pos = offsetRightLine
        GlobalVar().set_offset(offset)

        left_line.radius_of_curvature = left_curverad
        right_line.radius_of_curvature = right_curverad

        average_left_fit, average_right_fit, average_left_fitx, average_right_fitx = process_lines(left_fit, left_fitx, left_line, right_fit, right_fitx, right_line)
        GlobalVar().line_detected.append(True)

        GlobalVar().left_lines.append(left_line)
        GlobalVar().right_lines.append(right_line)

    elif (len(GlobalVar().left_lines) > 0) & (len(GlobalVar().right_lines) > 0) :
        average_left_fit, average_right_fit, average_left_fitx, average_right_fitx, left_curverad, right_curverad, offset = get_last_conf_vals()

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    GlobalVar().set_left_fit(average_left_fit)
    GlobalVar().set_right_fit(average_right_fit)
    if bool(np.array(average_left_fitx).any()) & bool(np.array(average_right_fitx).any()):
        offset = GlobalVar().offset
        out_img = plot_back_to_orig(average_left_fitx, average_right_fitx, ploty)
        annotate_vals(left_curverad,  offset, out_img, right_curverad)

        # print(left_curverad, 'm', right_curverad, 'm', '\n')
    return out_img


def initialize_lines(left_fit, left_fitx, leftx, lefty, right_fit, right_fitx, rightx, righty):
    left_line = Line()
    right_line = Line()
    left_line.allx = leftx
    left_line.ally = lefty
    right_line.allx = rightx
    right_line.ally = righty
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit
    left_line.current_fitx = left_fitx
    right_line.current_fitx = right_fitx
    return left_line, right_line


def annotate_vals(left_curverad, offset, out_img, right_curverad):
    float_formatter = lambda x: "%.2f" % x
    global skipcount, text
    if (skipcount % 2 == 0):
        text = "(" + str(float_formatter(right_curverad)) + "m" + ", " + str(
            float_formatter(left_curverad)) + "m" + ") - " + str(float_formatter(offset)) + "m"
    cv2.putText(out_img, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 1, lineType=cv2.LINE_AA)
    skipcount += 1


def calculate_offset(img, left_fitx, right_fitx):
    lane_midpoint = ((np.nanmax(right_fitx) - np.nanmin(left_fitx)) / 2) + np.nanmin(left_fitx)
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    offset = np.abs(img.shape[1] // 2 - lane_midpoint) * xm_per_pix
    offsetLeftLine = np.abs(img.shape[1] // 2 - np.nanmin(left_fitx)) * xm_per_pix
    offsetRightLine = np.abs(img.shape[1] // 2 - np.nanmax(right_fitx)) * xm_per_pix
    return offset, offsetLeftLine, offsetRightLine


def process_lines(left_fit, left_fitx, left_line, right_fit, right_fitx, right_line):
    left_lines = GlobalVar().left_lines
    recent_xfitted = []
    recent_polycoeff = []
    average_left_fit = left_fit
    average_left_fitx = left_fitx
    average_right_fitx = right_fitx
    if (len(left_lines) > 0):
        for temp_line in left_lines:
            recent_xfitted.append(temp_line.current_fitx)
            recent_polycoeff.append(temp_line.current_fit)

        average_left_fitx = np.mean(recent_xfitted,  axis=0, keepdims=True)
        if len(recent_polycoeff) > 1:
            average_left_fit = np.mean(recent_polycoeff, axis=0, keepdims=False)
        fit_diifs = np.subtract(left_fit, recent_polycoeff[len(recent_polycoeff) - 1])
        left_line.best_fit = average_left_fit
        left_line.bestx = average_left_fitx
        left_line.diffs = fit_diifs
    # Right Lane Lines
    right_lines = GlobalVar().right_lines
    recent_xfitted = []
    recent_polycoeff = []
    average_right_fit = right_fit
    if (len(right_lines) > 0):
        for temp_line in right_lines:
            recent_xfitted.append(temp_line.current_fitx)
            recent_polycoeff.append(temp_line.current_fit)
        average_right_fitx = np.mean(recent_xfitted, axis=0, keepdims=True)
        if len(recent_polycoeff) > 1:
            average_right_fit = np.mean(recent_polycoeff, axis=0, keepdims=False)
        fit_diifs = np.subtract(right_fit, recent_polycoeff[len(recent_polycoeff) - 1])
        right_line.best_fit = average_right_fit
        right_line.bestx = average_right_fitx
        right_line.diffs = fit_diifs
    return average_left_fit, average_right_fit, average_left_fitx, average_right_fitx

# performs the camera calibration, image distortion correction and
# returns the undistorted image


#from util.Utils import get_undistorted_image, get_threshold_binary_image, apply_perspective, find_lane_boundary

