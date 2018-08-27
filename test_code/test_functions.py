import numpy as np

import cv2
import matplotlib.pyplot as plt
from util.Utils import get_threshold_binary_image, apply_perspective, find_lane_boundary, get_undistorted_image
from util.sliding_window import find_lane_pixels

image = cv2.imread('../test_images/straight_lines1.jpg')
undist = get_undistorted_image(image)
bin_output = get_threshold_binary_image(undist)

# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(bin_output, cmap='gray')
# ax2.set_title("Combined Threshold", fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()

cv2.imwrite("../output_images/test1_binary_output.jpg", bin_output)

warped_img = apply_perspective(bin_output)
cv2.imwrite("../output_images/test1_warped_output.jpg", warped_img)

left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, out_img = find_lane_boundary(warped_img)
cv2.imwrite("../output_images/test1_lane_boundary_output.jpg", out_img)

plt.imshow(out_img)
plt.show()
# out_img[lefty, leftx] = [255, 0, 0]
# out_img[righty, rightx] = [0, 0, 255]
# #
# # # Plots the left and right polynomials on the lane lines
# ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.imshow(out_img)
# plt.show()


