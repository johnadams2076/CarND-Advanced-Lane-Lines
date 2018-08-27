import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Load our image - this should be a new frame since last time!
#binary_warped = mpimg.imread('../sobel/colorspaces/colorspace_test_images/warped-example.jpg')

# Polynomial fit values from the previous frame
# Make sure to grab the actual values from the previous step in your project!
#left_fit = np.array([2.23090058e-04, -3.90812851e-01, 4.76902175e+02])
#right_fit = np.array([4.19709859e-04, -4.93848953e-01, 1.11522544e+03])
from util.Utils import plot_back_to_orig
from util.global_variables import GlobalVar



def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    try:
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = (1 * ploty ** 2 + 1 * ploty)
        right_fitx = (1 * ploty ** 2 + 1 * ploty)

    return left_fit, right_fit, left_fitx, right_fitx, ploty


def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if (not ((leftx.size == 0 or lefty.size == 0 or rightx.size == 0 or righty.size == 0 ))):

        # Fit new polynomials
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)


        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # window_img = np.zeros_like(out_img)
        # # Color in left and right line pixels
        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        #
        # # Generate a polygon to illustrate the search window area
        # # And recast the x and y points into usable format for cv2.fillPoly()
        # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
        #                                                                 ploty])))])
        # left_line_pts = np.hstack((left_line_window1, left_line_window2))
        # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
        #                                                                  ploty])))])
        # right_line_pts = np.hstack((right_line_window1, right_line_window2))
        #
        # # Draw the lane onto the warped blank image
        # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##
        #out_img = plot_back_to_orig(left_fitx, right_fitx, ploty)
    else:
        from util.Utils import find_lane_boundary
        # GlobalVar().set_idx(0)
        # from util.sliding_window import fit_polynomial
        # out_img, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty  = fit_polynomial(binary_warped)

    return left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty


# Run image through the pipeline
# Note that in your project, you'll also want to feed in the previous fits
#result = search_around_poly(binary_warped)

# View your output
#plt.imshow(result)
#plt.show()