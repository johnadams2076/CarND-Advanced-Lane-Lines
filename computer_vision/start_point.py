import glob
import time

import cv2
import matplotlib.pyplot as plt

from computer_vision.computer_vision_lane_lines import pipeline

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


white_output = '../output_images/harder_challenge_video_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("../harder_challenge_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


    #final_img, left_curverad, right_curverad = pipeline(img)
    #print(left_curverad, 'm', right_curverad, 'm', '\n')

#cv2.imwrite("../output_images/straight_lines1_final_img.jpg", final_img)
# Get object and image points for camera calibration input.


#plt.imshow(final_img)
#plt.show()
