# a = 0.353772
# b = 0.324035
# c = (a+b) / 2
# import random
# for _ in range(3):
#     print(c+(a-b)/2*(random.randrange(1000)-500)/500)

# import pandas as pd
# a = [2.378798129,2.379137925,2.379728386,2.378665383,2.380813182,2.379428472,2.378278658,2.37854379,2.37832403,2.377241082,2.378306558,2.378138779]
# b = [0.14646,0.14865,0.145923,0.137563,0.145271,0.144723032,0.142944,0.131826,0.147774,0.137352,0.144257,0.140717264]
#
# data = pd.DataFrame({'a':a,'b':b})
# print(data.corr())

# import fcntl
# f = open('./test.txt','w')
# for i in range(10):
#     f.write(str(i))
# fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
# try:
#     f0 = open('./test.txt','a')
#     f0.write('he')
#     fcntl.flock(f0, fcntl.LOCK_EX|fcntl.LOCK_NB)
#     f0.write('hello')
# except:
#     print('succesfully!')
#     f0.close()
# finally:
#     f.close()
# f0 = open('./test.txt','a')
# fcntl.flock(f0, fcntl.LOCK_EX|fcntl.LOCK_NB)
# f0.write('world!')
# f0.close()

import cv2 as cv
import numpy as np
import os

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=1000,
                          qualityLevel=0.1,
                          minDistance=7,
                          blockSize=7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# The video feed is read in as a VideoCapture object
img_dir = 'image_2'
img_path = [os.path.join(img_dir,img) for img in os.listdir(img_dir)]
n = len(img_path)
# Variable for color to draw optical flow track
color = (0, 255, 0)
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
first_frame = cv.imread(img_path[0])
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
# https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)
fps = 10
size = first_frame.shape[:2]
writer = cv.VideoWriter('sparse_optical_flow.avi', cv.VideoWriter_fourcc(*'XVID'), fps, size)

idx = 0
while idx < n:
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    frame = cv.imread(img_path[idx])
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates sparse optical flow by Lucas-Kanade method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    # Selects good feature points for previous position
    good_old = prev[status == 1]
    # Selects good feature points for next position
    good_new = next[status == 1]
    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        # Draws line between new and old position with green color and 2 thickness
        mask = cv.line(mask, (a, b), (c, d), color, 2)
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        frame = cv.circle(frame, (a, b), 3, color, -1)
    # Overlays the optical flow tracks on the original frame
    output = cv.add(frame, mask)
    # Updates previous frame
    prev_gray = gray.copy()
    # Updates previous good feature points
    #     prev = good_new.reshape(-1, 1, 2)
    # Opens a new window and displays the output frame
    cv.imshow("sparse optical flow", output)
    writer.write(output)
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    prev = good_new.reshape(-1, 1, 2)
    if len(prev) < 50:
        prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        mask = np.zeros_like(first_frame)

# The following frees up resources and closes all windows
cv.destroyAllWindows()
writer.release()
