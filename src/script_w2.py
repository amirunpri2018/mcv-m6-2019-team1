
import os
import utils as ut
import utilsBG as bg
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')

# For visulization
import matplotlib.pyplot as plt

"""
task 1
1 Gaussian Background model
Estimating on 25% of the video frame
- Estimation of the background without consideration of the foreground in the gt.txt
- Estimation " " with respect to the BBox - ignoring them from the calculation
'
"""
frames_dir = '../m6_week1_frames/frames'
frame_list = ut.get_files_from_dir2(frames_dir,ext = '.jpg')
#frame_list = ut.get_files_from_dir(frames_dir, excl_ext='jpg')
#print frame_list
# training
N =len(frame_list)
Nt = int(N*0.01)
trainig_list = frame_list[:Nt]
testing_list = frame_list[Nt:]
print(N)
print('Training')
print(len(trainig_list))
print('Testing')
print(len(testing_list))
[muBG,varBG] = bg.getGauss_bg(trainig_list, D=1)

fig = plt.figure(1)
ax1 = plt.subplot(211)
    #ax3 = plt.subplot(312)
ax2 = plt.subplot(212)

#img1_ex = np.expand_dims(muBG,-1)
#img2_ex = np.expand_dims(varBG,-1)
#hsv = np.concatenate((img1_ex,np.zeros_like(img1_ex),img2_ex),axis=2 )
#hsv = cv.cvtColor(hsv,  cv.COLOR_HSV2BGR);
    #cv::imshow("optical flow", bgr);
    #rgb = np.concatenate((img1_ex,img2_ex,img2_ex),axis=2 )
#ax2.imshow(matplotlib.colors.hsv_to_rgb(hsv),cmap='hsv')
#ax2.imshow(hsv)
ax1.imshow(muBG,cmap='gray')
ax2.imshow(varBG,cmap='gray')
    #ax1.imshow(img1,cmap ='Blues')
    #ax1.imshow(img2,cmap ='Reds',alpha=.6)

    #ax2.imshow(img1,cmap='gray')
#ax2.set_title("Mean and Var-noise")
ax1.set_title("Mean BG model - #"+str(len(trainig_list))+" frames")
ax2.set_title("Var noise BG model")

plt.show()
# showing the background model , mu +noise
#print("mu: {},var: {}".format(len(muBG),len(varBG)))
