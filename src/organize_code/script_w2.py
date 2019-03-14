
import os
import src.utils as ut
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
frames_dir = '../frames'
gt_file = '../datasets/AICity_data/train/S03/c010/gt/gt.txt'
frame_list = ut.get_files_from_dir2(frames_dir,ext = '.jpg')
frame_list.sort(key=ut.natural_keys)

output_dir = '../week2_results/'
output_subdir = 'BG_3G/'
if not os.path.isdir(output_dir+output_subdir):
    os.mkdir(output_dir+output_subdir)
#frame_list = ut.get_files_from_dir(frames_dir, excl_ext='jpg')
#print frame_list
# training
N =len(frame_list)
d =3
if d==1:
    Clr_flag = cv.IMREAD_GRAYSCALE
else :
    Clr_flag = cv.IMREAD_COLOR
#ffmpeg

# I couldnt run it with 25% on my computer due to memory problem-
#PercentPerTraining = 0.25
Nt = int(N*0.01)
#trainig_list = frame_list[:Nt]
trainig_list = frame_list[:Nt]
testing_list = frame_list[Nt:]
print(np.shape(trainig_list))
print("Total # of frames {}".format(N))
print("Training: # {}".format(len(trainig_list)))
print("Testing: # {}".format(len(testing_list)))


#[muBG,stdBG] = bg.getGauss_bg2(trainig_list, D=3,gt_file = gt_file)
[muBG2,stdBG2] = bg.getGauss_bg2(trainig_list, D=d,gt_file = None,output_dir = output_dir+output_subdir)
#[muBG,stdBG] = bg.getGauss_bg2(trainig_list, D=1)
# Output 1
#Out1 = cv.merge((muBG,stdBG,np.zeros(np.shape(muBG))))
Out2 = cv.merge((muBG2,stdBG2,np.zeros(np.shape(muBG2))))
#print(np.shape(Out1))
#cv.imwrite(output_dir+'BG_1gC_masked.png',Out1)
cv.imwrite(output_dir+output_subdir+'BG.png',Out2)
#np.save(output_dir+'Bg_1g',Out1)

"""
GOOD IDEA IS TO SAVE TO BG MODEL SO WE CAN RUN ON IT DIFFERENT TESTS
"""
fig = plt.figure(1)
ax1 = plt.subplot(211)
    #ax3 = plt.subplot(312)
ax2 = plt.subplot(212)


ax1.imshow(muBG2,cmap='gray')
ax2.imshow(stdBG2,cmap='gray')

ax1.set_title("Mean BG model - #"+str(len(trainig_list))+" frames")
ax2.set_title("Std noise BG model")

plt.show()

# Testing Example
th = 3
I = cv.imread(testing_list[500],Clr_flag)
mapBG = bg.foreground_from_GBGmodel(muBG2,stdBG2,I,th =th)
fig = plt.figure(1)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax1.imshow(I,cmap='gray')
ax2.imshow(mapBG,cmap='gray')
ax1.set_title(testing_list[500])
ax2.set_title("Foreground Map, with th="+str(th))
plt.show()
# showing the background model , mu +noise
#print("mu: {},var: {}".format(len(muBG),len(varBG)))
