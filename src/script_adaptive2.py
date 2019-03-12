
import os
import utils as ut
import utilsBG as bg
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')

# For visulization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
output_subdir = 'BG_1G/'
exp_name = 'BG1G'

if not os.path.isdir(output_dir+output_subdir):
    os.mkdir(output_dir+output_subdir)
#frame_list = ut.get_files_from_dir(frames_dir, excl_ext='jpg')
#print frame_list
# training
N =len(frame_list)
d =1
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
[muBG,stdBG] = bg.getGauss_bg(trainig_list, D=d,gt_file = None)
np.save(output_dir+output_subdir+exp_name+'_mu.npy',muBG)
np.save(output_dir+output_subdir+exp_name+'_std.npy',stdBG)


"""
GOOD IDEA IS TO SAVE TO BG MODEL SO WE CAN RUN ON IT DIFFERENT TESTS
"""
fig = plt.figure(1)
ax1 = plt.subplot(211)
    #ax3 = plt.subplot(312)
ax2 = plt.subplot(212)


ax1.imshow(muBG,cmap='gray')
ax2.imshow(stdBG,cmap='gray')

ax1.set_title("Mean BG model - over "+str(len(trainig_list))+" frames")
ax2.set_title("Std noise BG model")

plt.show()

# Testing Example
th = [2,2.5,3,3.5]
I = cv.imread(testing_list[500],Clr_flag)
s = np.shape(I)
Bbox = ut.get_bboxes_from_MOTChallenge(gt_file)
frm = ut.frameIdfrom_filename(testing_list[500])

m0,cbbox = getbboxmask(Bbox,frm,(s[0],s[1]))

fig, axs = plt.subplots(2,3, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()
axs[0].imshow(I,cmap='gray')
axs[0].set_title(testing_list[500])

for b in cbbox:
    rect = patches.Rectangle((b[0],b[2]),b[1]-b[0],b[3]-b[2],linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    axs[0].add_patch(rect)
axs[1].imshow(m0,cmap='gray')
i= 2
for a in th:
    mapBG = bg.foreground_from_GBGmodel(muBG,stdBG,I,th =a)

    axs[i].imshow(mapBG,cmap='gray')

    axs[i].set_title("th={}".format(a))
    i+=1
plt.show()
# showing the background model , mu +noise
#print("mu: {},var: {}".format(len(muBG),len(varBG)))
