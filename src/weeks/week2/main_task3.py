
import os
import src.utils as ut
import src.utilsBG as bg
import numpy as np
import cv2 as cv
import matplotlib

from src.utilsBG import MOGBackgroundSubstractor, GMGBackgroundSubstractor, LSBPBackgroundSubstractor

matplotlib.use('TkAgg')

# For visulization
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_erosion, disk, opening, closing
import matplotlib.patches as patches


# Local libraries
import src.utils as ut
import src.utilsBG as bg
from src import weeks as w


def plot_bboxes(img, l_bboxes, title=''):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)

    colors = 'bgrcmykw'

    for bboxes in l_bboxes:
        color = colors[np.random.choice(len(colors))]
        for bbox in bboxes:
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor=color, facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.set

    # plt.show()

"""
task 1
1 Gaussian Background model
Estimating on 25% of the video frame
- Estimation of the background without consideration of the foreground in the gt.txt
- Estimation " " with respect to the BBox - ignoring them from the calculation
'
"""

# SET PARAMETERS:

COLOR_SPACE = None
COLOR_CHANNELS = None
d = 1

# Morphology:
F_MORPH = True
F_CONN_COMP = True

# Connected components:
AREA_MIN = None
AREA_MAX = None
FF_MIN = None
FF_MAX = None
FR_MIN = None
PLOT_BBOX = False

# SET DIRECTORIES:

exp_name = 'BG1G_noGT'

# Set useful directories
frames_dir = os.path.join(w.ROOT_DIR, 'datasets', 'm6_week1_frames', 'frames')
results_dir = os.path.join(w.OUTPUT_DIR, 'week2', 'task1', exp_name)

# Ground truth file path
gt_file = os.path.join(w.ROOT_DIR,
                       'datasets', 'AICity_data', 'train', 'S03',
                       'c010', 'gt', 'gt.txt')
frame_list = ut.get_files_from_dir2(frames_dir,ext = '.jpg')
frame_list.sort(key=ut.natural_keys)

# Create folders if they don't exist
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

output_dir = results_dir
output_subdir = results_dir


# if not os.path.isdir(output_dir+output_subdir):
#     os.mkdir(output_dir+output_subdir)
#frame_list = ut.get_files_from_dir(frames_dir, excl_ext='jpg')
#print frame_list


N =len(frame_list)

# if d==1 or d==0:
#     Clr_flag = cv.IMREAD_GRAYSCALE
# else :
#     Clr_flag = cv.IMREAD_COLOR
# #ffmpeg
# 
# # I couldnt run it with 25% on my computer due to memory problem-
# #PercentPerTraining = 0.25
# Nt = int(N*0.25)
# #trainig_list = frame_list[:Nt]
# trainig_list = frame_list[:Nt]
# testing_list = frame_list[Nt:]
# print(np.shape(trainig_list))
# print("Total # of frames {}".format(N))
# print("Training: # {}".format(len(trainig_list)))
# print("Testing: # {}".format(len(testing_list)))
# 
# 
# if os.path.isfile(output_dir+output_subdir+exp_name+'_mu.npy'):
#     muBG = np.load(output_dir+output_subdir+exp_name+'_mu.npy')
#     stdBG = np.load(output_dir+output_subdir+exp_name+'_std.npy')
# 
# else:
#     # BG_1G/BG1G
#     #[muBG,stdBG] = bg.getGauss_bg(trainig_list, D=d,gt_file = None)
#     # BG_1G_GT/BG1G
#     #gt_file = None
#     [muBG,stdBG] = bg.getGauss_bg(trainig_list,
#                                   D=d,
#                                   gt_file=gt_file,
#                                   color_space=COLOR_SPACE,
#                                   color_channels=COLOR_CHANNELS)
# 
# 
#     if d==1:
#         muBG =np.squeeze(muBG, axis=2)
#         stdBG =np.squeeze(stdBG, axis=2)
#     np.save(output_dir+output_subdir+exp_name+'_mu.npy',muBG)
#     np.save(output_dir+output_subdir+exp_name+'_std.npy',stdBG)



"""
GOOD IDEA IS TO SAVE TO BG MODEL SO WE CAN RUN ON IT DIFFERENT TESTS
"""
"""
fig = plt.figure(1)
ax1 = plt.subplot(211)
    #ax3 = plt.subplot(312)
ax2 = plt.subplot(212)

if d==3:
    ax1.imshow(muBG,vmin=0,vmax=255)
    ax2.imshow(stdBG,vmin=0,vmax=255)
else:
    ax1.imshow(muBG,cmap='gray')
    ax2.imshow(stdBG,cmap='gray')

ax1.set_title("Mean BG model - over "+str(len(trainig_list))+" frames")
ax2.set_title("Std noise BG model")

plt.show()
"""


# Testing Example
th = [2,2.5,3,3.5]
loc = 1
I = cv.imread(frame_list[1])
s = np.shape(I)

fscore_tot = []
iou_tot = []
map_tot = []
# bboxTP_tot = []
# bboxFN_tot = []
# bboxFP_tot = []
bboxTP_tot = 0
bboxFN_tot = 0
bboxFP_tot = 0

Bbox = ut.get_bboxes_from_MOTChallenge(gt_file)  #Ground truth Bboxes

bg_subs = {
    'MOG': MOGBackgroundSubstractor(),
    'GMG': GMGBackgroundSubstractor(),
    'LSBP': LSBPBackgroundSubstractor()
}

model = 'MOG'

for loc, filename in enumerate(frame_list):

    I = ut.getImg_D(filename, D = d,
                                color_space = COLOR_SPACE,
                                color_channels = COLOR_CHANNELS)
    if d==1:
        I =np.squeeze(I,axis=2)
        I =np.squeeze(I,axis=2)

    frm = ut.frameIdfrom_filename(filename)  # Get the number of the frame from filename

    # print(frm)

    _, cbbox = ut.getbboxmask(Bbox, frm, (s[0], s[1]))  # List of GT Bboxes per specific frame
    #m0, cbbox = ut.getbboxmask(Bbox, frm, (s[0], s[1]))

    # results = dict()
    # for model, bg_sub in bg_subs.iteritems():
    #     results[model] = bg_sub.apply(I)
    mapBG = bg_subs[model].apply(I)

    # mapBG = bg.foreground_from_GBGmodel(muBG,stdBG,I,th =3.5) # Mask of foreground for specific frame

    #plt.imshow(mapBG)
    #plt.show()

    # Refine masks and obtain Bboxes:

    # If morph mask is set:
    if F_MORPH:

        #kernel = np.ones((3, 3))
        kernel = np.ones((11, 11))

        morp_masks = binary_fill_holes(closing(mapBG, kernel))
        morp_masks = binary_fill_holes(opening(morp_masks, kernel))


        mapBG = morp_masks

    #plt.imshow(mapBG)
    #plt.show()

    # If connected component flag is set:
    if F_CONN_COMP:
        # For each max, compute the bounding boxes found in the mask
        bboxes_in_img = bg.connected_components(mapBG, area_min=AREA_MIN, area_max=AREA_MAX,
                                                ff_min=FF_MIN, ff_max=FF_MAX, fr_min=FR_MIN, plot=PLOT_BBOX)
    #
    # print(bboxes_in_img)
    # print(cbbox)

    fscore, iou, map, bboxTP, bboxFN, bboxFP = bg.compute_metrics_general(cbbox, bboxes_in_img,
                                                                                k = 5,
                                                                                iou_thresh = 0.5)
    # print(fscore, iou, map, bboxTP, bboxFN, bboxFP)

    # plot_bboxes(mapBG, [bboxes_in_img, cbbox])
    # plot_bboxes(mapBG, bboxes_in_img)
    # plot_bboxes(mapBG, cbbox)

    # plt.show()

    fscore_tot.append(fscore)
    iou_tot.append(iou)
    map_tot.append(map)

    bboxTP_tot += bboxTP
    bboxFN_tot += bboxFN
    bboxFP_tot += bboxFP

    precision = ut.precision(bboxTP_tot, bboxFP_tot)
    recall = ut.recall(bboxTP_tot, bboxFN_tot)
    fsc = ut.fscore(bboxTP_tot, bboxFP_tot, bboxFN_tot)

plt.plot(fscore_tot)
plt.plot(iou_tot)
plt.plot(map_tot)

print 5


    #if not cbbox==[]:
    #    I = cv.imread(testing_list[loc],Clr_flag)
    #    break

"""
fig, axs = plt.subplots(2,3, figsize=(15, 6), facecolor='w', edgecolor='g')
fig.subplots_adjust(hspace = .5, wspace=.01)

axs = axs.ravel()
axs[0].imshow(I,cmap='gray')
axs[0].set_title(testing_list[loc])

for b in cbbox:
    rect = patches.Rectangle((b[0],b[2]),b[1]-b[0],b[3]-b[2],linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    axs[0].add_patch(rect)
axs[1].imshow(m0,cmap='gray')
axs[1].set_title('GT map')
i= 2



for a in th:
    mapBG = bg.foreground_from_GBGmodel(muBG,stdBG,I,th =a)

    axs[i].imshow(mapBG,cmap='gray')

    axs[i].set_title("th={}".format(a))
    i+=1
plt.show()
# showing the background model , mu +noise
#print("mu: {},var: {}".format(len(muBG),len(varBG)))
"""