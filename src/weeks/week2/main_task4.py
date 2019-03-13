# -*- coding: utf-8 -*-

# Standard libraries
import os

# Related 3rd party libraries
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Local libraries
#import src.utils as ut
#from src import weeks as w
import utils as ut
import utilsBG as bg


ROOT_DIR = os.path.dirname(
            os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
SRC_DIR = os.path.join(ROOT_DIR, 'src')
DATA_DIR = os.path.join(ROOT_DIR, 'data')


# Some constant for the script
# Precentage of images for training
N = 0.01
GT = 'on'
# Number of Channels
DIM = 1

# Color_space: in the cv format OR none
# cv.COLOR_BGR2HSV
COLOR_SPACE = None

# If we want to select only a few channels from the color space
# Only valid if it isnt Greyscale
COLOR_CHANNELS = []

# refine mask with morphological filters
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

PLOT_FLAG = False

IMG_SHAPE = (1080, 1920)

threshold = 0.5 # IoU Threshold

EXP_NAME = '{}GT_N{}_DIM{}'.format(GT, N, DIM)

if __name__ == '__main__':
    """
    Script to compute the background and foreground of a sequence
    of frames using a Gaussian distribution method.

    This script uses the 100*N % first frames for training and the
    rest for testing.

    Meaningful variables are defined at the beginning of this script.
    """
    # Estimating on 25% of the video frame
    # - Estimation of the background without consideration of the foreground in the gt.txt
    # - Estimation " " with respect to the BBox - ignoring them from the calculation

    # Set useful directories
    frames_dir = os.path.join(w.ROOT_DIR, 'frames')
    results_dir = os.path.join(w.OUTPUT_DIR, 'week2', 'task1', EXP_NAME)
    # Ground truth file path
    gt_file = os.path.join(w.ROOT_DIR,
                           'datasets', 'AICity_data', 'train', 'S03',
                           'c010', 'gt', 'gt.txt')

    # Create folders if they don't exist
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # Get file paths for each of the frames and sort them according
    # to the frame number
    frame_paths = ut.get_files_from_dir2(frames_dir, ext='.jpg')
    frame_paths.sort(key=ut.natural_keys)

    # Total number of frames
    num_frames = len(frame_paths)

    # Get the the images for training
    num_frames_test = int(num_frames * N)

    # Separate frames for training and testing
    train_frames = frame_paths[:num_frames_test]
    test_frames = frame_paths[num_frames_test:]

    # Print useful information
    print("Total number of frames       : {}".format(num_frames))
    print("Number of frames for training: {}".format(len(train_frames)))
    print("Number of frames for testing : {}".format(len(test_frames)))


    """
    I. Training
    """
    # Model numpy files
    mu_file = os.path.join(results_dir, 'mu.npy')
    std_file = os.path.join(results_dir, 'std.npy')



    # If the files exist, load the values. If not, compute them
    if os.path.isfile(mu_file):
        mu_bg = np.load(mu_file)
        std_bg = np.load(std_file)
    else:

        if GT=='no':
            mu_bg, std_bg = bg.getGauss_bg(train_frames,
                                            D=DIM,
                                            gt_file=None,
                                            color_space=COLOR_SPACE,
                                            color_channels= COLOR_CHANNELS)
        else:
            mu_bg, std_bg = bg.getGauss_bg(train_frames,
                                            D=DIM,
                                            gt_file=gt_file,
                                            color_space=COLOR_SPACE,
                                            color_channels= COLOR_CHANNELS)

        if d==1:
            mu_bg =np.squeeze(mu_bg, axis=2)
            std_bg =np.squeeze(std_bg, axis=2)
        # Save the model of the specific exp
        #np.save(output_dir+output_subdir+exp_name+'_mu.npy',muBG)
        #np.save(output_dir+output_subdir+exp_name+'_std.npy',stdBG)

    # Plot the mean and the standard deviation computed and save it

    if PLOT_FLAG:
        fig = plt.figure(1, figsize=(6, 8))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        if DIM == 3:
            ax1.imshow(mu_bg, vmin=0, vmax=255)
            ax2.imshow(std_bg, vmin=0, vmax=255)
        elif d==2:
            s = np.shape(mu_bg)
            mu_bg2 = np.dstack((mu_bg,np.zeros((s[0],s[1]))))
            std_bg2 = np.dstack((std_bg,np.zeros((s[0],s[1]))))

            ax1.imshow(mu_bg2,vmin=0,vmax=255)
            ax2.imshow(std_bg2,vmin=0,vmax=255)
        else:
            ax1.imshow(mu_bg, cmap='gray')
            ax2.imshow(std_bg, cmap='gray')
        ax1.set_title(
            "Mean background model over {} frames".format(len(train_frames)))
        ax2.set_title("Standard noise backgrond model")
        plt.savefig(os.path.join(results_dir, 'mean_std_testing.png'))



    """
    II. Testing
    """


    # Size of images
    s = np.shape(ut.getImg_D(test_frames[loc],D=DIM,color_space = COLOR_SPACE))
    # Get bounding boxes from ground truth
    bboxes_gt = ut.get_bboxes_from_MOTChallenge(gt_file)
    # Threshold to create different masks
    alphas = np.linspace(1.5, 0.25, 4.0)

    fscore_tot = []
    iou_tot = []
    map_tot = []
    bboxTP_tot = 0
    bboxFN_tot = 0
    bboxFP_tot = 0
    precision = []
    recall = []
    fsc = []
    # Loop on all the alphas
    for alpha in alphas:
    # Get image size of the frames
    #frame_img = cv.imread(test_frames[1], color_flag)
    #loc = 1


        # Iterate over testing frames to choose one with bounding boxes
        for test_frame in test_frames:
            # Get frame ID from frame filename
            frm = ut.frameIdfrom_filename(test_frame)

            # Get mask and list of bounding boxes from the
            _, cbbox = ut.getbboxmask(bboxes_gt, frm, (s[0],s[1]))
            # get current frame in the experiment format
            I = ut.getImg_D(test_frame,D=DIM,color_space = COLOR_SPACE,color_channels= COLOR_CHANNELS)
            # get the mask
            map_bg = bg.foreground_from_GBGmodel(mu_bg, std_bg, test_frame, th=alpha)

            # refine mask with morphological filter
            if F_MORPH:
                kernel = np.ones((11, 11))

                morp_masks = binary_fill_holes(closing(map_bg, kernel))
                morp_masks = binary_fill_holes(opening(morp_masks, kernel))

                map_bg = morp_masks
            # Get BBox from mask
            bboxes_in_img = bg.connected_components(map_bg, area_min=AREA_MIN, area_max=AREA_MAX,
                                        ff_min=FF_MIN, ff_max=FF_MAX, fr_min=FR_MIN, plot=PLOT_BBOX)

            # Get mesurments for current frame
            fscore, iou, map, bboxTP, bboxFN, bboxFP = bg.compute_metrics_general(cbbox, bboxes_in_img,
                                                                            k = 5,
                                                                            iou_thresh = 0.5)


            # collect of the measures
            fscore_tot.append(fscore)
            iou_tot.append(iou)
            map_tot.append(map)

            bboxTP_tot += bboxTP
            bboxFN_tot += bboxFN
            bboxFP_tot += bboxFP


        # Precision -Recall of this Experiment
        precision.append( ut.precision(bboxTP_tot, bboxFP_tot))
        recall.append( ut.recall(bboxTP_tot, bboxFN_tot))
        fsc.append( ut.fscore(bboxTP_tot, bboxFP_tot, bboxFN_tot))

    # Compute the mAP over the precision and Recall

    mAp = compute_mAP(precision,recall)
    print('mAP : {}'.format(mAp))
        # If there are bounding boxes in the ground truth
        #if any(cbbox):
            # check if it is a good example for ploting:
            #if PLOT_FLAG:


    # Plot different thresholds
    #fig, axs = plt.subplots(2, 3, figsize=(15, 6), facecolor='w', edgecolor='g')
    #fig.subplots_adjust(hspace=.5, wspace=.01)

    # axs = axs.ravel()
    # axs[0].imshow(frame_img, cmap='gray')
    # axs[0].set_title(os.path.basename(test_frame))
    #
    # # Iterate over each bounding box in the ground truth and add them
    # # to the image
    # for bbox in cbbox:
    #     # Draw rectangle in the image
    #     rect = patches.Rectangle((bbox[0], bbox[2]),
    #                              bbox[1] - bbox[0],
    #                              bbox[3] - bbox[2],
    #                              linewidth=1, edgecolor='r', facecolor='none')
    #     # Add the patch to the Axes
    #     axs[0].add_patch(rect)
    # axs[1].imshow(fore_mask, cmap='gray')
    # axs[1].set_title('GT map')

    # Plot the mask for the different thresholds
    # i = 2
    #
    #
    #     axs[i].imshow(map_bg, cmap='gray')
    #     axs[i].set_title("th = {}".format(alpha))
    #     i += 1
    #
    # plt.savefig(os.path.join(results_dir, 'thresholds.png'))
    # print("Done!")
