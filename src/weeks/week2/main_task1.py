# -*- coding: utf-8 -*-

# Standard libraries
import os

# Related 3rd party libraries
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Local libraries
import utils as ut
import utilsBG as bg
import src


# Some constant for the script
N = 0.01
GT = 'no'
DIM = 3
EXP_NAME = '{}GT_N{}_DIM{}'.format(GT, N, DIM)
TASK = 'task2'
WEEK = 'week2'


def main():
    """
    Function to compute the background and foreground of a sequence
    of frames using a Gaussian distribution method.

    This script uses the 100*N % first frames for training and the
    rest for testing.

    Meaningful variables are defined at the beginning of this script.
    """

    # Set useful directories
    frames_dir = os.path.join(
        src.ROOT_DIR,
        'datasets',
        'm6_week1_frames',
        'frames')
    results_dir = os.path.join(src.OUTPUT_DIR, WEEK, TASK, EXP_NAME)
    # Ground truth file path
    gt_file = os.path.join(src.ROOT_DIR,
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

    # Flag to show the results based on image dimension
    color_flag = cv.IMREAD_GRAYSCALE if DIM <= 1 else cv.IMREAD_COLOR

    # Get the the images for training
    num_frames_test = int(num_frames * N)

    # Separate frames for training and testing
    train_frames = frame_paths[:num_frames_test]
    test_frames = frame_paths[num_frames_test:]

    # Print useful information
    print("Total number of frames       : {}".format(num_frames))
    print("Number of frames for training: {}".format(len(train_frames)))
    print("Number of frames for testing : {}".format(len(test_frames)))

    # Model numpy files
    mu_file = os.path.join(results_dir, 'mu.npy')
    std_file = os.path.join(results_dir, 'std.npy')

    # If the files exist, load the values. If not, compute them
    if os.path.isfile(mu_file):
        mu_bg = np.load(mu_file)
        std_bg = np.load(std_file)
    else:
        mu_bg, std_bg = bg.getGauss_bg(train_frames, D=DIM, gt_file=gt_file)

    # Plot the mean and the standard deviation computed and save it
    fig = plt.figure(1, figsize=(6, 8))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    if DIM == 3:
        ax1.imshow(mu_bg, vmin=0, vmax=255)
        ax2.imshow(std_bg, vmin=0, vmax=255)
    else:
        ax1.imshow(mu_bg, cmap='gray')
        ax2.imshow(std_bg, cmap='gray')
    ax1.set_title(
        "Mean background model over {} frames".format(len(train_frames)))
    ax2.set_title("Standard noise backgrond model")
    plt.savefig(os.path.join(results_dir, 'mean_std_testing.png'))

    # Threshold to create different masks
    alphas = np.linspace(1.5, 0.25, 4.0)

    # Get image size of the frames
    frame_img = cv.imread(test_frames[1], color_flag)
    # Get bounding boxes from ground truth
    bboxes_gt = ut.get_bboxes_from_MOTChallenge(gt_file)

    # Iterate over testing frames to choose one with bounding boxes
    for test_frame in test_frames:
        # Get frame ID from frame filename
        frm = ut.frameIdfrom_filename(test_frame)
        # TODO: VER
        # Get mask and list of bounding boxes from the
        fore_mask, cbbox = ut.getbboxmask(bboxes_gt, frm, frame_img.shape[:2])

        # If there are bounding boxes in the ground truth
        if any(cbbox):
            frame_img = cv.imread(test_frame, color_flag)
            break

    # Plot different thresholds
    fig, axs = plt.subplots(2, 3, figsize=(15, 6), facecolor='w', edgecolor='g')
    fig.subplots_adjust(hspace=.5, wspace=.01)

    axs = axs.ravel()
    axs[0].imshow(frame_img, cmap='gray')
    axs[0].set_title(os.path.basename(test_frame))

    # Iterate over each bounding box in the ground truth and add them
    # to the image
    for bbox in cbbox:
        # Draw rectangle in the image
        rect = patches.Rectangle((bbox[0], bbox[2]),
                                 bbox[1] - bbox[0],
                                 bbox[3] - bbox[2],
                                 linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        axs[0].add_patch(rect)
    axs[1].imshow(fore_mask, cmap='gray')
    axs[1].set_title('GT map')

    # Plot the mask for the different thresholds
    i = 2
    for alpha in alphas:
        map_bg = bg.foreground_from_GBGmodel(mu_bg, std_bg, frame_img, th=alpha)
        axs[i].imshow(map_bg, cmap='gray')
        axs[i].set_title("th = {}".format(alpha))
        i += 1

    plt.savefig(os.path.join(results_dir, 'thresholds.png'))
    print("Done!")


if __name__ == '__main__':
    main()
