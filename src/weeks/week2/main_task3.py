# -*- coding: utf-8 -*-

# Standard libraries
import os

# Related 3rd party libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import opening, closing

# Local libraries
import utils as ut
import utilsBG as bg
import src
from utilsBG import MOGBackgroundSubstractor, GMGBackgroundSubstractor, LSBPBackgroundSubstractor


def plot_bboxes(img, l_bboxes):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)

    colors = 'bgrcmykw'

    for bboxes in l_bboxes:
        color = colors[np.random.choice(len(colors))]
        for bbox in bboxes:
            rect = patches.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     linewidth=1, edgecolor=color, facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)


# Some constant for the script
COLOR_SPACE = None
COLOR_CHANNELS = None
DIM = 1
EXP_NAME = 'BG1G_noGT'
MODEL = 'MOG'
TASK = 'task3'
WEEK = 'week2'
# Morphology
F_MORPH = True
F_CONN_COMP = True
# Connected components
AREA_MIN = None
AREA_MAX = None
FF_MIN = None
FF_MAX = None
FR_MIN = None
PLOT_BBOX = False

def main():
    """
    Function to compute the metrics for the background and foreground
    estimation of a sequence of frames using state-of-the-art method.

    Meaningful variables are defined at the beginning of this script.

    :return: Nothing
    """

    # Set useful directories
    frames_dir = os.path.join(
        src.ROOT_DIR,
        'datasets',
        'm6_week1_frames',
        'frames')
    results_dir = os.path.join(src.OUTPUT_DIR, WEEK, TASK, EXP_NAME)

    # Ground truth file path and frames' path
    gt_file = os.path.join(src.ROOT_DIR,
                           'datasets', 'AICity_data', 'train', 'S03',
                           'c010', 'gt', 'gt.txt')
    frames_path = ut.get_files_from_dir2(frames_dir, ext='.jpg')
    frames_path.sort(key=ut.natural_keys)

    # Create folders if they don't exist
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # Create the output folder if it does not exist
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # List to store the metrics
    fscore_tot = list()
    iou_tot = list()
    map_tot = list()
    # True positives, false positives and false negatives accumulators
    bbox_tp_tot = 0
    bbox_fn_tot = 0
    bbox_fp_tot = 0

    # DataFrame with the bounding boxes of the ground truth file
    bboxes_gt = ut.get_bboxes_from_MOTChallenge(gt_file)

    # State-of-the-art methods
    bg_subs = {
        'MOG': MOGBackgroundSubstractor(),
        'GMG': GMGBackgroundSubstractor(),
        'LSBP': LSBPBackgroundSubstractor()
    }

    i = 0
    # Iterate over frames
    for f_path in frames_path:
        # Get numpy array representation of the image from path
        img = ut.getImg_D(f_path, D=DIM, color_space=COLOR_SPACE,
                          color_channels=COLOR_CHANNELS)
        # Remove dimensions of size 1
        while img.shape[-1] == 1:
            img = np.squeeze(img, axis=len(img.shape) - 1)

        # Get the frame number from filename
        frm = ut.frameIdfrom_filename(f_path)
        print frm

        # List of ground truth bounding boxes for specific frame
        _, bboxes = ut.getbboxmask(bboxes_gt, frm, img.shape)

        # Compute the mask with the chosen model
        mask = bg_subs[MODEL].apply(img)

        # If morph mask is set, fill the holes of the mask
        if F_MORPH:
            kernel = np.ones((11, 11))
            mask = binary_fill_holes(closing(mask, kernel))
            mask = binary_fill_holes(opening(mask, kernel))

        plt.imsave(os.path.join(results_dir, 'mask_%s.png' % str(i).zfill(4)), mask)
        i += 1
        # If connected component flag is set
        if F_CONN_COMP:
            # For each max, compute the bounding boxes found in the mask
            bboxes_in_img = bg.connected_components(
                mask,
                area_min=AREA_MIN,
                area_max=AREA_MAX,
                ff_min=FF_MIN,
                ff_max=FF_MAX,
                fr_min=FR_MIN,
                plot=PLOT_BBOX)

        # Compute metrics
        fscore, iou, map, bbox_tp, bbox_fn, bbox_fp = bg.compute_metrics_general(
            bboxes, bboxes_in_img, k=5, iou_thresh=0.5)

        # Add metrics to the lists
        fscore_tot.append(fscore)
        iou_tot.append(iou)
        map_tot.append(map)

        # Update the TP, FP and FN
        bbox_tp_tot += bbox_tp
        bbox_fn_tot += bbox_fn
        bbox_fp_tot += bbox_fp

    print('Done!')


if __name__ == '__main__':
    main()