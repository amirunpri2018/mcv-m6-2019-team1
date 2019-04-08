"""
Estimates the motion between to frame images
 by running an Exhaustive search Block Matching Algorithm (EBMA).
Minimizes the norm of the Displaced Frame Difference (DFD).

Tried in Python 2.7.5

!! DEPENDENCIES !!
This script depends on the following Python Packages:
- argparse, to parse command line arguments
- scikit-image, to save images
- OpenCV, to scale an image
- numpy, to speed up array manipulation
- matplotlib, for quiver
"""

import argparse
import os
import itertools
import math
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2 as cv


# import timeit

from skimage.io import imsave
from utils import positive_integer, subarray, show_quiver

# Local libraries:
import utils_opticalFlow as utOF

target_frame_path = '/home/agus/repos/mcv-m6-2019-team1/data/kitti_optical_flow/training_sequences/000045_10.png'
anchor_frame_path = '/home/agus/repos/mcv-m6-2019-team1/data/kitti_optical_flow/training_sequences/000045_11.png'

gt_folder = '/home/agus/repos/mcv-m6-2019-team1/data/kitti_optical_flow/gt'
sequence = '000045_10.png'

IN_DIR = '../../../../data/kitti_optical_flow/'
OUT_DIR = '../../../../output/week4/task1/'

target_frame_path = IN_DIR + '/training_sequences/000045_10.png'
anchor_frame_path = IN_DIR + '/training_sequences/000045_11.png'

gt_folder = IN_DIR + 'gt'
sequence = '000045_10.png'

#block_size_list = [4, 8, 16, 32, 64]
#search_range_list = [4, 8, 16, 32, 64]
block_size_list = [200]
search_range_list = [2]

norm = 2
pixel_acc = 1

vals = list()
mse_script = []
psnr_script = []
msen_values = []
pepn_values = []
time_values = []
block_size_values = []
search_range_values = []


def main():

    for block_size, search_range in itertools.product(block_size_list, search_range_list):

        t0 = time.time()

        res_filename = 'b' + str(block_size) + 's' + str(search_range)

        target_frm = cv.imread(target_frame_path)
        anchor_frm = cv.imread(anchor_frame_path)

        target_frm = cv.cvtColor(target_frm, cv.COLOR_BGR2GRAY)
        anchor_frm = cv.cvtColor(anchor_frm, cv.COLOR_BGR2GRAY)

        # store frames in PNG for our records
        os.system('mkdir -p frames_of_interest')
        imsave(OUT_DIR + 'frames_of_interest/' + res_filename + 'target.png', target_frm)
        imsave(OUT_DIR + 'frames_of_interest/' + res_filename + 'anchor.png', anchor_frm)

        ebma = EBMA_searcher(N=block_size,
                             R=search_range,
                             p=norm,
                             acc=pixel_acc)

        predicted_frm, motion_field = \
            ebma.run(anchor_frame=anchor_frm,
                     target_frame=target_frm)

        # store predicted frame
        imsave(OUT_DIR + 'frames_of_interest/' + res_filename + 'predicted_anchor.png', predicted_frm)

        motion_field_x = motion_field[:, :, 0]
        motion_field_y = motion_field[:, :, 1]

        # show motion field
        #show_quiver(motion_field_x, motion_field_y[::-1])

        # store error image
        error_image = abs(np.array(predicted_frm, dtype=float) - np.array(anchor_frm, dtype=float))
        error_image = np.array(error_image, dtype=np.uint8)
        imsave(OUT_DIR + 'frames_of_interest/' + res_filename + 'error_image_shelf.png', error_image)

        # Peak Signal-to-Noise Ratio of the predicted image
        mse = (np.array(error_image, dtype=float) ** 2).mean()
        psnr = 10 * math.log10((255 ** 2) / mse)
        #print 'PSNR: %s dB' % psnr
        mse_script.append(mse)
        psnr_script.append(psnr)

        # Save motion fields:
        utOF.save_motion_field(motion_field_x, motion_field_y, block_size, OUT_DIR, res_filename, gt_folder, sequence)

        # Calculate error metrics:

        errmap, MSEN, PEPN = utOF.of_metrics(gt_folder, sequence, OUT_DIR + 'motion_fields', res_filename + 'mfield.png')
        tf = time.time() - t0

        # Save values in lists
        block_size_values.append(block_size)
        search_range_values.append(search_range)
        msen_values.append(MSEN)
        pepn_values.append(PEPN)
        time_values.append(tf)

        print('block', block_size_values)
        print('search', search_range_values)
        print('time', time_values)
        print('MSEN', msen_values)
        print('pepn', pepn_values)
        print('mse script', mse_script)
        print('psnr script', psnr_script)

        # Save MSEN errormap
        plt.figure()
        plt.title('Optical Flow Error Map')
        plt.imshow(errmap)
        plt.colorbar()
        plt.savefig(OUT_DIR + 'errormaps/' + res_filename + '_error.png')

    # Save results in pkl file:

    vals.append([block_size_values, search_range_values, msen_values, pepn_values, time_values])

    df_metrics = pd.DataFrame(
            vals,
            columns=['block_size', 'search_range', 'msen', 'pepn', 'time']
        )

    df_metrics.to_pickle(OUT_DIR + 'of_box_match_norm2.pkl')


class EBMA_searcher():
    """
    Estimates the motion between to frame images
     by running an Exhaustive search Block Matching Algorithm (EBMA).
    Minimizes the norm of the Displaced Frame Difference (DFD).
    """

    def __init__(self, N, R, p=1, acc=1):
        """
        :param N: Size of the blocks the image will be cut into, in pixels.
        :param R: Range around the pixel where to search, in pixels.
        :param p: Norm used for the DFD. p=1 => MAD, p=2 => MSE. Default: p=1.
        :param acc: 1: Integer-Pel Accuracy (no interpolation),
                    2: Half-Integer Accuracy (Bilinear interpolation)
        """

        self.N = N
        self.R = R
        self.p = p
        self.acc = acc

    def run(self, anchor_frame, target_frame):
        """
        Run!
        :param anchor_frame: Image that will be predicted.
        :param target_frame: Image that will be used to predict the target frame.
        :return: A tuple consisting of the predicted image and the motion field.
        """

        acc = self.acc
        height = anchor_frame.shape[0]
        width = anchor_frame.shape[1]
        N = self.N
        R = self.R
        p = self.p

        # interpolate original images if half-pel accuracy is selected
        if acc == 1:
            pass
        elif acc == 2:
            target_frame = cv.resize(target_frame, dsize=(width * 2, height * 2))
        else:
            raise ValueError('pixel accuracy should be 1 or 2. Got %s instead.' % acc)

        # predicted frame. anchor_frame is predicted from target_frame
        predicted_frame = np.empty((height, width), dtype=np.uint8)

        # motion field consisting in the displacement of each block in vertical and horizontal
        motion_field = np.empty((int(height / N), int(width / N), 2))

        # loop through every NxN block in the target image
        for (blk_row, blk_col) in itertools.product(xrange(0, height - (N - 1), N),
                                                    xrange(0, width - (N - 1), N)):

            # block whose match will be searched in the anchor frame
            blk = anchor_frame[blk_row:blk_row + N, blk_col:blk_col + N]

            # minimum norm of the DFD norm found so far
            dfd_n_min = np.infty

            # search which block in a surrounding RxR region minimizes the norm of the DFD. Blocks overlap.
            for (r_col, r_row) in itertools.product(range(-R, (R + N)),
                                                    range(-R, (R + N))):
                # candidate block upper left vertex and lower right vertex position as (row, col)
                up_l_candidate_blk = ((blk_row + r_row) * acc, (blk_col + r_col) * acc)
                low_r_candidate_blk = ((blk_row + r_row + N - 1) * acc, (blk_col + r_col + N - 1) * acc)

                # don't search outside the anchor frame. This lowers the computational cost
                if up_l_candidate_blk[0] < 0 or up_l_candidate_blk[1] < 0 or \
                                low_r_candidate_blk[0] > height * acc - 1 or low_r_candidate_blk[1] > width * acc - 1:
                    continue

                # the candidate block may fall outside the anchor frame
                candidate_blk = subarray(target_frame, up_l_candidate_blk, low_r_candidate_blk)[::acc, ::acc]
                assert candidate_blk.shape == (N, N)

                dfd = np.array(candidate_blk, dtype=np.float16) - np.array(blk, dtype=np.float16)

                if p==2:
                    dfd = dfd.astype(int)

                candidate_dfd_norm = np.linalg.norm(dfd, ord=p)



                # a better matching block has been found. Save it and its displacement
                if candidate_dfd_norm < dfd_n_min:
                    dfd_n_min = candidate_dfd_norm
                    matching_blk = candidate_blk
                    dy = r_col
                    dx = r_row

            # construct the predicted image with the block that matches this block
            predicted_frame[blk_row:blk_row + N, blk_col:blk_col + N] = matching_blk

            #print str((blk_row / N, blk_col / N)) + '--- Displacement: ' + str((dx, dy))

            # displacement of this block in each direction
            motion_field[blk_row / N, blk_col / N, 1] = dx
            motion_field[blk_row / N, blk_col / N, 0] = dy

        return predicted_frame, motion_field


if __name__ == "__main__":
    main()

    # tictoc = timeit.timeit(main, number=1)
    # print 'Time to run the whole main(): %s' %tictoc