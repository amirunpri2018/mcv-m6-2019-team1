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
import scipy.ndimage
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2 as cv


# import timeit

from skimage.io import imsave


# Local libraries:
import organize_code.code.utils_opticalFlow as utOF

"""
def main():
    # parse the command line arguments to attributes of 'args'
    parser = argparse.ArgumentParser(description='EBMA searcher. '
                                                 'Estimates the motion between to frame images '
                                                 ' by running an Exhaustive search Block Matching Algorithm (EBMA).')
    parser.add_argument('--anchor-frame', dest='anchor_frame_path', required=True, type=str,
                        help='Path to the frame that will be predicted.')
    parser.add_argument('--target-frame', dest='target_frame_path', required=True, type=str,
                        help='Path to the frame that will be used to predict the anchor frame.')
    parser.add_argument('--frame-width', dest='frame_width', required=True, type=positive_integer,
                        help='Frame width.')
    parser.add_argument('--frame-height', dest='frame_height', required=True, type=positive_integer,
                        help='Frame height.')
    parser.add_argument('--norm', dest='norm', required=False, type=float, default=1,
                        help='Norm used for the DFD. p=1 => MAD, p=2 => MSE. Default: p=1.')
    parser.add_argument('--block-size', dest='block_size', required=True, type=positive_integer,
                        help='Size of the blocks the image will be cut into, in pixels.')
    parser.add_argument('--search-range', dest='search_range', required=True, type=positive_integer,
                        help="Range around the pixel where to search, in pixels.")
    parser.add_argument('--pixel-accuracy', dest='pixel_acc', type=positive_integer, default=1, required=False,
                        help="1: Integer-Pel Accuracy (no interpolation), "
                             "2: Half-Pel Integer Accuracy (Bilinear interpolation")
    args = parser.parse_args()


parser = argparse.ArgumentParser(description='EBMA searcher. '
                                             'Estimates the motion between to frame images '
                                             ' by running an Exhaustive search Block Matching Algorithm (EBMA).')
"""

anchor_frame_path = '/home/agus/repos/mcv-m6-2019-team1/data/kitti_optical_flow/training_sequences/000045_10.png'
target_frame_path = '/home/agus/repos/mcv-m6-2019-team1/data/kitti_optical_flow/training_sequences/000045_11.png'

gt_folder = '/home/agus/repos/mcv-m6-2019-team1/data/kitti_optical_flow/gt'
sequence = '000045_10.png'

#frame_width = 1226
#frame_height = 370
#block_size_list = [4, 8, 16, 32, 64]
#search_range_list = [4, 8, 16, 32, 64]
norm = 1
pixel_acc = 1

vals = list()
mse_script = []
psnr_script = []
msen_values = []
pepn_values = []
time_values = []
block_size_values = []
search_range_values = []

block_size_list = [64]
search_range_list = [4]

def subarray(array, (upper_left_pix_row, upper_left_pix_col), (lower_right_pix_row, lower_right_pix_col)):
    """
    Return a subarray containing the pixels delimited by the pixels between upper_left_pix and lower_right.
    If asked for pixels outside the image boundary, such pixels have value 0.
    """

    if upper_left_pix_row > lower_right_pix_row or upper_left_pix_col > lower_right_pix_col:
        raise ValueError('coordinates of the subarray should correspond to a meaningful rectangle')

    orig_array = np.array(array)

    num_rows = lower_right_pix_row - upper_left_pix_row + 1
    num_cols = lower_right_pix_col - upper_left_pix_col + 1

    subarr = np.zeros((num_rows, num_cols), dtype=orig_array.dtype)

    # zoomed outside the original image
    if lower_right_pix_col < 0 or lower_right_pix_row < 0 or \
                    upper_left_pix_col > orig_array.shape[1] - 1 or upper_left_pix_row > orig_array.shape[0] - 1:
        return subarr

    # region of the original image that is inside the desired region
    # (i = col, j=row)
    # _________________________________
    # |                                | original image
    # |   _____________________________|____
    # |   |(j_o_1, i_o_1)              |    |
    # |   |             (j_o_2, i_o_2) |    |
    # |___|____________________________|    |
    # |                                 |
    # |_________________________________|  sliced final image

    if upper_left_pix_col < 0:
        i_o_1 = 0
    else:
        i_o_1 = upper_left_pix_col
    if upper_left_pix_row < 0:
        j_o_1 = 0
    else:
        j_o_1 = upper_left_pix_row

    if lower_right_pix_col > orig_array.shape[1] - 1:
        i_o_2 = orig_array.shape[1] - 1
    else:
        i_o_2 = lower_right_pix_col
    if lower_right_pix_row > orig_array.shape[0] - 1:
        j_o_2 = orig_array.shape[0] - 1
    else:
        j_o_2 = lower_right_pix_row


    # region of the final image that is inside the original image, and whose content will be taken from the orig im
    # (i = col, j=row)
    # _________________________________
    # |                                | original image
    # |   _____________________________|____
    # |   |(j_f_1, i_f_1)              |    |
    # |   |             (j_f_2, i_f_2) |    |
    # |___|____________________________|    |
    #     |                                 |
    #     |_________________________________|  sliced final image

    if upper_left_pix_col < 0:
        i_f_1 = -upper_left_pix_col
    else:
        i_f_1 = 0
    if upper_left_pix_row < 0:
        j_f_1 = -upper_left_pix_row
    else:
        j_f_1 = 0

    if lower_right_pix_col > orig_array.shape[1] - 1:
        i_f_2 = (orig_array.shape[1] - 1) - upper_left_pix_col
    else:
        i_f_2 = num_cols - 1
    if lower_right_pix_row > orig_array.shape[0] - 1:
        j_f_2 = (orig_array.shape[0] - 1) - upper_left_pix_row
    else:
        j_f_2 = num_rows - 1

    subarr[j_f_1:j_f_2 + 1, i_f_1:i_f_2 + 1] = orig_array[j_o_1:j_o_2 + 1, i_o_1:i_o_2 + 1]

    return subarr

def main():

    for block_size, search_range in itertools.product(block_size_list, search_range_list):

        t0 = time.time()

        # Pixel map of the frames in the [0,255] interval
        #target_frm = np.fromfile(args.target_frame_path, dtype=np.uint8, count=args.frame_height * args.frame_width)
        #anchor_frm = np.fromfile(args.anchor_frame_path, dtype=np.uint8, count=args.frame_height * args.frame_width)
        #target_frm = np.reshape(target_frm, (args.frame_height, args.frame_width))
        #anchor_frm = np.reshape(anchor_frm, (args.frame_height, args.frame_width))

        res_filename = 'b' + str(block_size) + 's' + str(search_range)

        target_frm = cv.imread(target_frame_path)
        anchor_frm = cv.imread(anchor_frame_path)

        target_frm = cv.cvtColor(target_frm, cv.COLOR_BGR2GRAY)
        anchor_frm = cv.cvtColor(anchor_frm, cv.COLOR_BGR2GRAY)

        # store frames in PNG for our records
        os.system('mkdir -p frames_of_interest')
        imsave('frames_of_interest/' + res_filename + 'target.png', target_frm)
        imsave('frames_of_interest/' + res_filename + 'anchor.png', anchor_frm)

        ebma = EBMA_searcher(N=block_size,
                             R=search_range,
                             p=norm,
                             acc=pixel_acc)

        predicted_frm, motion_field = \
            ebma.run(anchor_frame=anchor_frm,
                     target_frame=target_frm)

        # store predicted frame
        imsave('frames_of_interest/' + res_filename + 'predicted_anchor.png', predicted_frm)

        motion_field_x = motion_field[:, :, 0]
        motion_field_y = motion_field[:, :, 1]

        # show motion field
        #show_quiver(motion_field_x, motion_field_y[::-1])

        # store error image
        error_image = abs(np.array(predicted_frm, dtype=float) - np.array(anchor_frm, dtype=float))
        error_image = np.array(error_image, dtype=np.uint8)
        imsave('frames_of_interest/' + res_filename + 'error_image_shelf.png', error_image)

        # Peak Signal-to-Noise Ratio of the predicted image
        mse = (np.array(error_image, dtype=float) ** 2).mean()
        psnr = 10 * math.log10((255 ** 2) / mse)
        #print 'PSNR: %s dB' % psnr
        mse_script.append(mse)
        psnr_script.append(psnr)

        utOF.save_motion_field(motion_field_x, motion_field_y, block_size, res_filename, gt_folder, sequence)

        # Calculate error metrics:

        errmap, MSEN, PEPN = utOF.of_metrics(gt_folder, sequence, 'motion_fields', res_filename + 'mfield.png')
        tf = time.time() - t0

        # Save values in lists
        block_size_values.append(block_size)
        search_range_values.append(search_range)
        msen_values.append(MSEN)
        pepn_values.append(PEPN)
        time_values.append(tf)

        print('block', block_size_values)
        print('search', search_range_values)
        print('MSEN', msen_values)
        print('pepn', pepn_values)
        print('mse script', mse_script)
        print('psnr script', psnr_script)

        # Save MSEN errormap
        plt.figure()
        plt.title('Optical Flow Error Map')
        plt.imshow(errmap)
        plt.colorbar()
        plt.savefig('errormaps/' + res_filename + '_error.png')

    # Save results in pkl file:

    vals.append([block_size_values, search_range_values, msen_values, pepn_values, time_values])

    df_metrics = pd.DataFrame(
            vals,
            columns=['block_size', 'search_range', 'msen', 'pepn', 'time']
        )

    df_metrics.to_pickle('of_box_match.pkl')


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
