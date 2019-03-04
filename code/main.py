

import logging
import os

import numpy as np
import shutil
import time
import glob

import utils as u

# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# Directory in the root directory where the results will be saved
# Useful directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join('dataset', 'train')
TRAIN_DIR_GT = os.path.join(TRAIN_DIR, 'gt')

img_shape = [1080, 1920]

# Get Xtml list from Dir:

xml_gt = u.get_files_from_dir(TRAIN_DIR_GT)

# Get GT from Xml (Pandas format Bbox, frame number, track number) :

# ToDo: Write function that reads Xml file and outputs Pandas format Bbox [xtl ytl xbr ybr], frame number,
# track number) .

bboxes_gt = u.bbox_from_xml(xml_gt)     #ToDo!

# Add noise to GT depending on noise parameter:

noisy_bboxes = u.add_noise_to_bboxes(bboxes_gt, img_shape, noise_size=True, noise_size_factor=5.0,
                        noise_position=True, noise_position_factor=5.0)


# Randomly create / destroy Bboxes depending on probability parameter:

added_bboxes = u.create_bboxes(noisy_bboxes, img_shape, prob=0.5)

deleted_bboxes = u.destroy_bboxes(added_bboxes, prob=0.5)


#############################################################
# Compute F-score of GT against modified Bboxes PER FRAME NUMBER:
#############################################################

# ToDo: Add dependency on frame number

# ToDo: Create function fscore

# FIRST CASE: gt against gt (we expect excellent result)

fscore_original = []

for b, box in enumerate(bboxes_gt):
    fscore_original.append(u.fscore(bboxes_gt[b], bboxes_gt[b]))

# SECOND CASE: gt against noisy bboxes (we know the correspondance of gt against bbox)

fscore_noisy = []

for b, box in enumerate(bboxes_gt):
    fscore_noisy.append(u.fscore(bboxes_gt[b], noisy_bboxes[b]))

# THIRD CASE: gt against destroyed / added bboxes (we DONT know the correspondance of gt against bbox)


#############################################################
# Compute IoU of GT against modified Bboxes PER FRAME NUMBER:
#############################################################

# ToDo: Add dependency on frame number

# FIRST CASE: gt against gt (we expect excellent result)

iou_original = []

for b, box in enumerate(bboxes_gt):
    iou_original.append(u.bbox_iou(bboxes_gt[b], bboxes_gt[b])

# SECOND CASE: gt against noisy bboxes (we know the correspondance of gt against bbox)

iou_noisy = []

for b, box in enumerate(bboxes_gt):
    iou_noisy.append(u.bbox_iou(bboxes_gt[b], noisy_bboxes[b])

# THIRD CASE: gt against destroyed / added bboxes (we DONT know the correspondance of gt against bbox)

#ToDo!

#############################################################
# Compute MaP of GT against modified Bboxes PER FRAME NUMBER:
#############################################################

# ToDo: Add dependency on frame number

# FIRST CASE: gt against gt (we expect excellent result)

map_original = []

for b, box in enumerate(bboxes_gt):
    map_original.append(u.mapk(bboxes_gt[b], bboxes_gt[b])

# SECOND CASE: gt against noisy bboxes (we know the correspondance of gt against bbox)

map_noisy = []

for b, box in enumerate(bboxes_gt):
    map_noisy.append(u.mapk(bboxes_gt[b], noisy_bboxes[b])

# THIRD CASE: gt against destroyed / added bboxes (we DONT know the correspondance of gt against bbox)

##########################################
# TASK 2: Plot metrics against frame number
##########################################

#ToDo: Define frame_number!

#ToDo: We can vary noise parameter and plot several noisy results in same graph, labeling them by noise param.

# Plot F-score :

plt.plot(fscore_original, '-o', frame_number, label = 'Original')
plt.plot(fscore_noisy, '-o', frame_number, label='Noisy')
plt.legend()
plt.xlabel('Frame Number')
plt.ylabel('F-score')
plt.title('F-score in time')
plt.savefig('figures/fscore.png')

# Plot IoU :

plt.plot(iou_original, '-o', frame_number, label='Original')
plt.plot(iou_noisy, '-o', frame_number, label='Noisy')
plt.legend()
plt.xlabel('Frame Number')
plt.ylabel('IoU')
plt.title('IoU in time')
plt.savefig('figures/iou.png')

# Plot Mapk :

plt.plot(map_original, '-o', frame_number, label='Original')
plt.plot(map_noisy, '-o', frame_number, label='Noisy')
plt.legend()
plt.xlabel('Frame Number')
plt.ylabel('MaP')
plt.title('MaP in time')
plt.savefig('figures/map.png')

# Should we plot the results against the noise parameter??? I think this makes more sense!!


#######################
# TASK 3: Optical Flow
#######################

# Compute Optical Flow of the two Kitti sequences

# Obtain error metrics:

# MSEN: Mean Square Error in Non-occluded areas

# Use the following function:

# optical_flow_mse = u.mse(gt_optical_flow, result_optical_flow)

# PEPN: Percentage of Erroneous Pixels in Non-occluded areas

# Draw histograms of resulting metrics:

############################
# TASK 4: PLOT Optical Flow
############################


#########################################################################
# conf_mat = confusion_matrix(RESULT_DIR, TRAIN_MASKS_DIR)
# print_confusion_matrix(conf_mat)
# metrics = performance_evaluation_pixel(*conf_mat)
# print_metrics(metrics)
