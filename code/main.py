

import logging
import os

import numpy as np
import shutil
import time
import glob

from evaluation.evaluation_funcs import performance_evaluation_pixel
from utils import get_img, rgb2hsv, threshold_image, save_image, get_files_from_dir, confusion_matrix, \
    print_confusion_matrix, print_metrics
from export_data import export_image_and_mask

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
TRAIN_GTS_DIR = os.path.join(TRAIN_DIR, 'gt')
TRAIN_MASKS_DIR = os.path.join(TRAIN_DIR, 'mask')
#TEST_DIR = os.path.join('dataset', 'test')

train = True
if train:
    RESULT_DIR = os.path.join('results', 'masks', 'train')
    IMAGE_DIR = TRAIN_DIR
else:
    RESULT_DIR = os.path.join('results', 'masks', 'test')
    IMAGE_DIR = os.path.join('dataset', 'test')

# If the directory already exists, delete it
if os.path.exists(RESULT_DIR):
    shutil.rmtree(RESULT_DIR)

# Create directory
os.makedirs(RESULT_DIR)

# Get list of test images in test directory
test_images = get_files_from_dir(IMAGE_DIR)
#test_images = os.listdir(TEST_DIR)

# Set threshold based on ranges of interest
ths_h = np.array([
      [0.0, 0.05],     # Red threshold
      [0.55, 0.65],   # Blue threshold
      [0.95, 1.0]      # Res threshold
      ])
ths_s = np.array([[0.0, 1.0]])
ths_v = np.array([[0.0, 1.0]])

# Get elapsed time
t0 = time.time()
t_frame = 0

# Iterate over test image paths
for img_dir_temp in glob.glob(IMAGE_DIR + "/*.jpg"):
    img_dir = img_dir_temp.split("/")[-1]

    t_frame_0 = time.time()
    # Get numpy array of the image and convert it to HSV
    img = get_img(IMAGE_DIR, img_dir)
    img_hsv = rgb2hsv(img)
    # img_hsv = ndimage.filters.gaussian_filter(img_hsv, sigma=3)

    # Get the mask of the HSV image
    mask = threshold_image(img_hsv, ths_h, channel=0)
    #mask += threshold_image(img_hsv, ths_s, channel=1)
    #mask += threshold_image(img_hsv, ths_v, channel=2)

    # Create a numpy array where mask values are 255
    final = mask.astype(np.uint8)*255

    # Save mask as image
    fn, d = save_image(final, RESULT_DIR, img_dir)
    logger.info("'{filename}' saved in '{folder}' folder".format(filename=fn, folder=os.path.join(ROOT_DIR, d)))

    t_frame += time.time() - t_frame_0

logger.info(
    "%d masks saved in %.3fs (%.3fs per frame)" % (len(test_images), time.time() - t0, t_frame / len(test_images))
)

### Export images + masks results
# export_image_and_mask(IMAGE_DIR, RESULT_DIR, RESULT_DIR.replace("masks","image+masks"))


# conf_mat = confusion_matrix(RESULT_DIR, TRAIN_MASKS_DIR)
# print_confusion_matrix(conf_mat)
# metrics = performance_evaluation_pixel(*conf_mat)
# print_metrics(metrics)
