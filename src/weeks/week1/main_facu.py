# -*- coding: utf-8 -*-

# Built-in modules
import glob
import logging
import os
import shutil
import time

# 3rd party modules
import matplotlib as plt
import numpy as np
import pandas as pd

# Local modules
import src.utils as u


# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# Directory in the root directory where the results will be saved
# Useful directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join('dataset', 'train')
TRAIN_DIR_GT = os.path.join(TRAIN_DIR, 'gt')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

img_shape = (1080, 1920)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    dfs = list()
    for d in os.listdir(os.path.join(ROOT_DIR, 'annotation_pascal')):
        files = u.get_files_from_dir(os.path.join(ROOT_DIR, 'annotation_pascal', d), excl_ext='txt')
        dfs.append(u.get_bboxes_from_pascal(files, d))

    df = pd.concat(dfs)
    print(2)
    # f = os.path.join(ROOT_DIR, 'datasets', 'AICity_data', 'train', 'S03', 'c010', 'Anotation_40secs_AICITY_S03_C010.xml')
    # print(u.get_bboxes_from_aicity(f))
