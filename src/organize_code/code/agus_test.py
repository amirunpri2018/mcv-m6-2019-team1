import numpy as np
import matplotlib.pyplot as plt
import imageio

import utils_opticalFlow as opt

im = imageio.imread('/home/agus/repos/mcv-m6-2019-team1/datasets/kitti_optical_flow/gt/000045_10.png')

test = opt.readOF('/home/agus/repos/mcv-m6-2019-team1/datasets/kitti_optical_flow/gt', '000045_10.png')

#quiver(X, Y, U, V, **kw)

plt.figure(figsize=(15, 15))
plt.imshow(im[:,:,0])
plt.figure(figsize=(15, 15))
plt.imshow(im[:,:,1])

plt.figure(figsize=(15, 15))
plt.imshow(im[:,:,1] - im[:,:,0])