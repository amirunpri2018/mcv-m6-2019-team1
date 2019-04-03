import cv2 as cv
from PIL import Image
import os
import numpy as np
import matplotlib
import scipy.ndimage

#matplotlib.use('Agg')
matplotlib.use('TkAgg')

# For visulization
import matplotlib.pyplot as plt
from matplotlib import gridspec
#import matplotlib.ticker as ticker
# Our LIBS
import utils as utils
from skimage.measure import block_reduce


# Function for Optical Flow
#==========================

def readOF(OFdir,filename):
    """
    Reading Optical flow files:
    As descriped in the Kitti Deelopment kittols
    3 Dim validation - B channel
    1 Dim u - R channel
    2 Dim v - G channel
    """
    # Sequance 1
    OF_path = os.path.join(OFdir ,filename)
    OF = cv.imread(OF_path,-1)
    # BGR CHANNEL
    Fu = (OF[:,:,2].astype(np.float)-2**15) / 64.0
    Fv = (OF[:,:,1].astype(np.float)-2**15) / 64.0
    valid_OF = OF[:,:,0].astype(np.float)

    Fu = np.multiply(Fu,valid_OF)
    Fv = np.multiply(Fv,valid_OF)
    return Fu ,Fv, valid_OF

def writeOF(OFdir,filename,Fu,Fv,valid_OF=None):
    """
    Writing Optical flow files:
    As descriped in the Kitti Development kittols (BRG order)
    0 Dim validation
    1 Dim u
    2 Dim v
    """
    FLOW_SATURATION = 2**16-1

    if valid_OF == None:
        valid_OF = np.ones((np.shape(Fu)))

    Fu = np.expand_dims(Fu,-1)
    Fv = np.expand_dims(Fv,-1)
    valid_OF = np.expand_dims(valid_OF,-1)
    sat_map = np.full_like(Fu, FLOW_SATURATION)
    zero_map = np.full_like(Fu, 0)
    # Sequance 1
    OF_path = os.path.join(OFdir ,filename)

    su = np.expand_dims(np.amin(np.concatenate((Fu*64.0+2**15, sat_map), axis=2), axis=2),-1)
    nFu = np.expand_dims(np.amax(np.concatenate((su,zero_map),axis=2),axis=2),-1)

    sv = np.expand_dims(np.amin(np.concatenate((Fv*64.0+2**15, sat_map), axis=2), axis=2),-1)
    nFv = np.expand_dims(np.amax(np.concatenate((sv,zero_map),axis=2),axis=2),-1)

    OF = np.concatenate((valid_OF,nFu, nFv), axis=2)
    flag_save = cv.imwrite(OF_path, np.uint16(OF))




def plotOF(img1,img2, Fu, Fv, step = 20 , title_ = 'Test' ):
    """
    # TODO
    """
    imsize = np.shape(Fu)
    imsize = imsize +(3,)

    hsv = np.zeros(imsize, dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv.cartToPolar(Fu, Fv)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    rgb2 = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    #cv2.imshow("colored flow", rgb2)
    #mag, ang = cv.cartToPolar(Fu,Fv)
    Fu_dn = cv.resize(Fu, (0, 0), fx=1. / step, fy=1. / step)
    Fv_dn = cv.resize(Fv, (0, 0), fx=1. / step, fy=1. / step)
    Fu_dn = block_reduce(Fu, block_size=(step,step), func=np.mean)
    Fv_dn = block_reduce(Fv, block_size=(step,step), func=np.mean)
    x_pos = np.arange(0, imsize[1], step)
    y_pos = np.arange(0, imsize[0], step)
    X = np.meshgrid(x_pos)
    Y = np.meshgrid(y_pos)
    #imsize = np.shape(Fu)
    #X,Y = np.meshgrid(np.arange(0,imsize[1],step),np.arange(0,imsize[0],step))
    fig = plt.figure(1)
    ax1 = plt.subplot(311)
    ax3 = plt.subplot(312)
    ax2 = plt.subplot(313)
    ax3.imshow(rgb2,interpolation = 'gaussian')
    img1_ex = np.expand_dims(img1,-1)
    img2_ex = np.expand_dims(img2,-1)
    rgb = np.concatenate((img1_ex,np.zeros_like(img1_ex),img2_ex),axis=2 )
    rgb = np.concatenate((img1_ex,img2_ex,img2_ex),axis=2 )
    ax1.imshow(rgb)
    #ax1.imshow(img1,cmap ='Blues')
    #ax1.imshow(img2,cmap ='Reds',alpha=.6)

    ax2.imshow(img1,cmap='gray')
    ax3.set_title("Magnitude and angle")
    ax2.set_title("quiver")
    ax1.set_title(title_ + ", Overlapping images")
    M = np.hypot(Fu_dn,Fv_dn)
    #Q = ax2.quiver(X,Y,Fu_dn,Fv_dn,M,units='xy' ,alpha=0.6)
    Q = ax2.quiver(X,Y,Fu_dn,-Fv_dn,M,units='xy' ,alpha=0.6)
    plt.show()
    #plt.savefig('OF' + title_ + '.png')
    #cv.waitKey()


def MSEN_PEPN(Fu1, Fv1, valid1, Fu2, Fv2, valid2):
    """
            # TODO:
    """
    ERR_TH = 3

    err_map,n_total = compute_errmap(Fu1, Fv1, valid1, Fu2, Fv2, valid2)
    outliers = np.zeros_like(err_map)
    outliers[err_map>ERR_TH] = 1
    pepn = np.sum(outliers)/n_total
    msen = np.sum(err_map)/n_total
    #Pe = [x for x in err_map[:] if x>ERR_TH]

    return err_map,msen,pepn

def compute_errmap(Fu1, Fv1, valid1, Fu2, Fv2, valid2):
    # compute error map between the optical flow result and GT
    err_u = (Fu1 - Fu2) ** 2
    err_v = (Fv1 - Fv2) ** 2

    err_map = np.sqrt(err_u + err_v)
    err_map[valid1==0] = 0
    n_total = np.count_nonzero(valid1)
    return err_map,n_total

def mse_vectors(Fu1, Fv1, valid1, Fu2, Fv2, valid2):

    err_u = (Fu1 - Fu2) ** 2
    err_v = (Fv1 - Fv2) ** 2

    err_map = np.sqrt(err_u + err_v)
    err_map[valid1==0] = 0
    err_val = np.sum(err_map) / np.count_nonzero(valid1)

    return err_map, err_val


def OF_err_disp(errmap,valid_mask,seq_name='Seq'):

    n_bins = 50
    fig = plt.figure(1)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    pos = ax1.imshow(errmap, cmap='afmhot', interpolation='none')

    ax1.set_title(seq_name +' motion err map')
    fig.colorbar(pos, ax=ax1)

    # Histogram of the err map
    valid_err = errmap[valid_mask==1]
    bins_h = np.linspace(np.min(valid_err), np.max(valid_err), num=n_bins)

    ax2.hist(valid_err.ravel(), bins=bins_h,alpha=0.65) #,width=0.8
    ax2.set(xticks=bins_h)
    ax2.locator_params(axis='x', nbins=10)

    ax2.axvline(valid_err.mean(), color='k', linestyle='dashed', linewidth=2)
    _, max_ = plt.ylim()
    ax2.text(valid_err.mean() + valid_err.mean()/10, max_ - max_/10,'Mean: {:.2f}'.format(valid_err.mean()))

    ax2.set_title(seq_name +' motion err histogram')
    plt.show()
    #plt.savefig('OF' + seq_name + '.png')

def err_flow(Fu1, Fv1, valid1, Fu2, Fv2, valid2):
    # convert from cartesian to polar
    mag1, ang1 = cv.cartToPolar(Fu1, Fv1)
    mag2, ang2 = cv.cartToPolar(Fu2, Fv2)
    magDif = mag1-mag2
    angDif = ang1-ang2
    angDif = angle_wrap(angDif,radians=True)

    return magDif ,angDif


def angle_wrap(angle,radians=False):
    '''
    Wraps the input angle to 360.0 degrees.

    if radians is True: input is assumed to be in radians, output is also in
    radians

    '''
    PI = np.pi

    if radians:
        wrapped = angle % (2.0*PI)
        wrapped[wrapped>PI] -= 2.0*PI

    else:

        wrapped = angle % 360.0
        wrapped[wrapped>180.0] -= 360.0

    return wrapped


def of_metrics(gt_folder, sequence, result_folder, result_file):

    """
    Calculate MSEN, PEPN for calculated optical flow.
    :param: u1lk: x motion field
    :param: v1lk: y motion field
    :param: valid1lk: mask of valid pixels

    :return: MSEN: MSEN error value
    :return: PEPN: PEPN error value
    :return: MSEN: errmap1: Error map
    """

    #gt_folder = '/home/agus/repos/mcv-m6-2019-team1/data/kitti_optical_flow/gt'
    #seq1 = '000045_10.png'


    [u1gt,v1gt,valid1gt] = readOF(gt_folder,sequence)
    [u1lk, v1lk, valid1lk] = readOF(result_folder, result_file)

    f, c = np.shape(valid1lk)
    u1gt = u1gt[:f, :c]
    v1gt = v1gt[:f, :c]
    valid1gt = valid1gt[:f, :c]

    errmap1, mse1,pepe1 = MSEN_PEPN(u1gt,v1gt,valid1gt,u1lk,v1lk,valid1lk)

    return errmap1, mse1, pepe1


def save_motion_field(motion_field_x, motion_field_y, block_size, out_dir, res_filename, gt_folder, sequence):
    # Upsample motion field to fit image size

    motion_field_x_up1 = scipy.ndimage.zoom(motion_field_x, block_size, order=0)
    motion_field_y_up1 = scipy.ndimage.zoom(motion_field_y, block_size, order=0)

    [_, _, valid1gt] = readOF(gt_folder, sequence)

    fgt, cgt = np.shape(valid1gt)
    fx, cx = np.shape(motion_field_x_up1)
    fy, cy = np.shape(motion_field_y_up1)

    filx = fgt - fx
    fily = fgt - fy
    colx = cgt - cx
    coly = cgt - cy

    motion_field_x_up = np.pad(motion_field_x_up1,
                               ((int(np.floor(filx / 2.)), int(np.ceil(filx / 2.))),
                                (int(np.floor(colx / 2.)), int(np.ceil(colx / 2.)))),
                               mode='edge')
    motion_field_y_up = np.pad(motion_field_y_up1,
                               ((int(np.floor(fily / 2.)), int(np.ceil(fily / 2.))),
                                (int(np.floor(coly / 2.)), int(np.ceil(coly / 2.)))),
                               mode='edge')

    # Save motion field file:
    writeOF(out_dir + 'motion_fields', res_filename + 'mfield.png', motion_field_x_up, motion_field_y_up)


def of_block_matching(target_frame_path, anchor_frame_path, block_size, search_range, save=False):

        res_filename = 'b' + str(block_size) + 's' + str(search_range)

        target_frm = cv.imread(target_frame_path)
        anchor_frm = cv.imread(anchor_frame_path)

        target_frm = cv.cvtColor(target_frm, cv.COLOR_BGR2GRAY)
        anchor_frm = cv.cvtColor(anchor_frm, cv.COLOR_BGR2GRAY)

        ebma = EBMA_searcher(N=block_size,
                             R=search_range,
                             p=norm,
                             acc=pixel_acc)

        predicted_frm, motion_field = \
            ebma.run(anchor_frame=anchor_frm,
                     target_frame=target_frm)

        motion_field_x = motion_field[:, :, 0]
        motion_field_y = motion_field[:, :, 1]

        error_image = abs(np.array(predicted_frm, dtype=float) - np.array(anchor_frm, dtype=float))
        error_image = np.array(error_image, dtype=np.uint8)

        # Peak Signal-to-Noise Ratio of the predicted image
        mse = (np.array(error_image, dtype=float) ** 2).mean()
        psnr = 10 * math.log10((255 ** 2) / mse)

        if save:

            # store frames in PNG for our records
            os.system('mkdir -p frames_of_interest')
            imsave('frames_of_interest/' + res_filename + 'target.png', target_frm)
            imsave('frames_of_interest/' + res_filename + 'anchor.png', anchor_frm)

            # store frames in PNG for our records
            os.system('mkdir -p frames_of_interest')
            imsave('frames_of_interest/' + res_filename + 'target.png', target_frm)
            imsave('frames_of_interest/' + res_filename + 'anchor.png', anchor_frm)

            # store predicted frame
            imsave('frames_of_interest/' + res_filename + 'predicted_anchor.png', predicted_frm)

            # store error image
            imsave('frames_of_interest/' + res_filename + 'error_image_shelf.png', error_image)

        
        return motion_field_x, motion_field_y