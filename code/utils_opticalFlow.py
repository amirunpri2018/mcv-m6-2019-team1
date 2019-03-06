import cv2 as cv
from PIL import Image
import os
import numpy as np
import matplotlib

#matplotlib.use('Agg')
matplotlib.use('TkAgg')

# For visulization
import matplotlib.pyplot as plt
from matplotlib import gridspec
#import matplotlib.ticker as ticker
# Our LIBS
import utils as utils



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

def writeOF(OFdir,filename,Fu,Fv,valid_OF):
    """
    Writing Optical flow files:
    As descriped in the Kitti Development kittols (BRG order)
    0 Dim validation
    1 Dim u
    2 Dim v
    """
    FLOW_SATURATION = 2**16-1

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
    #mag, ang = cv.cartToPolar(Fu,Fv)
    Fu_dn = cv.resize(Fu, (0, 0), fx=1. / step, fy=1. / step)
    Fv_dn = cv.resize(Fv, (0, 0), fx=1. / step, fy=1. / step)

    imsize = np.shape(Fu)
    X,Y = np.meshgrid(np.arange(0,imsize[1],step),np.arange(0,imsize[0],step))
    fig = plt.figure(1)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    img1_ex = np.expand_dims(img1,-1)
    img2_ex = np.expand_dims(img2,-1)
    rgb = np.concatenate((img1_ex,np.zeros_like(img1_ex),img2_ex),axis=2 )
    rgb = np.concatenate((img1_ex,img2_ex,img2_ex),axis=2 )
    ax1.imshow(rgb)
    #ax1.imshow(img1,cmap ='Blues')
    #ax1.imshow(img2,cmap ='Reds',alpha=.6)

    ax2.imshow(img1,cmap='gray')
    ax1.set_title(title_)
    M = np.hypot(Fu_dn,Fv_dn)
    #Q = ax2.quiver(X,Y,Fu_dn,Fv_dn,M,units='xy' ,alpha=0.6)
    Q = ax2.quiver(X,Y,Fu_dn,Fv_dn,M,units='xy' ,alpha=0.6)
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
