import cv2 as cv
from PIL import Image
import os
import numpy as np

def readOF(OFdir,filename):
    """
    Reading Optical flow files:
    As descriped in the Kitti Deelopment kittols
    0 Dim validation
    1 Dim u
    2 Dim v
    """
    # Sequance 1
    OF_path = os.path.join(OFdir ,filename)
    OF = cv.imread(OF_path,-1)
    print(np.shape(OF))
    print(type(OF[2][2][1]))
    Fu = (OF[:,:,1].astype(np.float)-2**15) / 64.0
    Fv = (OF[:,:,2].astype(np.float)-2**15) / 64.0
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
    print(OF_path)

    su = np.expand_dims(np.amin(np.concatenate((Fu*64.0+2**15, sat_map), axis=2), axis=2),-1)
    nFu = np.expand_dims(np.amax(np.concatenate((su,zero_map),axis=2),axis=2),-1)

    sv = np.expand_dims(np.amin(np.concatenate((Fv*64.0+2**15, sat_map), axis=2), axis=2),-1)
    nFv = np.expand_dims(np.amax(np.concatenate((sv,zero_map),axis=2),axis=2),-1)

    OF = np.concatenate((valid_OF,nFu, nFv), axis=2)
    flag_save = cv.imwrite(OF_path, np.uint16(OF))




    def plotOF(img, Fu, Fv):
        """
        # TODO
        """
        mag, ang = cv.cartToPolar(Fu,Fv)



    def MSEN_PEPN(OF1,OF2):
        """
            # TODO:
            """
        return msen,pepn
