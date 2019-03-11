import numpy as np
import utils as ut
# Read frames from video

import cv2 as cv

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.animation as manimation
# For visulization
import matplotlib.pyplot as plt

# fig = plt.figure()
# l, = plt.plot([], [], 'k-o')
#
#
# with writer.saving(fig, "writer_test.mp4", 100):
#     for i in range(100):
#         x0 += 0.1 * np.random.randn()
#         y0 += 0.1 * np.random.randn()
#         l.set_data(x0, y0)
#         writer.grab_frame()



def getGauss_bg2(file_list, D=1 , gt_file = None, output_dir = None):
    """
    Computes Gaussian Background model.

    :file_list: list of all images used to estimate the bg model
    :D: False positives
    : git_file , ignoring bbox of foreground when creating the bg model
    :return: mu_bg -  mean, var_bg
    """


# Create aa video of constructing the background MODEL


    if D==1:
        Clr_flag = cv.IMREAD_GRAYSCALE
    else :
        Clr_flag = cv.IMREAD_COLOR

    # if there is bbox to ignore fron
    if gt_file:
        Bbox = ut.get_bboxes_from_MOTChallenge(gt_file)
    # Count the number of Images

    # get image size
    N = len(file_list)

    s = np.shape(cv.imread(file_list[0],Clr_flag))
    m0 = np.zeros((s[0],s[1]),dtype=bool)
    # initializing the cumalitive frame matrix
    if D==1:
        A = np.zeros((s[0],s[1]))
        ma = np.full((s[0],s[1]), float(N))
    #fig = plt.figure(1)

    if output_dir:
        #FFMpegWriter = manimation.writers['ffmpeg']
        #writer = FFMpegWriter(fps=20, metadata=dict(artist='Team1',title='BG model'))

        fig_ani = plt.figure(2)
        ax1_ani = plt.subplot(221) # mu
        ax2_ani = plt.subplot(222) # std
        ax3_ani = plt.subplot(223) # mask
        ax4_ani = plt.subplot(224) # mask
        ax1_ani.imshow(A,cmap='gray')
        ax2_ani.imshow(A,cmap='gray')
        ax3_ani.imshow(ma,cmap='gray',vmin=0,vmax=N)
        ax4_ani.imshow(m0,cmap='gray')
        #writer.saving(fig_ani, vid_file, dpi=100)
    # I. Loop to obtain mean


    #for j in range(n):
        #update_figure(n)         #moviewriter.grab_frame()
    for i,image_path in enumerate(file_list, start=0):
        if gt_file:
            frm = ut.frameIdfrom_filename(image_path)
            m0 = ut.getbboxmask(Bbox,frm,(s[0],s[1]))

        #Upload frame
        if D==1:
            # Read the frame from the video and NOT the images
            I= cv.imread(image_path,Clr_flag)
            if output_dir:
                ax2_ani.cla()
                ax4_ani.cla()
                ax2_ani.imshow(I,cmap='gray')
                ax4_ani.imshow(m0,cmap='gray',vmin=0,vmax=1)
                ax4_ani.set_title("Current mask")
                ax2_ani.set_title("frame {}".format(frm))
                plt.pause(0.01)

            np.place(I, m0, 0.0)
            ma -= m0
            # Adding frames values
            A+= I
            if output_dir:
                ax1_ani.cla()
                ax3_ani.cla()
                ax1_ani.imshow(A,cmap='gray')
                ax3_ani.imshow(ma,cmap='gray',vmin=0,vmax=N)

                ax1_ani.set_title("Cumulative frames")
                fig_ani.savefig(output_dir+'frm'+str(frm)+'.png')

        #writer.grab_frame()
                            #ax1_ani.cla()
    # Do ma need to be float??
    mu_bg = A/ma
    if output_dir:
        ax1_ani.cla()
        ax1_ani.imshow(mu_bg,cmap='gray')
        ax1_ani.set_title("Mean BG, over {} frames".format(frm))

    if D==1:
        A = np.zeros((s[0],s[1]))
    # II. Loop to obtain std
    # sqrt (1/N * sum((x-mu)^2))
    for i,image_path in enumerate(file_list, start=0):
        print i
        if gt_file:
            frm = ut.frameIdfrom_filename(image_path)
            ma0 = ut.getbboxmask(Bbox,frm,(s[0],s[1]))

        #Upload frame
        if D==1:
            I = (cv.imread(image_path,Clr_flag)-mu_bg)**2
            if output_dir:
                ax2_ani.cla()
                ax2_ani.imshow(I,cmap='gray')
                ax2_ani.set_title("frame #{}".format(frm))
                ax4_ani.cla()
                ax4_ani.imshow(m0,cmap='gray',vmin=0,vmax=1)
                ax4_ani.set_title("Current mask")
            np.place(I, m0, 0.0)
            A+= I
            if output_dir:
                ax3_ani.cla()
                ax3_ani.imshow(A,cmap='gray')
                ax3_ani.set_title("Cumulative STD")
                fig_ani.savefig(output_dir+'frm'+str(frm)+'.png')

    std_bg = np.sqrt(A/ma)
    #ani.save('HeroinOverdosesJumpy.mp4', writer=writer)
    if output_dir:
        ax3_ani.cla()
        ax3_ani.imshow(std_bg,cmap='gray')
        ax3_ani.set_title("Std BG, over {} frames".format(frm))
        fig_ani.savefig(output_dir+'final.png')
        plt.close(fig_ani)
    return mu_bg,std_bg



def foreground_from_GBGmodel(bg_mu,bg_std,I,th =2):
    """
    Apply to any N Gaussian model, (as long the dimensions of I and bg agrees)
    output : Boolien map
        0 Background
        1 Foreground

        For now 8/3 only works for 1D (grayscale)
    """
    s = np.shape(I)
    foreground_map = np.zeros((s[0],s[1]), dtype=bool )

    # centered Image with repect to mu of the Background
    Ic = np.abs(I-bg_mu)
    foreground_map[Ic>=th*(bg_std+2)] = True
    return foreground_map
