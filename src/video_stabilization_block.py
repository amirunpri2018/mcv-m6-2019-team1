# Import numpy and OpenCV
import numpy as np
import cv2 as cv
from utils.ebma_searcher import EBMA_searcher
import time
import os
DATA_DIR = '../data'
OUT_DIR = '../output/week4/task2'
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def OFToRot(Du,Dv):
    s = np.shape(Du)
    #
    x1,y1 = np.meshgrid(range(s[1]),range(s[0]))
    x2 = x1+Du
    y2 = y1+Dv
    mag1, ang1 = cv.cartToPolar(x1,y1)
    mag2, ang2 = cv.cartToPolar(x2,y2)

    ang = ang2-ang1
    return ang

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv.warpAffine(frame, T, (s[1], s[0]))
    return frame

def im_rescale(img):

    width = int(img.shape[1] * 0.5)
    height = int(img.shape[0] * 0.5)
    dim = (width, height)
    # resize image
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)


block_size = 16
search_range = 30
norm = 1
pixel_acc = 1

# The larger the more stable the video, but less reactive to sudden panning
SMOOTHING_RADIUS = 50

# Read input video
cap = cv.VideoCapture(os.path.join(DATA_DIR,'video.mp4'))

# Get frame count

#n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
n_frames = int(cap.get(cv.cv.CV_CAP_PROP_FRAME_COUNT))

# Get width and height of video stream
#w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
#h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv.cv.CV_CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.cv.CV_CAP_PROP_FRAME_HEIGHT))

# Get frames per second (fps)
#fps = cap.get(cv.CAP_PROP_FPS)
fps = cap.get(cv.cv.CV_CAP_PROP_FPS)

# Define the codec for output video
#fourcc = cv.VideoWriter_fourcc(*'MJPG')
fourcc = cv.cv.CV_FOURCC(*'XVID')

# Set up output video
#out = cv.VideoWriter(os.path.join(DATA_DIR,'video_out.avi'), fourcc, fps, (2 * w, h))

# Read first frame
_, prev = cap.read()
prev = im_rescale(prev)
# Pre-define transformation-store array
transforms = np.zeros((n_frames - 1, 3), np.float32)

print('Start iterating over frames...')

#seq_stabilized = np.zeros((h,w,3,n_frames-1))
prev_grey = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
#seq_stabilized[:,:,:,0] = prev

N = 100
for i in range(1,N):   #(n_frames - 2):
    print('{}/{}'.format(i,N-1))
    # Read next frame
    success, curr = cap.read()
    curr = im_rescale(curr)
    #if i<33:
    #    continue

    if not success:
        break
    #target_frm = curr
    #anchor_frm = prev

    #anchor_frm_grey = prev_grey
    prev_grey = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    curr_grey = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)

    ebma = EBMA_searcher(N=block_size,
                         R=search_range,
                         p=norm,
                         acc=pixel_acc)

    predicted_frm, motion_field = \
        ebma.run(anchor_frame=prev_grey,target_frame=curr_grey)

    motion_field_x = motion_field[:, :, 0]
    motion_field_y = motion_field[:, :, 1]

    #mag, ang = cv.cartToPolar(motion_field_x, motion_field_y)
    #uniques, counts = np.unique(mag, return_counts=True)
    #mc_mag = uniques[counts.argmax()]

    #uniques, counts = np.unique(ang, return_counts=True)
    #mc_ang = uniques[counts.argmax()]

    #u, v = pol2cart(mc_mag, mc_ang)
    u = np.median(motion_field_x)
    v = np.median(motion_field_y)
    #ang = np.median(OFToRot(motion_field_x,motion_field_y))
    ang = 0
    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(ang)
    m[0, 1] = -np.sin(ang)
    m[1, 0] = np.sin(ang)
    m[1, 1] = np.cos(ang)
    m[0, 2] = -u
    m[1, 2] = -v
    affine_H = m
    #affine_H = np.float32([[1, 0, -v], [0, 1, -u]])

    next_stabilized = cv.warpAffine(curr, affine_H, (curr.shape[1], curr.shape[0]))
    frame_out = cv.hconcat([curr, next_stabilized])
    prev = next_stabilized

    #seq_stabilized[:, :, :, i] = next_stabilized
    #cv.imwrite(os.path.join(OUT_DIR,'original_frames/' + str(i) + '_frame.png'), curr)
    cv.imwrite(os.path.join(OUT_DIR,'video_frames/' + str(i) + '_frame.png'), frame_out)


"""
################################

# Extract rotation angle
#_, da_tot = cv.cartToPolar(motion_field_x, motion_field_y)
#da = np.mean(np.mean(da_tot))

dx = np.mean(np.mean(motion_field_x))
dy = np.mean(np.mean(motion_field_y))
#da = np.arctan2(dy, dx)
da = 0
# Store transformation
transforms[i] = [dx, dy, da]

# Move to next frame
prev = curr

print("Frame: " + str(i) + "/" + str(n_frames))
"""
"""
# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

# Create variable to store smoothed trajectory
smoothed_trajectory = smooth(trajectory)

# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory

# Calculate newer transformation array
transforms_smooth = transforms + difference

# Reset stream to first frame
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Write n_frames-1 transformed frames
for i in range(n_frames - 2):
    # Read next frame
    success, frame = cap.read()
    if not success:
        break

    # Extract transformations from the new transformation array
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    # Apply affine wrapping to the given frame
    frame_stabilized = cv.warpAffine(frame, m, (w, h))

    # Fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized)

    # Write the frame to the file
    frame_out = cv.hconcat([frame, frame_stabilized])

    # If the image is too big, resize it.
    if (frame_out.shape[1] > 1920):
        frame_out = cv.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2));

    cv.imshow("Before and After", frame_out)
    cv.waitKey(10)
    out.write(frame_out)

# Release video
cap.release()
out.release()
# Close windows
cv.destroyAllWindows()
"""
