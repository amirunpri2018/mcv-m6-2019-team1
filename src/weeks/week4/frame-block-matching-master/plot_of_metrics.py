# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.interpolate import interp2d
from matplotlib import cm

# Load metrics data:

with open('of_box_match1.pkl', 'rb') as f:
    data = pickle.load(f)

block_size = data['block_size'][0]
search_range = data['search_range'][0]
msen = data['msen'][0]
pepn = data['pepn'][0]
time = data['time'][0]


df_metrics = pd.DataFrame(
    {'block_size': block_size,
     'search_range': search_range,
     'msen': msen,
     'pepn': pepn,
     'time': time
    })

"""
df_block = df_metrics.groupby(block_size)
plt.figure()
for block, values in df_block:
    plt.plot(values['search_range'], values['time'], 'o-', label = 'Block size = ' + str(block))
plt.legend()
plt.xlabel('Search Range (px)')
plt.ylabel('Time (s)')
plt.title('Time vs. Search range')
plt.savefig('time_vs_search.png')
plt.show()

plt.figure()
for block, values in df_block:
    plt.plot(values['msen'], values['time'], 'o-', label = 'Block size = ' + str(block))
plt.legend()
plt.xlabel('MSEN')
plt.ylabel('Time (s)')
plt.title('Time vs. MSEN')
plt.savefig('time_vs_msen_perblock.png')
plt.show()

df_search = df_metrics.groupby(search_range)
plt.figure()
for search, values in df_search:
    plt.plot(values['msen'], values['time'], 'o-', label = 'Search range = ' + str(search))
plt.legend()
plt.xlabel('MSEN')
plt.ylabel('Time (s)')
plt.title('Time vs. MSEN')
plt.savefig('time_vs_msen_persearch.png')
plt.show()
"""

df_search = df_metrics.groupby(search_range)
plt.figure()
for search, values in df_search:
    plt.plot(values['block_size'], values['time'], 'o-', label = 'Search range = ' + str(search))
plt.legend()
plt.xlabel('Block Size')
plt.ylabel('Time (s)')
plt.title('Time vs. Block size')
plt.savefig('time_vs_block')
plt.show()


df_search = df_metrics.groupby(search_range)
plt.figure()
for search, values in df_search:
    plt.plot(values['block_size'], values['msen'], 'o-', label = 'Search range = ' + str(search))
plt.legend()
plt.xlabel('Block size (px)')
plt.ylabel('MSEN')
plt.title('MSEN vs Block size (px)')
plt.savefig('msen_vs_block.png')
plt.show()


df_search = df_metrics.groupby(search_range)
plt.figure()
for search, values in df_search:
    plt.plot(values['block_size'], values['pepn'], 'o-', label = 'Search range = ' + str(search))
plt.legend()
plt.xlabel('Block size (px)')
plt.ylabel('PEPN')
plt.title('PEPN vs Block size (px)')
plt.savefig('pepn_vs_block.png')
plt.show()


plt.figure()
for search, values in df_search:
    plt.plot(values['time'][:-1], values['msen'][:-1], 'o-', label = 'Search range = ' + str(search))
    break
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('MSEN')
plt.title('MSEN vs Time (s)')
plt.savefig('msen_vs_time_s4.png')
plt.show()



"""
# Create meshgrid for surface plot:
x = np.arange(0, 70)
y = np.arange(0, 70)
X, Y = np.meshgrid(x, y)

# Interpolate for surface plot:
msen_interp_f = interp2d(block_size, search_range, msen, kind='linear')
msen_val = msen_interp_f(x, y)
msen_clip = np.clip(msen_val, np.min(msen), None)

pepn_interp_f = interp2d(block_size, search_range, pepn, kind='linear')
pepn_val = pepn_interp_f(x, y)


# Surface plot:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, msen_clip, cmap=cm.coolwarm)
###############
#m = cm.ScalarMappable(cmap=cm.coolwarm)
#m.set_array(msen_clip)
#plt.colorbar(m)
###############
ax.set_xlabel('Block size', fontsize=30)
ax.set_ylabel('Search Range', fontsize=30)
ax.set_zlabel('MSEN', fontsize=30)
ax.yaxis._axinfo['label']['space_factor'] = 3.0
# change fontsize
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(25)
for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(25)
for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(25)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, pepn_val, cmap=cm.coolwarm)
ax.set_xlabel('Block size', fontsize=30, linespacing=3.4)
ax.set_ylabel('Search Range', fontsize=30, linespacing=3.4)
ax.set_zlabel('PEPN', fontsize=30, linespacing=3.4)
# change fontsize
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(25)
for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(25)
for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(25)
#m = cm.ScalarMappable(cmap=cm.coolwarm)
#m.set_array(pepn_val)
#plt.colorbar(m)
plt.show()
"""

print('agus capa')