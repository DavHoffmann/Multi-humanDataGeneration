# -*- coding: utf-8 -*-
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import matplotlib.cm as cm
import cv2
import camera_helper_functions as chf
import argparse


###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--inBasePath', default='./train/1/')
parser.add_argument('--fileNr', default=3, type=int)
parser.add_argument('--humanNr', default=0, type=int)

args = parser.parse_args()
inBasePath = args.inBasePath
humanNr = args.humanNr
imgNr = args.fileNr
imgNr = '%05d' %imgNr

camIn = os.path.join(inBasePath, 'camera_RT_4x4', imgNr + '.npy')
joint3D = os.path.join(inBasePath, 'pose_joints_3d', imgNr + '.npy')
joint2D = os.path.join(inBasePath, 'pose_joints_2d', imgNr + '.npy')
imgPath =  os.path.join(inBasePath, 'composition', imgNr + '.png')

cam = np.load(camIn, allow_pickle=True)
joints3D = np.load(joint3D, allow_pickle=True).item()
joints2D = np.load(joint2D, allow_pickle=True).item()
img = cv2.imread(imgPath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
sHumanNames = joints2D.keys()


############### 3d joints in camera coordinate system ##########################
# same scatter plot but this time transform to camera coordinates and opencv
# coordinate frame
height, width, _ = img.shape
K = chf.cam_compute_intrinsic([width, height])

joints3D_camCoord = {}
joints_projected = {}
for i, key in enumerate( sHumanNames ):
    gt3d_xyz, gt_points_projected = chf.point_in_camera_coords( cam,
                                                                joints3D[key],
                                                                K )
    joints3D_camCoord[key] = gt3d_xyz
    joints_projected[key] = gt_points_projected

camlocArr = np.array([cam[0,3], cam[1,3], cam[2,3]])
camlocArr = np.reshape(camlocArr, (1,3))
cam_inCamCoords,_ = chf.point_in_camera_coords(cam, camlocArr, K)




key = list(sHumanNames)[humanNr]
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.imshow(img)
for j in range(joints_projected[key].shape[0]):
#         ax2.scatter(joints_projected[key][j,0], joints_projected[key][j,1])
    ax2.scatter(joints_projected[key][j,0], joints_projected[key][j,1])
    ax2.annotate(j, (joints_projected[key][j,0], joints_projected[key][j,1]))
fig.suptitle('3D projected')
plt.show()

key = list(sHumanNames)[humanNr]
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.imshow(img)
for j in range(joints2D[key].shape[0]):
#         ax2.scatter(joints_projected[key][j,0], joints_projected[key][j,1])
    ax2.scatter(joints2D[key][j,0], joints2D[key][j,1])
    ax2.annotate(j, (joints2D[key][j,0], joints2D[key][j,1]))
fig.suptitle('2D')


plt.show()
