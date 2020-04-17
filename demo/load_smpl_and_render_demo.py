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
'''
Load shape and pose parameters of SMPL+H and pass them to model.
Render an image using opendr.renderer
'''
#Todo: pip install opendr, pip install transforms3d,  pip install mathutils

import matplotlib.pyplot as plt
import numpy as np
import os
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
import chumpy as ch
import cv2
import camera_helper_functions as chf
import math
import imp
import transforms3d #Todo: additional dependency! not listed in readme
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./train/1/')
parser.add_argument('--smplH_modelFolder', default='../datageneration/smpl_data/mano_v1_2/models')
parser.add_argument('--fileNr', default=3, type=int)
parser.add_argument('--humanNr', default=0, type=int)
parser.add_argument('--scatter', default=False, type=bool)
args = parser.parse_args()

lm = imp.load_source(
    'load_model',
    '../datageneration/smpl_data/mano_v1_2/webuser/smpl_handpca_wrapper_multiHuman.py')

######################### helper function ######################################
def rotAroundXXX(radIN):
    return np.array([[1,  0,              0            ],
                     [0,  np.cos(radIN), -np.sin(radIN)],
                     [0,  np.sin(radIN),  np.cos(radIN)]])

def rotateBody(RzBody, pelvisRotVec):
    angle = np.linalg.norm(pelvisRotVec)
    Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
    globRotMat = np.dot(RzBody, Rpelvis)
    R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
    globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(np.dot(R90, globRotMat))
    globRotVec = globRotAx * globRotAngle
    return globRotVec

def x_rot_matrix(xrot):
    rotMat = np.array(((1,  0,                          0),
                       (0,  math.cos(xrot), -math.sin(xrot)),
                       (0, math.sin(xrot), math.cos(xrot))))
    return rotMat

def y_rot_matrix(yrot):
    rotMat = np.array(((math.cos(yrot), 0, math.sin(yrot)),
                       (0,              1,               0),
                       (-math.sin(yrot), 0, math.cos(yrot))))
    return rotMat

def z_rot_matrix(zrot):
    rotMat = np.array(((math.cos(zrot), math.sin(zrot), 0),
                       (-math.sin(zrot),  math.cos(zrot), 0),
                       (0,               0,              1)))
    return rotMat


######################### parse data ##########################################
path= args.path
#you need to download the SMPL+H model for this code to work
modelpath = args.smplH_modelFolder
#select a file nr
fileNr = args.fileNr
humanNr = args.humanNr
toOpenGL = not(args.scatter)

#names of
imgdir = 'composition/'
posedir_3d = 'pose_joints_3d/'
cameradir = 'camera_RT_4x4/'
posedir_2d = 'pose_joints_2d/'
posedir = 'pose_coeffs_fullPCA/'
shapedir = 'shapes/'
gender = 'gender/'
translation_folder = 'subj_pelvis_T'
rotation_folder = 'subj_ZRot'
obj_id_folder = 'objectId_obname'

if toOpenGL:
    opencv2opengl = np.array([1, -1, -1])
else:
    opencv2opengl = np.array([1,1,1])

# get files and sort
camera_files        = os.listdir(os.path.join(path, cameradir))
pose_files          = os.listdir(os.path.join(path, posedir))
shape_files         = os.listdir(os.path.join(path, shapedir))
gender_files        = os.listdir(os.path.join(path, gender))
img_files           = os.listdir(os.path.join(path,imgdir))
translation_files   = os.listdir(os.path.join(path, translation_folder))
rot_files           = os.listdir(os.path.join(path, rotation_folder))
pose3d_files        = os.listdir(os.path.join(path, posedir_3d))
objId_files        = os.listdir(os.path.join(path, obj_id_folder))
pose3d_files = [pose3d_f for pose3d_f in pose3d_files if not('coco' in pose3d_f)]


camera_files.sort()
pose_files.sort()
shape_files.sort()
gender_files.sort()
img_files.sort()
translation_files.sort()
rot_files.sort()
pose3d_files.sort()
objId_files.sort()

# get camera params (not saved, set in blender)
img = cv2.imread(os.path.join(path, imgdir, img_files[fileNr]))
if not(toOpenGL):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, _ = img.shape

res = [width, height]
cam_lens = 60
cam_sensor = 32


###################### load data ###################################
pose = np.load(os.path.join(path, posedir, pose_files[fileNr]),
               allow_pickle=True)
pose = pose.item()
pose_limitsOrTransitions = np.copy(pose)

sHumanNames = list(pose.keys())

pose = pose[sHumanNames[humanNr]]
pose = pose[:]
pose_limitsOrTransitions = np.copy(pose)

shape_coef = np.load(os.path.join(path, shapedir, shape_files[fileNr]),
                     allow_pickle=True)
shape_coef = shape_coef.item()
shape_coef = shape_coef[sHumanNames[humanNr]]

gender = np.load(os.path.join(path, gender, gender_files[fileNr]),
                 allow_pickle=True)
gender = gender.item()
gender = gender[sHumanNames[humanNr]]

translation = np.load(os.path.join(path,
                                   translation_folder,
                                   translation_files[fileNr]),
                      allow_pickle=True)
translation = translation.item()
translation = translation[sHumanNames[humanNr]]

pose3d = np.load(os.path.join(path, posedir_3d, pose3d_files[fileNr]),
                 allow_pickle=True)
pose3d = pose3d.item()
pose3d = pose3d[sHumanNames[humanNr]]

rotation = np.load(os.path.join(path, rotation_folder, rot_files[fileNr]),
                   allow_pickle=True)
rotation = rotation.item()
rotation = rotation[sHumanNames[humanNr]]

camera_info = np.load(os.path.join(path, cameradir, camera_files[fileNr]),
                      allow_pickle=True)

# rotate if necessary
#note that two mocap datasets used for LTSH have different inital rotation.
#Since we do not save which mocap data is used we need to try both rotations
#and pick the better one
if not(toOpenGL):
    rotation = -rotation
    # only for limits and transitions. They have a different inital rotation,
    #which is not saved, so we need to figure that out. (only used for LTSH)
    rotMat_limits = x_rot_matrix(2*np.pi)
    try:
        pose_limitsOrTransitions[0:3] = rotateBody(rotMat_limits,
                                                   pose_limitsOrTransitions[0:3])
    except:
        # if pose is all zeros we need to add a small offset
        pose_limitsOrTransitions[0:3] = rotateBody(rotMat_limits,
                                                   pose_limitsOrTransitions[0:3]+1e-12)
else:
    # for opengl we need to rotate the SMPL model to make the projection correct
    rotMat = x_rot_matrix(0.5*np.pi)
    try:
        pose[0:3] = rotateBody(rotMat, pose[0:3])
        # only for limits and transitions. They have a different inital rotation,
        #which is not saved, so we need to figure that out. (only used for LTSH)
        rotMat_limits = x_rot_matrix(1*np.pi)
        pose_limitsOrTransitions[0:3] = rotateBody(rotMat_limits,
                                                   pose_limitsOrTransitions[0:3])
    except:
       pose[0:3] = rotateBody(rotMat, pose[0:3]+1e-12)
       # only for limits and transitions. They have a different inital rotation,
       #which is not saved, so we need to figure that out. (only used for LTSH)
       rotMat_limits = x_rot_matrix(1*np.pi)
       pose_limitsOrTransitions[0:3] = rotateBody(rotMat_limits,
                                                  pose_limitsOrTransitions[0:3]+1e-12)

rotMat = z_rot_matrix(rotation)

try:
    pose[0:3] = rotateBody(rotMat, pose[0:3])
    pose_limitsOrTransitions[0:3] = rotateBody(rotMat,
                                               pose_limitsOrTransitions[0:3])
except:
    pose[0:3] = rotateBody(rotMat, pose[0:3]+1e-12)
    pose_limitsOrTransitions[0:3] = rotateBody(rotMat,
                                               pose_limitsOrTransitions[0:3]+1e-12)


##############################################################################
# initialize two models and compute error to 3D joints to find out the inital rotation
model = lm.load_model(os.path.join(modelpath, 'SMPLH_'+gender+ '.pkl'),
                      mano_path=modelpath)
model_limitsOrTransitions = lm.load_model(os.path.join(modelpath,
                                                      'SMPLH_'+gender+ '.pkl'),
                                          mano_path=modelpath)

model.pose[:]  = pose
model.betas[:] = shape_coef
model.trans[:] = translation * opencv2opengl - model.J_transformed.r[0]

model_limitsOrTransitions.pose[:]  = pose_limitsOrTransitions
model_limitsOrTransitions.betas[:] = shape_coef
model_limitsOrTransitions.trans[:] = translation * opencv2opengl - model_limitsOrTransitions.J_transformed.r[0]

errors = [np.mean(np.abs(model.J_transformed.r - pose3d[:52,:])),
          np.mean(np.abs(model_limitsOrTransitions.J_transformed.r - pose3d[:52,:]))]

modelIdx = np.argmin(errors)
model = [model, model_limitsOrTransitions][modelIdx]


######################## rendering ############################################
rn = ColoredRenderer()
w, h = width, height
RT = np.array(chf.get_3x4_RT_matrix_from_blender(camera_info))
KKK = chf.cam_compute_intrinsic(res)

TTT = RT[:3, 3]
RRR = RT[:3, :3]
RRR = RRR * opencv2opengl

fff = model.f.copy()
vvv = model.r.copy()

rn.camera = ProjectPoints(f=ch.array([KKK[0, 0], KKK[1, 1]]),
                          rt=cv2.Rodrigues(RRR)[0].flatten(),
                          t=ch.array(TTT),
                          k=ch.array([0, 0, 0, 0]),
                          c=ch.array([KKK[0, 2], KKK[1, 2]]))
rn.frustum = {'near': 0.1, 'far': 15., 'width': w, 'height': h}
rn.set(v=vvv, f=fff, bgcolor=ch.zeros(3))
rn.background_image = img / 255. if img.max() > 1 else img

# Construct point light source
rn.vc = LambertianPointLight(
    f=model.f,
    v=rn.v,
    num_verts=len(model),
    light_pos=ch.array([0, 0, 0]),
    vc=np.ones_like(model)*.5,
    light_color=ch.array([-100., -100., -100.]))

if toOpenGL:
    cv2.imshow('render_SMPL', rn.r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    smpl_vertices = model.r
    proj_smpl_vertices = chf.project_vertices(smpl_vertices, KKK, RT)
    proj_smpl_joints = chf.project_vertices(model.J_transformed.r , KKK, RT)
    proj_joints3d_vertices = chf.project_vertices(pose3d, KKK, RT)

    plt.imshow(img)
    # plt.scatter(proj_smpl_vertices[:, 0], proj_smpl_vertices[:, 1], 1)
    plt.scatter(proj_joints3d_vertices[:, 0], proj_joints3d_vertices[:, 1], 1)
    plt.scatter(proj_smpl_joints[:, 0], proj_smpl_joints[:, 1], 1)
    plt.show()
