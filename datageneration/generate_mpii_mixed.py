
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

#../blender-2.79b-linux-glibc219-x86_64/blender -b -P generate_mpii_mixed.py -- 0 0 4
# parts of this script are deprecated and solved better in Generate_multiHumanFlow, for example placement in visible area
# I suggest to replace the respective parts with code from Generate_multiHumanFlow
# the original code for Learning to train with Synthetic Humans is left for reproducibility

import sys
print(sys.version)
import bpy
from bpy_extras.object_utils import world_to_camera_view as world2cam
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler
import math

from os.path import join, dirname, realpath, exists, basename
from glob import glob
from random import choice
from os import remove
import cv2

from scipy.io import loadmat, savemat
import json
import importlib.util

from itertools import combinations
from scipy.special import binom
from scipy import array as array
import pickle

import bounding_box_collision as collisionDetector
import ray_back_projection_visibleArea as ray_back_projection

###################################
# Todo: set output directory
PATH_out = '' #outout folder
PATH_tmp = ''  # tmp-output folder

mocapDataName = 'mocapAll'
dataset_path_prefix = 'MPI_occlusionng_bg_test2'
###################################


#Todo: set path to MPII images
img_base_path = './MPII_images' #Todo:download images from http://human-pose.mpi-inf.mpg.de/#download
img_base_path = ''

import logging
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
###################################
###################################
###################################
DBG_MODE_ENFORCE_FLAT_HAND = False
###################################
###################################
###################################
DBG_MODE_ENFORCE_POSE_ZERO = False
###################################
###################################
###################################
UPPER_HEAD = True
#what images to save
save_dbug_Imgs = True
save_flow = False

res = [1920,1280]
cam_lens = 60
cam_sensor = 32
nr_of_frames_per_bg = 50
FLAG_flip_BG = True
MODE_NODES = 'OLD'        # create_textured_plane
#####################################################
#####################################################    POSITION
RANDOM_POSITION_perCHUNCK = True
#####################################################
#####################################################    BACKGROUND
RANDOM_BG = True
RANDOM_BG_per_CHUNK = True
#####################################################
#####################################################    TEXTURE
RANDOM_TXT = True
RANDOM_TXT_per_CHUNK = True
#####################################################
#####################################################    SHAPE
RANDOM_SHAPE = True
RANDOM_SHAPE_per_CHUNK = RANDOM_SHAPE
RANDOM_SHAPE_mode = 'randomShape'
#####################################################
#####################################################    LIGHTS
RANDOM_LIGHTS = True
RANDOM_LIGHTS_per_CHUNK = True
#####################################################
#####################################################    HAND POSE
RANDOM_HAND_POSE = True
RANDOM_Pxl_BLUR_SIZE = True
RANDOM_CAMERA_JITTER = False
RANDOM_CROPPING_FACTOR = False
#####################################################
#####################################################
RANDOM_CAM_matrix_world = False
USE_MOTION_BLUR = True

#####################################################
#####################################################    CAMERA ROT - Z
#TODO: leads to movement of people outside of scene, since they are rotated arround their initial position
RANDOM_ROT_Z = False
RANDOM_BG_per_FRAME = not RANDOM_BG_per_CHUNK
RANDOM_TXT_per_FRAME = not RANDOM_TXT_per_CHUNK
RANDOM_LIGHTS_per_FRAME = not RANDOM_LIGHTS_per_CHUNK

n_bones = 52
total_pose_DOFs = 78
#####################################################
#####################################################
####################################################
DBG_FLAG_writeOnDisk = False
DBG_print_MOCAP_keys = False

bg_plane_dist = 10.0
bg_plane_dist_TOLERRANCE = 0.50
bg_plane_dist_INF = 1e+10 # depth larger than cut of depth are set to this value (not used in this
stepsizeFactor = 2

FILE_segm_per_v = './resources/segm_per_v_overlap_SMPLH.pkl'

FILE_sh_original = './resources/sh.osl'

PATHs_texture_participants_GENDER_SPECIFIC = True
PATHs_texture_participants = './smpl_data' \
                             '/fixed_textures' \
                             '/<GENDER' \
                             '>/<CEASAR>_<GENDER>*.jpg'

PATH_MoCap_SMPLH = './smpl_data/handPoses/per_SEQ___bodyHands/'
fingertipVerticesPATH = './resources/fingertips_SMPLH.json'

data_folder = './smpl_data'
FILE_smpl_data_MOCAP = join(data_folder, mocapDataName+'.npz')

data_folder_other = './resources'
FILE_smpl_data_OTHER = join(data_folder_other, 'smpl_data_ONLY_SHAPES_REGRESSORS.npz')


#


DBG_motion_data_filter = False
DBG_FBX_bone_names = False
DBG_segmentation = False


#####################################################
stepsize_hands = 10
frames_per_shape = 1
#####################################################
shape_totalCoeffs = 10
shape_ndofs = 10
#####################################################
#####################################################
#####################################################

####################################################
vblur_factor_Std = 0.03
vblur_factor_Miu = 0.03
vblur_factor = 0.

########used as plane loc after all the plane object was removed!
x_loc = -3
y_loc = -1
z_loc = 0

org_plane_loc = Vector((x_loc, y_loc, z_loc))

def project_joints3d_imwrite(joint_3d, img, intrinsic=None, extrinsic=None, imgPATH=None, cam_ob=None, scene=None, save_dbug_Imgs=True):
    for key in list(joint_3d.keys()):
        for ii in range(joint_3d[key].shape[0]):
            j3d = joint_3d[key][ii, :]
            j3d_4x1 = np.vstack([j3d.reshape((3, 1)), np.array([1])])
            if not(intrinsic is None or extrinsic is None):
                joint_2d = np.dot(np.dot(intrinsic, extrinsic), j3d_4x1)
                joint_2d = np.array([int(np.round(joint_2d[0] / joint_2d[2])),
                                     int(np.round(joint_2d[1] / joint_2d[2]))])
                cv2.circle(img, tuple((joint_2d[0], joint_2d[1])), 2, (0, 255, 255), 1)
            else:
                render_scale = scene.render.resolution_percentage / 100
                render_size = (int(scene.render.resolution_x * render_scale),
                               int(scene.render.resolution_y * render_scale))
                joint_2d = world2cam(scene, cam_ob, Vector((j3d_4x1[:])))
                joint_2d.x = joint_2d.x * render_size[0]
                joint_2d.y = -joint_2d.y * render_size[1]  + render_size[1] - 1

            cv2.circle(img, tuple((int(np.round(joint_2d.x)), int(np.round(joint_2d.y)))), 2, (0, 255, 255), 1)

    if save_dbug_Imgs:
        cv2.imwrite(imgPATH, img)
    print(imgPATH)


def cam_compute_intrinsic():

    res_x_px         = res[0]
    res_y_px         = res[1]
    f_mm             = cam_lens
    sensor_w_mm      = cam_sensor
    sensor_h_mm = sensor_w_mm * res_y_px / res_x_px

    scale = 1
    skew  = 0
    pixel_aspect_ratio = 1

    fx_px = f_mm * res_x_px * scale / sensor_w_mm
    fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

    # Center of the image
    u = np.round(res_x_px * scale / 2)
    v = np.round(res_y_px * scale / 2)

    # Intrinsic camera matrix
    K = np.array([[fx_px,  skew,   u],
                  [0,      fy_px,  v],
                  [0,      0,      1]])

    return K


def cam_compute_extrinsic(camera_RT_4_4):
    #does not work with camera pitch and z movement of camera.
    # I assume some of the matrices must be corrected for angle
    RT4x4 = camera_RT_4_4
    RRR = RT4x4[:3, :3]
    TTT = RT4x4[:3, 3]

    R_world2bcam = np.array([[0, 0, 1],
                             [0, -1, 0],
                             [-1, 0, 0]])
    T_world2bcam = -1 * np.dot(R_world2bcam, TTT)

    R_bcam2cv = np.array([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]])
    R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
    T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

    #
    RT = np.column_stack((R_world2cv, T_world2cv))
    # print
    # print RT.shape
    # print RT
    # print
    return RT


##########################################################################################################################
###########################################################################################################################

def load_hand_poses(PATH_hand_poses):
    print('\n\n\n')
    from glob import glob
    from os.path import join
    import pickle as pk
    paths = sorted(glob(join(PATH_hand_poses,'*.pkl')))
    print()
    hand_poses = []
    print('load_hand_poses')
    for pp in paths:
        print(pp)
        with open(pp, "rb") as fin:
            hand_poses_CURR = pk.load(fin, encoding='latin1')
        #
        # keep hands only !!!
        for ii in range(len(hand_poses_CURR)):
            hand_poses_CURR[ii] = hand_poses_CURR[ii][-ncomps:]     # keep only hands PCA pose, reject SIGA body pose  # ertert
            hand_poses_CURR[ii] = hand_poses_CURR[ii]
        #
        ignFr = 20
        hand_poses.append(hand_poses_CURR[ignFr:-ignFr])

    return hand_poses


def set_background(FLAG_RANDOM, bg_img):

    # bg_img = bg_img_INN.copy()  # not needed when not in function
    # with 100% probability, switch
    if True: # flip_coin(.1):
        bg_img.user_clear()
        bpy.data.images.remove(bg_img)
        #
        if FLAG_RANDOM:
            PATH_bg = choice(bg_paths)
        else:
            PATH_bg = bg_paths[0]
        #
        if FLAG_flip_BG:
            bg_img_TMP = cv2.imread(PATH_bg)
            bg_img_TMP = cv2.flip(bg_img_TMP, 1)
            #
            PATH_bg_TMP = join(PATH_tmp, join(name, 'bg_TMP.jpg'))
            if not exists(dirname(PATH_bg_TMP)):
                makedirs(dirname(PATH_bg_TMP))
            cv2.imwrite(PATH_bg_TMP, bg_img_TMP)
            bg_img = bpy.data.images.load(PATH_bg_TMP)
        else:
            bg_img = bpy.data.images.load(PATH_bg)

        scene.node_tree.nodes['Image'].image = bg_img
    return bg_img, PATH_bg


#############################################################################
#############################################################################


def set_txt(FLAG_RANDOM, cloth_img,txt_paths, i):
    # cloth_img = cloth_img_INN.copy()  # not needed when not in function
    # with 100% probability, switch texture
    if True: # flip_coin(.1):
        cloth_img.user_clear()
        bpy.data.images.remove(cloth_img)
        if FLAG_RANDOM:
            txt_PATH = choice(txt_paths)
        else:
            txt_PATH = txt_paths[0]
        print(txt_PATH)
        cloth_img = bpy.data.images.load(txt_PATH)
    return cloth_img, txt_PATH


##############################################################################
#############################################################################


def set_shape(FLAG_RANDOM, RANDOM_mode, i):

    if FLAG_RANDOM:
        if RANDOM_mode == 'randomShapeFromDist':
            shape = random_shapes[i](3.)    # RANDOM SHAPE from distribution
        elif RANDOM_mode == 'randomShape':
            shape = choice(fshapes)     # RANDOM SHAPE
        else:
            print('\n\n\n', 'not defined - set_sh ape - RANDOM_mode', '\n\n\n')
    else:
        shape = fshapes[0]

    return shape


##############################################################################
#############################################################################


def set_lights(RANDOM_LIGHTS):
    if RANDOM_LIGHTS:

        sh_coeffs = .7 * (2 * np.random.rand(9) - 1)
        sh_coeffs[0] = .5 + .9 * np.random.rand()  # first coeff is ambient
        sh_coeffs[1] = -.7 * np.random.rand()
    else:
        sh_coeffs = .7 * (2 * np.ones(9)*1.0 - 1)
        sh_coeffs[0] = .4 + .7 * 0.5  # first coeff is ambient
        sh_coeffs[1] = -.7 * 0.5
    return sh_coeffs


def get_x_range(cam_ob, plane_ob):

    cam_distance = org_plane_loc  - cam_ob.location
    lower_visibility_line = z_rotation(cam_distance, -cam_ob.data.angle_y)
    s = cam_ob.location[1] / lower_visibility_line[1]
    # s = cam_ob.location[1] / lower_visibility_line[1]
    x_upper_bound = cam_ob.location[0] - lower_visibility_line[0] * s
    if cam_ob.location[0] < -2:
        x_upper_bound=4.5
    x_range = [org_plane_loc[0], x_upper_bound]

    return x_range


# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'.encode()
#
def flow_write(filename,uv,v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        uv_ = np.array(uv)
        assert(uv_.ndim==3)
        if uv_.shape[0] == 2:
            u = uv_[0,:,:]
            v = uv_[1,:,:]
        elif uv_.shape[2] == 2:
            u = uv_[:,:,0]
            v = uv_[:,:,1]
        else:
            print('Wrong format for flow input')
            exit(-1)
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
#
def flow_read(filename, return_validity=False):
    """ Read optical flow from file, return (U,V) tuple.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    tmp = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width*2))
    u = tmp[:,np.arange(width)*2]
    v = tmp[:,np.arange(width)*2 + 1]

    if return_validity:
        valid = u<1e19
        u[valid==0] = 0
        v[valid==0] = 0
        return u,v,valid
    else:
        return u,v


def flow_2_img_fromFlowRawImg(flow):
    xx = flow[:, :, 0].astype(np.float32)  # horizontal
    yy = flow[:, :, 1].astype(np.float32)  # vertical

    mag, ang = cv2.cartToPolar(xx, yy, angleInDegrees=True)
    maxmag = 10
    mag = np.clip(mag, 0, maxmag) / maxmag

    hsv = np.ones((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = mag

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
#

def render_2d_pose(PATH_OUT_pose_joints_2d_VIZ, pose_joints_2d, sizYX):
    sizYX = (int(np.round(sizYX[0])), int(np.round(sizYX[1])))
    IMG_OUT_2d_pose = np.zeros((sizYX[0],sizYX[1],1), np.uint8)
    for key in list(pose_joints_2d.keys()):
        for ii in range(pose_joints_2d[key].shape[0]):
            centerr = tuple(np.round(pose_joints_2d[key][ii, :]).astype(int))
            cv2.circle(IMG_OUT_2d_pose, centerr, 0, (255, 255, 255))
    cv2.imwrite(PATH_OUT_pose_joints_2d_VIZ, IMG_OUT_2d_pose)


def setState0():
    for ob in bpy.data.objects.values():
        ob.select=False
    bpy.context.scene.objects.active = None


# create ONE MATERIAL PER PART
# as defined in a PKL with the segmentation
# this is useful to render the segmentation in a material pass
def create_segmentation(ob, person_nr):
    print('creating segmentation')
    mat_dict = {}
    vgroups = {}
    #
    print('exists(' + FILE_segm_per_v + ') --> ' + str(exists(FILE_segm_per_v)))
    with open(FILE_segm_per_v, 'rb') as f:
        vsegm = load(f)
    bpy.ops.object.material_slot_remove()
    parts = sorted(vsegm.keys())
    materialID_2_part = ['' for ii in range(len(part2num))]
    #
    for ipart, part in enumerate(parts):
        vs = vsegm[part]
        vgroups[part] = ob.vertex_groups.new(part)
        vgroups[part].add(vs, 1.0, 'ADD')
        bpy.ops.object.vertex_group_set_active(group=part)
        mat_dict[part] = bpy.data.materials['Material'].copy()
        mat_dict[part].pass_index = part2num[part] #+ person_nr*len(parts)
        bpy.ops.object.material_slot_add()
        ob.material_slots[-1].material = mat_dict[part]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_select()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')
        materialID_2_part[int((mat_dict[part].pass_index) / ((person_nr+   1)*len(parts))) -1] = part
        if DBG_segmentation:
            print('create_segmentation - (id+1) - %02d  //  %02d - %03d' % (ipart + 1, mat_dict[part].pass_index, mat_dict[part].pass_index * 5), ' -', part)
    if DBG_segmentation:
        exit(1)

    return(mat_dict, materialID_2_part)


# create the different passes that we render
def create_composite_nodes(tree, name, img=None):  # , idxx=0

    ######################################################################################################################
    res_paths = {k:join(PATH_tmp, name, '%s_BLENDER'%(k)) for k in ('depth', 'normal', 'txt', 'flow', 'segm','object_Id')}
    ######################################################################################################################

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create node for foreground image
    #################################################
    layers = tree.nodes.new('CompositorNodeRLayers')
    #################################################

    # Create node for background image
    bg_im = tree.nodes.new('CompositorNodeImage')
    if img is not None:
        bg_im.image = img

    FORMAT_file_FLOAT = 'OPEN_EXR'  # OPEN_EXR_MULTILAYER
    FORMAT_file_NON_float = 'PNG'
    NAMING_CONVENTION = '#####'

    # Create node for mixing foreground and background images
    mix = tree.nodes.new('CompositorNodeMixRGB')
    mix.use_alpha = True

    if MODE_NODES == 'OLD':
        # create node for mixing foreground and background images
        mix = tree.nodes.new('CompositorNodeMixRGB')
        mix.location = 40, 30
        mix.use_alpha = True
    # # Create node for the final output
    ############################################################
    composite_out = tree.nodes.new('CompositorNodeComposite')
    ############################################################

    # Create node for saving depth
    depth_out = tree.nodes.new('CompositorNodeOutputFile')
    depth_out.format.file_format = FORMAT_file_FLOAT
    depth_out.base_path = res_paths['depth']
    depth_out.file_slots[0].path = NAMING_CONVENTION

    # Create node for saving normals
    normals_out = tree.nodes.new('CompositorNodeOutputFile')
    normals_out.format.file_format = FORMAT_file_FLOAT
    normals_out.base_path = res_paths['normal']
    normals_out.file_slots[0].path = NAMING_CONVENTION

    # Create node for saving foreground image
    txt_out = tree.nodes.new('CompositorNodeOutputFile')
    txt_out.format.file_format = FORMAT_file_NON_float
    txt_out.base_path = res_paths['txt']
    txt_out.file_slots[0].path = NAMING_CONVENTION

    flow_out = tree.nodes.new('CompositorNodeOutputFile')
    flow_out.format.file_format = FORMAT_file_FLOAT
    flow_out.base_path = res_paths['flow']
    flow_out.file_slots[0].path = NAMING_CONVENTION

    segm_out = tree.nodes.new('CompositorNodeOutputFile')
    segm_out.format.file_format = FORMAT_file_FLOAT
    segm_out.base_path = res_paths['segm']
    segm_out.file_slots[0].path = NAMING_CONVENTION

    objectid_out = tree.nodes.new('CompositorNodeOutputFile')
    objectid_out.format.file_format = FORMAT_file_FLOAT
    objectid_out.base_path = res_paths['object_Id']
    objectid_out.file_slots[0].path = NAMING_CONVENTION

    if MODE_NODES == 'NEW' or USE_MOTION_BLUR:
        vecblur = tree.nodes.new('CompositorNodeVecBlur')
        vecblur.samples = 64
        vecblur.factor = vblur_factor
    if MODE_NODES == 'OLD':
        if USE_MOTION_BLUR:
            # Create node for saving output of vector blurred image
            vblur_out = tree.nodes.new('CompositorNodeComposite')
    elif MODE_NODES == 'NEW':
        finalblur = tree.nodes.new('CompositorNodeBlur')

    if MODE_NODES == 'NEW':
        tree.links.new(layers.outputs['Image'], vecblur.inputs[0])
        if '2.79' in bpy.app.version_string:
            tree.links.new(layers.outputs['Depth'], vecblur.inputs[1])
            tree.links.new(layers.outputs['Vector'], vecblur.inputs[2])
        else:
            tree.links.new(layers.outputs['Z'], vecblur.inputs[1])
            tree.links.new(layers.outputs['Speed'], vecblur.inputs[2])
        tree.links.new(vecblur.outputs[0], finalblur.inputs[0])
        #
        tree.links.new(finalblur.outputs[0], composite_out.inputs[0])
        tree.links.new(finalblur.outputs[0], txt_out.inputs[0])
        #
        if '2.79' in bpy.app.version_string:
            tree.links.new(layers.outputs['Depth'], depth_out.inputs[0])
            tree.links.new(layers.outputs['Vector'], flow_out.inputs[0])
        else:
            tree.links.new(layers.outputs['Z'], depth_out.inputs[0])
            tree.links.new(layers.outputs['Speed'], flow_out.inputs[0])
        tree.links.new(layers.outputs['Normal'], normals_out.inputs[0])
        tree.links.new(layers.outputs['IndexMA'], segm_out.inputs[0])
        tree.links.new(layers.outputs['IndexOB'], objectid_out.inputs[0])

    elif MODE_NODES == 'OLD':
        # Merge fg and bg images
        tree.links.new(bg_im.outputs[0], mix.inputs[1])                     #bg
        tree.links.new(layers.outputs['Image'], mix.inputs[2])              #fg
        #
        if not USE_MOTION_BLUR:
            tree.links.new(mix.outputs[0], composite_out.inputs[0])

        tree.links.new(layers.outputs['Image'], txt_out.inputs[0])          #Save fg

        if '2.79' in bpy.app.version_string:
            tree.links.new(layers.outputs['Depth'], vecblur.inputs[1])
            tree.links.new(layers.outputs['Vector'], vecblur.inputs[2])
        else:
            tree.links.new(layers.outputs['Z'], vecblur.inputs[1])
            tree.links.new(layers.outputs['Speed'], vecblur.inputs[2])


        tree.links.new(layers.outputs['Normal'], normals_out.inputs[0])     #Save normal                                                  OK
        tree.links.new(layers.outputs['IndexMA'], segm_out.inputs[0])       #Save segmentation  # Index MAterial - Material ID            OK
        tree.links.new(layers.outputs['IndexOB'], objectid_out.inputs[0])
        #
        if USE_MOTION_BLUR:
            #if(output_types['vecblur']):
            tree.links.new(mix.outputs[0], vecblur.inputs[0])               #apply vector blur on the bg+fg image,         - blur
            if '2.79' in bpy.app.version_string:
                tree.links.new(layers.outputs['Depth'], vecblur.inputs[1])
                tree.links.new(layers.outputs['Vector'], vecblur.inputs[2])
            else:
                tree.links.new(layers.outputs['Z'], vecblur.inputs[1])          #  using depth,
                tree.links.new(layers.outputs['Speed'], vecblur.inputs[2])      #  and flow.
            tree.links.new(vecblur.outputs[0], vblur_out.inputs[0])         #Save vblurred output

    return(res_paths)


# Creation of the spherical harmonics material, using an OSL script
def create_sh_material(tree, PATH_sh, img=None):
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_im = tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400
    if img is not None:
        uv_im.image = img

    rgb = tree.nodes.new('ShaderNodeRGB')
    rgb.location = -400, 200

    script = tree.nodes.new('ShaderNodeScript')
    script.location = -230, 400
    script.mode = 'EXTERNAL'
    script.filepath = PATH_sh
    script.update()

    # the emission node makes it independent of the scene lighting
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = -60, 400

    mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400

    tree.links.new(uv.outputs[2], uv_im.inputs[0])
    tree.links.new(uv_im.outputs[0], script.inputs[0])
    tree.links.new(script.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs[0])


def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector)

def create_textured_plane(cam_ob, scene, res, cam_distance=9, impath=None,camera_pitch=None):
    x_loc = -3
    y_loc = -1
    z_loc = 0
    bpy.ops.mesh.primitive_plane_add(radius=3, location=(x_loc, y_loc, z_loc), rotation=(1.5708, 1.5708, -1.5708))

    cam_distance = Vector((x_loc, y_loc, z_loc)) - cam_ob.location
    lower_visibility_line = z_rotation(cam_distance, -cam_ob.data.angle_x)

    s = (cam_ob.location[0]-3) / lower_visibility_line[0]
    y_visibl = cam_ob.location[1] - s*lower_visibility_line[1]
    print('------------------------------------------------')
    print(y_visibl)

    # select the plane
    bpy.ops.object.select_all(action='DESELECT')
    plane = bpy.data.objects['Plane']
    plane.select = True
    scene.objects.active = plane

    #scale plane to cover visible area
    vertices = [plane.matrix_world * vert.co for vert in   bpy.data.meshes['Plane'].vertices]
    vertices = np.array(vertices)
    max_x, max_z, max_y = np.max(vertices, axis=0)
    min_x, min_z, min_y = np.min(vertices, axis=0)
    visible_line_y = math.tan(cam_ob.data.angle_x / 2) * (12) #12 is distance to plane
    visible_line_z = math.tan(cam_ob.data.angle_y / 2) * (12)
    scale_y = visible_line_y / max_y
    scale_z = scale_y * res[1] / res[0]
    bpy.ops.transform.resize(value=(1, scale_z, scale_y))


    plane_ob = bpy.data.objects['Plane']
    planemat = bpy.data.materials.new('PlaneMat')
    planemat.alpha=0.
    plane_ob.data.materials.append(planemat)
    planemat.use_nodes = True

    bpy.ops.object.editmode_toggle()
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001, correct_aspect=True)
    bpy.ops.object.editmode_toggle()

    tree  = planemat.node_tree
    uv    = tree.nodes.new('ShaderNodeTexCoord')
    texim = tree.nodes.new('ShaderNodeTexImage')
    mat   = tree.nodes.new('ShaderNodeEmission')
    lpath = tree.nodes.new('ShaderNodeLightPath')
    out   = tree.nodes['Material Output']
    mix   = tree.nodes.new('ShaderNodeMixShader')

    if impath is not None:
        im = bpy.data.images.load(impath)
        texim.image  = im

    tree.links.new(uv.outputs[0], texim.inputs[0])
    tree.links.new(texim.outputs[0], mat.inputs[0])
    tree.links.new(mat.outputs[0], mix.inputs[2])
    tree.links.new(lpath.outputs[0], mix.inputs[0])
    tree.links.new(mix.outputs[0], out.inputs[0])
    extrema = {}
    extrema[' max_z'] = max_z
    extrema[' max_y'] = max_y
    extrema[' min_z'] = min_z
    extrema[' min_y'] = min_y
    extrema[' min_x'] = min_x
    extrema[' min_x'] = min_x

    plane_ob.location[1] = -100
    return plane_ob, texim, extrema, lower_visibility_line


# Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)


def init_scene(gender, nr=0, lower_visibility_line=None):
    ###########################################################################
    ###########################################################################

    pathhh = join(data_folder, '%s_avg_noFlatHand.fbx' % gender[0])
    print(pathhh)
    bpy.ops.import_scene.fbx(filepath=pathhh, axis_forward='X', axis_up='Z', global_scale=100)
    obname = '%s_avg' % gender[0]
    ##########################################################################
    ##########################################################################

    ob = bpy.data.objects[obname]
    ob.data.use_auto_smooth = False  # autosmooth creates weird artifacts

    # assign the existing spherical harmonics material
    ob.active_material = bpy.data.materials['Material']

    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    if 'Cube' in bpy.data.objects.keys():
        bpy.data.objects['Cube'].select = True
        bpy.ops.object.delete(use_global=False)
        # delete the default cube (which held the material)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Lamp'].select = True
        bpy.ops.object.delete(use_global=False)

    cam_ob = bpy.data.objects['Camera']
    #
    scn = bpy.context.scene
    ##########################################
    # if scene is set up for the first time
    bpy.ops.object.select_all(action='DESELECT')
    if nr == 0:
        # set camera properties and initial position
        scn.objects.active = cam_ob
        camT_mu    = 9.0
        camT_stdev = 0.1
        if RANDOM_CAM_matrix_world:
            trr = np.random.normal(camT_mu, camT_stdev)
        else:
            trr = camT_mu

        cam_y_offset = 0 * trr #* 0.5
        if cam_y_offset == 0:
            alpha = math.radians(90)
        else:
            alpha = math.atan( (trr + 3) /-cam_y_offset)  #+3 is the plane distance!
        camera_pitch = -math.radians(180 - 90 - math.degrees(alpha))
        print(math.degrees(camera_pitch))

        cam_ob.matrix_world = Matrix(((0.0,  0.0,  1.0,  trr),
                                      (0.0, -1.0,  0.0, cam_y_offset-1),
                                      (1.0,  0.0,  0.0,  0.0),
                                      (0.0,  0.0,  0.0,  1.0)))

        cam_ob.select = True
        bpy.ops.transform.rotate(value=camera_pitch, axis=(0,0,1))

        cam_ob.data.lens = cam_lens
        cam_ob.data.sensor_width = cam_sensor
        cam_ob.data.clip_start = 0.1

        plane_ob, texim, extrema, lower_visibility_line = create_textured_plane(cam_ob, scene, res, cam_distance=trr+3,
                                                                                camera_pitch=camera_pitch)
        ##########################################

        # Setup an empty object in the center which will be the parent of the Camera
        # This allows to easily rotate an object around the origin
        scn.cycles.film_transparent = True
        scn.render.layers["RenderLayer"].use_pass_vector = True
        scn.render.layers["RenderLayer"].use_pass_normal = True
        scene.render.layers['RenderLayer'].use_pass_emit = True
        scene.render.layers['RenderLayer'].use_pass_emit = True
        scene.render.layers['RenderLayer'].use_pass_material_index = True
        scene.render.layers['RenderLayer'].use_pass_object_index = True
        if '2.79' in bpy.app.version_string:
            scene.render.layers['RenderLayer'].use_pass_vector = True

        # Set render size
        scn.render.resolution_x = res[0]
        scn.render.resolution_y = res[1]
        scn.render.resolution_percentage = 100
        scn.render.image_settings.file_format = 'PNG'
    else:
        plane_ob = bpy.data.objects['Plane']

    # clear existing animation data
    ob.data.shape_keys.animation_data_clear()
    arm_ob = bpy.data.objects['Armature']
    arm_ob.animation_data_clear()

    ##########################################
    # and translate it randomly on x-y plane
    bpy.ops.object.select_all(action='DESELECT')
    arm_ob.select = True
    bpy.context.scene.objects.active = arm_ob

    cam_distance =  org_plane_loc - cam_ob.location
    lower_visibility_line = z_rotation(cam_distance, -cam_ob.data.angle_y)
    s = cam_ob.location[1] / lower_visibility_line[1]
    x_upper_bound = cam_ob.location[0] - lower_visibility_line[0] * s
    x_range = [org_plane_loc[0], x_upper_bound]

    translation = random_placement_in_visibl_area(cam_ob, x_range)

    bpy.ops.transform.translate(value=(translation[0], 0, translation[1]))

    ob.name = obname + '.%03d' % nr
    obname = ob.name
    for name in arm_ob.pose.bones.keys():
        bone = arm_ob.pose.bones.get(name)
        if bone is None:
            continue
        bone.name = obname + name.split('avg')[-1]
    # give a new name to distinguish objects later
    arm_ob.name = 'Armature.%03d' % nr
    arm_ob.pass_index = nr + 1
    ob.pass_index = nr + 1

    if not(nr == 0):
        plane_ob = bpy.data.objects['Plane']
        return(ob, obname, arm_ob, cam_ob, plane_ob, None, lower_visibility_line)
    else:
        return(ob, obname, arm_ob, cam_ob, plane_ob, texim, lower_visibility_line)


def random_placement_in_visibl_area(cam_ob, x_range=[-3,2]):
    translation = np.zeros((2,1))
    translation[0] = np.random.uniform(x_range[0], x_range[1], 1)
    visible_line = math.tan( cam_ob.data.angle/2 ) * ( -translation[0] + cam_ob.location[0] )
    translation[1] = np.random.uniform(-visible_line, visible_line, 1)

    return translation

def vector_move(colliding_person, translation=None, y=None):
    if y is None:
        y = colliding_person.location[1]

    print(colliding_person.location)
    colliding_person.location = Vector((0,y,0))
    scene.update()
    print(colliding_person.location)
    if translation is None:
        translation = random_placement_in_visibl_area(cam_ob)
    vec = Vector((translation[0],y, translation[1]))
    colliding_person.location = vec

    scene.update()
    print(colliding_person.location)


# transformation between pose and blendshapes. Final choice in SMPL paper was
#   a flattened version of the rotation matrix after subtracting the identity
def rodrigues2bshapes(pose, n_bones):
    rod_rots = np.asarray(pose).reshape(n_bones, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]

    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])

    return(mat_rots, bshapes)


# apply trans pose and shape to character
def apply_trans_pose_shape(trans, pose, shape, ob, arm_ob, obname, scene, cam_ob, n_bones, frame=None, init_trans=Vector((0,0,0)),  DBG_exitIfNeeded=False):
    # set z to zero, to make sure all are translated to zero-height (y-in blender)
    init_trans[2] = 0
    pose_coeffs = pose.copy()
    pose = []
    pose_coeffs[body_pose_dofs:(body_pose_dofs + ncomps)].dot(selected_components)
    ###############################################################
    full_hand_pose = pose_coeffs[body_pose_dofs:(body_pose_dofs + ncomps)].dot(selected_components)
    ##################################################################
    mixed_body_full_hand_pose = np.concatenate((pose_coeffs[:body_pose_dofs], hands_mean + full_hand_pose))
    ##########################################################################################################
    pose = mixed_body_full_hand_pose.copy()
    ##############################################
    mrots, bsh = rodrigues2bshapes(pose, n_bones)
    ##############################################

    if DBG_FBX_bone_names:
        print(arm_ob.pose.bones)
        for ii, bb in enumerate(arm_ob.pose.bones):
            print('%02d - %-16s' % (ii, bb.name))
        exit(1)

    # set the location of the first bone to the translation parameter
    arm_ob.pose.bones[obname+'_'+part_match['bone_00']].location = trans - init_trans
    if frame is not None:
        arm_ob.pose.bones[obname+'_'+part_match['root']].keyframe_insert('location', frame=frame)
    for ibone, mrot in enumerate(mrots):

        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

    # apply pose blendshapes
    for ibshape, bshape in enumerate(bsh):
        ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
        if frame is not None:
            ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].slider_min = -10.0
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].slider_max =  10.0
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
        if frame is not None:
            ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

    return init_trans

def get_head_boundingbox(arm_ob, ob, render_size):
    render_scale = scene.render.resolution_percentage / 100
    render_size = (int(scene.render.resolution_x * render_scale),
                   int(scene.render.resolution_y * render_scale))

    head_bounding_box_vertices = [486, 3975, 457, 331, 411, 3050] #corresponding to left ear, right ear, back of head, tip of nose, upper head, adams apple
    mesh            = ob.to_mesh(scene, True, 'PREVIEW')
    mesh_vertices   = [arm_ob.matrix_world * vert.co for vert in mesh.vertices]
    bone_locations_2d = np.zeros((6,2))
    for i,vertex in enumerate(head_bounding_box_vertices):
        co_2d = world2cam(scene, cam_ob, mesh_vertices[vertex])  # triangle 411 is on the top of the head

        bone_locations_2d[i,:] = [co_2d.x * render_size[0],
                                 -co_2d.y * render_size[1] + render_size[1] - 1]

    x_min, y_min = np.min(bone_locations_2d,axis=0)
    x_max, y_max = np.max(bone_locations_2d, axis=0)
    head_bb_dict = {'x1': x_min,
                   'x2': x_max,
                   'y1': y_min,
                   'y2': y_max}

    return head_bb_dict


def get_upper_head(arm_ob, ob, render_size):
    mesh            = ob.to_mesh(scene, True, 'PREVIEW')
    mesh_vertices   = [arm_ob.matrix_world * vert.co for vert in mesh.vertices]
    co_2d           = world2cam(scene, cam_ob, mesh_vertices[411])  # triangle 411 is on the top of the head
    co_3d           = mesh_vertices[411]

    bone_locations_2d = (co_2d.x * render_size[0],
                         -co_2d.y * render_size[1] + render_size[1] - 1)
    bone_locations_3d = (co_3d.x,
                         co_3d.y,
                         co_3d.z)

    bpy.data.meshes.remove(mesh)

    return bone_locations_2d, bone_locations_3d


def get_bone_locs(arm_ob, ob, obname, scene, cam_ob, n_bones, UPPER_HEAD):
    render_scale = scene.render.resolution_percentage / 100
    render_size = (int(scene.render.resolution_x * render_scale),
                   int(scene.render.resolution_y * render_scale))

    n_fingertips = 10

    if UPPER_HEAD:
        bone_locations_2d = np.empty((n_bones + n_fingertips + 1, 2))
        bone_locations_3d = np.empty((n_bones + n_fingertips + 1, 3), dtype='float32')
    else:
        bone_locations_2d = np.empty((n_bones + n_fingertips, 2))
        bone_locations_3d = np.empty((n_bones + n_fingertips, 3), dtype='float32')
    bone_loc_names = []


    idx = 0
    for ibone in range(n_bones):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        if UPPER_HEAD and part_match['bone_%02d' % ibone] == 'Head':
            bone_locations_2d[-1], bone_locations_3d[-1] = get_upper_head(arm_ob, ob, render_size)
        infoAdd = ''
        ######################################################################
        ######################################################################
        if True:
            if True:
                co_2d = world2cam(scene, cam_ob, arm_ob.matrix_world * bone.head)
                co_3d = arm_ob.matrix_world * bone.head
                bone_locations_3d[idx] = (co_3d.x,
                                          co_3d.y,
                                          co_3d.z)

                bone_locations_2d[idx] = (co_2d.x * render_size[0],
                                         -co_2d.y * render_size[1] + render_size[1]-1)
                bone_loc_names.append(part_match['bone_%02d' % ibone] + infoAdd)
                #############
                idx = idx + 1
        ########################################################################
        #######################################################################
        endEffectorNamesList = ['index2','middle2','pinky2','ring2','thumb2']

        #
        if any(pp in part_match['bone_%02d' % ibone] for pp in endEffectorNamesList):
            co_2d = world2cam(scene, cam_ob, arm_ob.matrix_world * bone.tail)
            co_3d = arm_ob.matrix_world * bone.tail
            #
            bone_locations_3d[idx] = (co_3d.x,
                                      co_3d.y,
                                      co_3d.z)
            bone_locations_2d[idx] = (co_2d.x * render_size[0],
                                     -co_2d.y * render_size[1] + render_size[1]-1)
            #
            infoAdd = '------------------------------ TAIL'
            print('~~~ ibone %02d - %02d - %-15s' % (ibone, idx, part_match['bone_%02d' % ibone]), bone_locations_2d[idx], bone_locations_3d[idx], infoAdd)
            bone_loc_names.append(part_match['bone_%02d' % ibone] + infoAdd)
            #############
            idx = idx + 1

    return(bone_locations_2d, bone_locations_3d, bone_loc_names)


# reset the joint positions of the character according to its new shape
def reset_joint_positions(shape, ob, arm_ob, obname, scene, cam_ob, reg_ivs, joint_reg, n_bones, total_pose_DOFs):

    # Since the regression is sparse, only the relevant vertex
    #     elements (joint_reg) and their indices (reg_ivs) are loaded
    # zero the pose and trans to obtain joint positions in zero pose
    print('\n\n', 'apply_trans_pose_shape - AAA', '\n\n')
    ######################################################################
    _ = apply_trans_pose_shape(orig_trans, np.zeros(total_pose_DOFs), shape, ob, arm_ob, obname, scene, cam_ob, n_bones, DBG_exitIfNeeded=False)
    ########################################################################

    # obtain a mesh after applying modifiers
    bpy.ops.wm.memory_statistics()
    me = ob.to_mesh(scene, True, 'PREVIEW')

    full_mesh = np.empty((len(me.vertices), 3))
    for ivertexx, vertexx in enumerate(me.vertices):
        full_mesh[ivertexx] = np.array(vertexx.co)


    reg_vs = np.empty((len(reg_ivs), 3))
    for iiv, iv in enumerate(reg_ivs):
        reg_vs[iiv] = me.vertices[iv].co
    bpy.data.meshes.remove(me)

    # regress joint positions in rest pose
    #######################################
    joint_xyz = joint_reg.dot(reg_vs)
    #######################################

    # adapt joint positions in rest pose
    arm_ob.hide = False
    bpy.ops.object.mode_set(mode='EDIT')
    arm_ob.hide = True
    for ibone in range(n_bones):
        bb = arm_ob.data.edit_bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bboffset = bb.tail - bb.head
        bb.head = joint_xyz[ibone]
        bb.tail = bb.head + bboffset  # there is a Maya-Blender compatibility issue!!! - this will be overwritten for fingertips _2 exactly below !!!
        boneName = part_match['bone_%02d' % ibone]
        if any(pp in boneName for pp in ['rring2', 'rmiddle2', 'rindex2', 'lthumb2', 'lmiddle2', 'lring2', 'rthumb2', 'lindex2', 'lpinky2', 'rpinky2']):
            currFingertipVerticesNP = full_mesh[fingertipVerticesDICT[boneName[:-1]]]
            bb.tail = np.average(currFingertipVerticesNP, axis=0)

    bpy.ops.object.mode_set(mode='OBJECT')
    #
    return(shape)


# load poses and shapes
def load_body_data(smpl_data_MOCAP, smpl_data_OTHER, ob, obname, gender,i=None):  # ='female'):
    # load moshed data from MOCAP dataset
    # create a dictionary with key the sequence name and values the pose and trans
    cmu_parms = {}
    cmu_frames = {}
    cmu_stepsize = {}
    for seq in smpl_data_MOCAP.files:
        if not(seq.startswith('pose')):
            continue

        cmu_parms[seq.replace('pose_', '')] = {'poses':smpl_data_MOCAP[seq][:,:body_pose_dofs],
                                               'trans':smpl_data_MOCAP[seq.replace('pose_','trans_')]}
        cmu_frames[seq.replace('pose_', '')] = len(smpl_data_MOCAP[seq])
        if 'cmu' in seq:
            cmu_stepsize[seq.replace('pose_', '')] = 5
            cmu_stepsize[seq.replace('pose_', '')] = 5
        else:
            cmu_stepsize[seq.replace('pose_', '')] = 10

    print('cmu_parms')
    for ii, kk in enumerate(cmu_parms.keys()):
        print('%03d  -  %-20s' % (ii, kk))
    #
    if DBG_motion_data_filter:
        print('len(cmu_parms) = ' + str(len(cmu_parms)))
        exit(1)

    n_sh_bshapes = len([k for k in ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])

    fshapes = smpl_data_OTHER['%sshapes' % gender][:, :n_sh_bshapes]

    return(cmu_parms, fshapes, cmu_frames, cmu_stepsize)


def render_segmentation(PATH_BLENDER_segm, PATH_OUT_segm_PNG, PATH_OUT_segm_VIZ, save_dbug_imgs):
    import numpy as np
    import cv2
    segm = cv2.imread(PATH_BLENDER_segm, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    segm = np.uint8(segm)
    if (len(segm.shape) == 3 and np.sum(np.abs(segm[:, :, 0] - segm[:, :, 1])) == 0 and np.sum(np.abs(segm[:, :, 1] - segm[:, :, 2])) == 0 and np.sum(np.abs(segm[:, :, 2] - segm[:, :, 0])) == 0):
        segm = segm[:, :, 0]
    segm_nonzero = segm[np.nonzero(segm)]
    print('\n\n\n')
    print(segm_nonzero.shape)
    print('\n\n\n')
    if segm_nonzero.shape[0] > 0:
        VIZmultipl = np.int(np.floor(255 / np.amax(segm_nonzero)))
    else:
        VIZmultipl = bg_plane_dist_INF
    segmVIZ = segm * VIZmultipl
    if save_dbug_Imgs:
        cv2.imwrite(PATH_OUT_segm_PNG, segm)
        cv2.imwrite(PATH_OUT_segm_VIZ, segmVIZ)
    #
    return segm


def depth_2_depthVIZ(PATH_depth_EXR, PATH_OUT_depth_VIZ, DBG_verbose=False, VIZ_HACK=True):
    #
    depth = cv2.imread(PATH_depth_EXR, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #
    depth_min = np.amin(depth[np.nonzero(depth)])
    depth_max = np.amax(depth[np.nonzero(depth)])
    if VIZ_HACK:
        depth[depth == 1e+10] = 0
        if DBG_verbose and depth[np.nonzero(depth)].shape[0] > 0:
            print(np.unique(np.hstack(depth[np.nonzero(depth)])))
            print(len(np.unique(np.hstack(depth[np.nonzero(depth)]))))
            print(' - VIZ HACK')
    #
    depthVIZ = np.zeros_like(depth)
    cv2.normalize(depth, depthVIZ, alpha=0., beta=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    depthVIZ_min = 0
    depthVIZ_max = 0
    if depthVIZ[np.nonzero(depthVIZ)].shape[0] > 0:
        depthVIZ_min = np.amin(depthVIZ[np.nonzero(depthVIZ)])
        depthVIZ_max = np.amax(depthVIZ[np.nonzero(depthVIZ)])
    if DBG_verbose:
        print(depthVIZ.shape)
        print('Non-Zero:')
        print('b - depthVIZ_min =', depthVIZ_min, '  ', 'depthVIZ_max =', depthVIZ_max)
    if VIZ_HACK:
        depthVIZ = np.clip(depthVIZ - depthVIZ_min * 0.80, 0., 255.)
        if DBG_verbose:
            print('depthVIZ_min * 0.98 = ', depthVIZ_min * 0.98)
    cv2.normalize(depthVIZ, depthVIZ, alpha=0., beta=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    if DBG_verbose and depthVIZ[np.nonzero(depthVIZ)].shape[0] > 0:
        depthVIZ_min = np.amin(depthVIZ[np.nonzero(depthVIZ)])
        depthVIZ_max = np.amax(depthVIZ[np.nonzero(depthVIZ)])
        print(depthVIZ.shape)
        print('Non-Zero:')
        print('c - depthVIZ_min =', depthVIZ_min, '  ', 'depthVIZ_max =', depthVIZ_max)
        print(np.unique(np.hstack(depthVIZ[np.nonzero(depthVIZ)])))
        print(len(np.unique(np.hstack(depthVIZ[np.nonzero(depthVIZ)]))))
    cv2.imwrite(PATH_OUT_depth_VIZ, depthVIZ)

def get_annorect(annolist, head_anno, img_anno, objectpos_anno, scale_anno, pose_joints_2d, joints_2d_vsbl, head_bbx_dict, img_path, scale_dict):
    smplToMPI = [8, 5, 2, 1, 4, 7, 0, 12, 15, 62, 21, 19, 17, 16, 18, 20]

    annorect = []
    head_bb_anno = []
    obj_pos = []
    scales = []
    for obname in joints_2d_vsbl.keys():
        pose_joints_2d_person = pose_joints_2d[obname][smplToMPI]
        joints_2d_vsbl_person = joints_2d_vsbl[obname][smplToMPI]
        head_bb = head_bbx_dict[obname]
        # point = []

        annopoints = []
        for i in range(pose_joints_2d_person.shape[0]):
            point = {}
            if pose_joints_2d_person[i,0] >= 0 and pose_joints_2d_person[i,0] <=res[0] and pose_joints_2d_person[i,1] >= 0 and pose_joints_2d_person[i,1] <=res[1]:
                point= {'id':i,
                        'x':pose_joints_2d_person[i,0],
                        'y':pose_joints_2d_person[i,1],
                        'is_visible':joints_2d_vsbl_person[i],
                        'is_real': 0}

            annopoints.append(point)
        annorect.append(annopoints)
        head_bb_anno.append((head_bb['x1'], head_bb['y1'], head_bb['x2'], head_bb['y2']))
        obj_pos.append((pose_joints_2d[obname][9,:]))
        scales.append(scale_dict[obname])

    annolist.append(annorect)
    head_anno.append(head_bb_anno)
    img_anno.append(img_path)
    objectpos_anno.append(obj_pos)
    scale_anno.append(scales)

    return annolist, head_anno, img_anno, objectpos_anno, scale_anno

def get_annorect_from_real_annolist(annolist, head_anno, img_anno, objectpos_anno, scale_anno, annolist_real,
                                    img, plane_ob, intrinsic, extrinsic, debug_res_reduction, res,
                                    cam_ob, scene, img_path,PATH_INN_segm_EXR,):
    rect = []
    head_bb_anno = []
    obj_pos = []
    scales = []

    try:
        for annorect in annolist_real.annorect:
            head_bb_anno.append((annorect.x1,annorect.y1, annorect.x2, annorect.y2))
            obj_pos.append([annorect.objpos.x, annorect.objpos.y])
            scales.append(annorect.scale)
            ids = []
            x = []
            y = []
            is_visible = []

            for i in range(len(annorect.annopoints.point)):
                ids.append(annorect.annopoints.point[i].id)
                x.append(annorect.annopoints.point[i].x)
                y.append(annorect.annopoints.point[i].y)
                if not(isinstance(annorect.annopoints.point[i].is_visible, int) or isinstance(annorect.annopoints.point[i].is_visible, str)):
                    if len(annorect.annopoints.point[i].is_visible)==0:
                        is_visible.append(1)
                    else:
                        if int(annorect.annopoints.point[i].is_visible[0]) == 1:
                            is_visible.append(1)
                        else:
                            is_visible.append(0)
                else:
                    if int(annorect.annopoints.point[i].is_visible) == 1:
                        is_visible.append(1)
                    else:
                        is_visible.append(0)


            x_y_locations, visibility = project_joints_bg_imwrite(x, y, img, plane_ob, intrinsic, extrinsic,
                                                                  is_visible, debug_res_reduction, res, cam_ob, scene, annorect.scale, img_path,PATH_INN_segm_EXR)


        # for annorect in annolist_real.annorect:
            annopoints = []
            for i in range(len(annorect.annopoints.point)):
                point = {'id': annorect.annopoints.point[i].id,
                         'x': annorect.annopoints.point[i].x,
                         'y': annorect.annopoints.point[i].y,
                         'is_visible': annorect.annopoints.point[i].is_visible,
                         'is_real':True}
                annopoints.append(point)
            rect.append(annopoints)
    except:
        annorect = annolist_real.annorect

        head_bb_anno.append((annorect.x1,annorect.y1, annorect.x2, annorect.y2))
        obj_pos.append([annorect.objpos.x, annorect.objpos.y])
        scales.append(annorect.scale)
        ids = []
        x = []
        y = []
        is_visible = []

        for i in range(len(annorect.annopoints.point)):
            ids.append(annorect.annopoints.point[i].id)
            x.append(annorect.annopoints.point[i].x)
            y.append(annorect.annopoints.point[i].y)
            if not(isinstance(annorect.annopoints.point[i].is_visible, int) or isinstance(annorect.annopoints.point[i].is_visible, str)):
                if len(annorect.annopoints.point[i].is_visible)==0:
                    is_visible.append(1)
                else:
                    if int(annorect.annopoints.point[i].is_visible[0]) == 1:
                        is_visible.append(1)
                    else:
                        is_visible.append(0)
            else:
                if int(annorect.annopoints.point[i].is_visible) == 1:
                    is_visible.append(1)
                else:
                    is_visible.append(0)


        x_y_locations, visibility = project_joints_bg_imwrite(x, y, img, plane_ob, intrinsic, extrinsic,
                                                              is_visible, debug_res_reduction, res, cam_ob, scene, annorect.scale, img_path,PATH_INN_segm_EXR)


    # for annorect in annolist_real.annorect:
        annopoints = []
        for i in range(len(annorect.annopoints.point)):
            point = {'id': annorect.annopoints.point[i].id,
                     'x': annorect.annopoints.point[i].x,
                     'y': annorect.annopoints.point[i].y,
                     'is_visible': annorect.annopoints.point[i].is_visible,
                     'is_real': True}
            annopoints.append(point)
        rect.append(annopoints)

    annolist.append(rect)
    head_anno.append(head_bb_anno)
    objectpos_anno.append(obj_pos)
    scale_anno.append(scales)

    return annolist, head_anno, img_anno, objectpos_anno, scale_anno

def project_joints_bg_imwrite(joint_z, joint_y, img, plane_ob, intrinsic, extrinsic, visibility, debug_res_reduction, res, cam_ob, scene, scale, img_path,PATH_INN_segm_EXR):

    render_scale = scene.render.resolution_percentage / 100
    render_size = (int(scene.render.resolution_x * render_scale),
                   int(scene.render.resolution_y * render_scale))

    joint_z = np.array(joint_z)
    joint_y = np.array(joint_y)
    visibility = np.array(visibility)
    locations = np.zeros((joint_z.shape[0],2))
    vertices = [plane_ob.matrix_world * vert.co for vert in bpy.data.meshes['Plane'].vertices]
    vertices = np.array(vertices)

    max_x, max_y, max_z = np.max(vertices, axis=0)
    min_x, min_y, min_z = np.min(vertices, axis=0)

    joint_rel_y = joint_y / (res[1] * debug_res_reduction)
    joint_rel_z = joint_z / (res[0] * debug_res_reduction)

    joint_y = (max_y - min_y) * joint_rel_y + min_y
    joint_z = (max_z - min_z) * joint_rel_z + min_z

    min_x, _, _ = np.min(vertices, axis=0)


    render_scale = scene.render.resolution_percentage / 100
    render_size = (int(scene.render.resolution_x * render_scale),
                   int(scene.render.resolution_y * render_scale))

    for i, (y,z) in enumerate(zip(joint_y, joint_z)):
        j3d = np.array([min_x, y, z])
        j3d_4x1 = np.vstack([j3d.reshape((3, 1)), np.array([1])])

        joint_2d = np.dot(np.dot(intrinsic, extrinsic), j3d_4x1)
        joint_2d = np.array([int(np.round(joint_2d[0] / joint_2d[2])), int(np.round(joint_2d[1] / joint_2d[2]))])
        locations[i,:] = joint_2d[::-1]

        joint_2d = world2cam(scene, cam_ob, Vector((j3d_4x1[:])))
        joint_2d.x = joint_2d.x * render_size[0]
        joint_2d.y = -joint_2d.y * render_size[1] + render_size[1] - 1
        locations[i,:] = [ joint_2d.x, joint_2d.y ]

    vsbl = check_for_occlusion_of_joint(locations[:,1], locations[:,0],
                                                 visibility,
                                                 PATH_INN_segm_EXR, scale)

    for i, (y, z) in enumerate(zip(joint_y, joint_z)):
        if vsbl[i]:
            cv2.circle(img, tuple((int(locations[i,0]), int(locations[i,1]))), 2, (0, 255, 0), 1)
        else:
            cv2.circle(img, tuple((int(locations[i,0]), int(locations[i,1]))), 2, (0, 0, 255),1)

    cv2.imwrite(img_path, img)
    locations_dict = {}
    locations_dict['bg_joint_locations'] =  locations

    return locations, vsbl


def check_for_occlusion_of_joint(joint_y, joint_x, visibility, PATH_INN_segm_EXR, scale):
        segm_EXR = cv2.imread(PATH_INN_segm_EXR, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        print(joint_y.shape)
        for joint_nr, (x, y) in enumerate(zip(joint_x, joint_y)):
            jj = np.round(x).astype(int)  # xx
            ii = np.round(y).astype(int)  # yy
            ii = ii
            jj = jj

            print(ii)
            print(jj)


            print(res)
            ff = 5 * scale
            iiMin = np.round((np.max((ii - ff, 0))))
            iiMin = int(iiMin)
            iiMax = np.round(np.min((ii + ff + 1, res[1])))
            iiMax = int(iiMax)
            jjMin = np.round(np.max((jj - ff, 0)))
            jjMin = int(jjMin)
            jjMax = np.round(np.min((jj + ff + 1, res[0])))
            jjMax = int(jjMax)
            print('---------------')
            print(iiMin)
            print(iiMax)
            print(jjMin)
            print(jjMax)

            testArea = segm_EXR[iiMin:iiMax, jjMin:jjMax].flatten()
            #
            print(testArea)
            print()
            if 0 in testArea and visibility[joint_nr] == 1 and not (np.size(testArea) == 0):

                print(visibility[joint_nr] == 1)
                print(0 in testArea)
                print(visibility)
                visibility[joint_nr] = 1
            else:

                print(visibility[joint_nr] == 1)
                print(0 in testArea)
                print(visibility)
                visibility[joint_nr] = 0

        visibility_dict = {}
        visibility_dict['visibility'] = visibility

        return np.array(visibility, dtype=bool)



sorted_parts = ['global',
                'leftThigh',
                'rightThigh',
                'spine',
                'leftCalf',
                'rightCalf',
                'spine1',
                'leftFoot',
                'rightFoot',
                'spine2',
                'leftToes',
                'rightToes',
                'neck',
                'leftShoulder',
                'rightShoulder',
                'head',
                'leftUpperArm',
                'rightUpperArm',
                'leftForeArm',
                'rightForeArm',
                'leftHand',
                'rightHand',
                'lIndex0',
                'lMiddle0',
                'lPinky0',
                'lRing0',
                'lThumb0',
                'rIndex0',
                'rMiddle0',
                'rPinky0',
                'rRing0',
                'rThumb0',
                'lIndex1',
                'lMiddle1',
                'lPinky1',
                'lRing1',
                'lThumb1',
                'rIndex1',
                'rMiddle1',
                'rPinky1',
                'rRing1',
                'rThumb1',
                'lIndex2',
                'lMiddle2',
                'lPinky2',
                'lRing2',
                'lThumb2',
                'rIndex2',
                'rMiddle2',
                'rPinky2',
                'rRing2',
                'rThumb2']


part_match = {'root':'root',
              'bone_00':'Pelvis',       # OK
              'bone_01':'L_Hip',        # OK
              'bone_02':'R_Hip',        # OK
              'bone_03':'Spine1',       # OK
              'bone_04':'L_Knee',       # OK
              'bone_05':'R_Knee',       # OK
              'bone_06':'Spine2',       # OK
              'bone_07':'L_Ankle',      # OK
              'bone_08':'R_Ankle',      # OK
              'bone_09':'Spine3',       # OK
              'bone_10':'L_Foot',       # OK
              'bone_11':'R_Foot',       # OK
              'bone_12':'Neck',         # OK
              'bone_13':'L_Collar',     # OK
              'bone_14':'R_Collar',     # OK
              'bone_15':'Head',         # OK
              'bone_16':'L_Shoulder',   # OK
              'bone_17':'R_Shoulder',   # OK
              'bone_18':'L_Elbow',      # OK
              'bone_19':'R_Elbow',      # OK
              'bone_20':'L_Wrist',      # OK
              'bone_21':'R_Wrist',      # OK
             #'bone_22':'L_Hand',
             #'bone_23':'R_Hand'
              'bone_22':'lindex0',
              'bone_23':'lindex1',
              'bone_24':'lindex2',
              'bone_25':'lmiddle0',
              'bone_26':'lmiddle1',
              'bone_27':'lmiddle2',
              'bone_28':'lpinky0',
              'bone_29':'lpinky1',
              'bone_30':'lpinky2',
              'bone_31':'lring0',
              'bone_32':'lring1',
              'bone_33':'lring2',
              'bone_34':'lthumb0',
              'bone_35':'lthumb1',
              'bone_36':'lthumb2',
              'bone_37':'rindex0',
              'bone_38':'rindex1',
              'bone_39':'rindex2',
              'bone_40':'rmiddle0',
              'bone_41':'rmiddle1',
              'bone_42':'rmiddle2',
              'bone_43':'rpinky0',
              'bone_44':'rpinky1',
              'bone_45':'rpinky2',
              'bone_46':'rring0',
              'bone_47':'rring1',
              'bone_48':'rring2',
              'bone_49':'rthumb0',
              'bone_50':'rthumb1',
              'bone_51':'rthumb2'
             }


part2num = {part:(ipart+1) for ipart,part in enumerate(sorted_parts)}


flat_hand_mean = True
body_pose_dofs = 66
ncomps = 12


###### load hand stuff #############################################
with open('./smpl_data/mano_v1_2/models/MANO_LEFT.pkl','rb') as infile:
    manoLeft = pickle.load(infile, encoding='latin1')

hands_componentsl = manoLeft['hands_components']
hands_meanl = np.zeros(
    hands_componentsl.shape[1]) if flat_hand_mean else manoLeft['hands_mean']
hands_coeffsl = manoLeft['hands_coeffs'][:, :ncomps // 2]

with open('./smpl_data/mano_v1_2/models/MANO_RIGHT.pkl','rb') as infile:
    manoRight = pickle.load(infile, encoding='latin1')

hands_componentsr = manoRight['hands_components']
hands_meanr = np.zeros(
    hands_componentsl.shape[1]) if flat_hand_mean else manoRight['hands_mean']
hands_coeffsr = manoRight['hands_coeffs'][:, :ncomps // 2]


selected_components = np.vstack((np.hstack((hands_componentsl[:ncomps//2], np.zeros_like(hands_componentsl[:ncomps//2]))),
                                 np.hstack((np.zeros_like(hands_componentsr[:ncomps//2]), hands_componentsr[:ncomps//2]))))
hands_mean = np.concatenate((hands_meanl, hands_meanr))

########################################################################
pose_coeffs = np.zeros(body_pose_dofs + selected_components.shape[0])
########################################################################

###########################################################################
full_hand_pose = pose_coeffs[body_pose_dofs:(body_pose_dofs+ncomps)].dot(selected_components)
###########################################################################
mixed_body_full_hand_pose = np.concatenate((pose_coeffs[:body_pose_dofs], hands_mean + full_hand_pose))


if __name__ == '__main__':

    import sys
    import h5py
    import gzip
    from pickle import load
    from os import makedirs, system
    logger.info('starting')
    #################################################################
    print(len(sys.argv))
    for ii, arg in enumerate(sys.argv):
        if arg == '--':
            print(ii, arg, '   :)')
        else:
            if ii == 7 or ii == 8:
                print(ii, arg, bool(int(arg)))
            else:
                print(ii, arg)
    ##########################################################################
    print('\n\n')
    logger.info(len(sys.argv))
    if sys.argv[-2] == '--':  # 0: Train
        print('LOCAL SCRIPT    --->  Will take into account the local TRAIN/VALID/TEST flag')  # 1: Valid
        idx = int(sys.argv[-1])  # 2: Test
    elif sys.argv[-4] == '--' and sys.argv[-2] != '--':
        logger.info('CLUSTER SCRIPT')
        print('CLUSTER SCRIPT  --->  Will overwrite the local TRAIN/VALID/TEST flag')
        idx = int(sys.argv[-3])

        ######################################################
        def split_ID_2_string(splitID):
            switcher = {
                0: 'train',
                1: 'valid',
                2: 'test',
            }
            return switcher.get(splitID, 'error')
        ######################################################
        split = split_ID_2_string(int(sys.argv[-2]))

        if split == 'error':
            print('\n\n\n')
            print('not defined - split argument')
            print('\n\n\n')
            exit()
        ######################################################

    idxOffset = int(sys.argv[-1])
    idx = idx + idxOffset
    nrOfPeople = 0
    if nrOfPeople == 0:
            logger.info('draw poisson sample')
            nrOfPeople = np.random.poisson(4, (1,))[0]
    if nrOfPeople <= 1:
        nrOfPeople = 2
    print('\n\n\n')
    print('*** split  = ', split,  '***')
    logger.info(str(nrOfPeople))


    #####################################################
    #####################################################
    with open(fingertipVerticesPATH) as data_file:
        fingertipVerticesDICT = json.load(data_file)
    #####################################################
    #####################################################

    #####################################################
    #####################################################
    shape = np.zeros(shape_totalCoeffs)
    #####################################################
    segmented_materials = True
    #####################################################

    ####################################################
    scene = bpy.data.scenes['Scene']
    ####################################################
    scene.render.engine = 'CYCLES'
    bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True
    ####################################################

    ###################################################################
    ###################################################################
    smpl_data_MOCAP = np.load(FILE_smpl_data_MOCAP, encoding='latin1',
                              allow_pickle=True)
    smpl_data_OTHER = np.load(FILE_smpl_data_OTHER)
    ###################################################################
    smpl_data___joint_regressorr = smpl_data_OTHER['joint_regressor']
    smpl_data___regression_verts = smpl_data_OTHER['regression_verts']
    ###################################################################
    ###################################################################
    PATH_out = join(PATH_out, dataset_path_prefix, split)
    PATH_tmp = join(PATH_tmp, dataset_path_prefix, split)

    from os import makedirs
    if not exists(PATH_out):
        makedirs(PATH_out)
    if not exists(PATH_tmp):
        makedirs(PATH_tmp)
    #######################################
    #######################################

    SMPLH_joint_regressorr = np.load(
        join(data_folder, 'SMPLH_data_joint_regressor.npy'))
    SMPLH_regression_verts = np.load(
        join(data_folder, 'SMPLH_data_regression_verts.npy'))

    smpl_data___joint_regressorr = SMPLH_joint_regressorr.copy()
    smpl_data___regression_verts = SMPLH_regression_verts.copy()

    ##############################################################
    #create some empty lists to store everything necessary for multi persons
    obs                 = []
    obnames             = []
    arm_obs             = []
    data                = []
    shapes              = []
    random_shapes       = []
    harmonics_scripts   = []
    materials           = []
    materialID_2_part   = []
    gender              = []
    cloth_img           = []
    cloth_img_PATHs     = []
    real_img            = []
    txt_paths           = []
    fend                = np.zeros((nrOfPeople, 1))
    N                   = np.zeros((nrOfPeople, ))
    random_zrot         = 2 * np.pi * np.random.rand(nrOfPeople)
    lower_visibility_line = None
    stepsize            = np.ones((nrOfPeople,), dtype=int)
    dataNames           = []
    annolist            = []
    head_anno           = []
    image_anno          = []
    objectpos_anno         = []
    scale_anno          = []

    ######################################################
    def gender_ID_2_string(splitID):
        switcher = {
            0: 'male',
            1: 'female',
        }
        return switcher.get(splitID, 'error')
    ######################################################
    ######################################################

    for i in range(nrOfPeople):
        gender.append( gender_ID_2_string(choice([0,1])) )
        ######################################################
        ######################################################
        if gender[i] == 'error':
            print('\n\n\n')
            print('not defined - gender argument')
            print('\n\n\n')
            exit()
        print('*** gender = ', gender[i], '***')
        print('\n\n\n')

        logger.info('init_scene')
        #################################################################
        train_idx = loadmat('./resources/annoIdx_train.mat')
        train_idx = train_idx['annoIdx_train'] - 1 # matlab idx to python conversion
        train_idx = np.array(train_idx, dtype=int)
        train_idx = train_idx.squeeze()

        annolist_real = loadmat('./resources/annolist_dataset_v12.mat', struct_as_record=False,squeeze_me=True) #Todo:download annolist from http://human-pose.mpi-inf.mpg.de/#download
        annolist_real = annolist_real['annolist']
        annolist_real = annolist_real[train_idx] #sort out all validation and test annotations

        bg_path = join(img_base_path, annolist_real[idx].image.name)
        #####################################################
        bg_img_TMP = cv2.imread(bg_path)

        PATH_bg_TMP = join(PATH_tmp, join('bgimg',str(idx), 'bg_TMP.jpg'))
        if not exists(dirname(PATH_bg_TMP)):
            makedirs(dirname(PATH_bg_TMP))
        cv2.imwrite(PATH_bg_TMP, bg_img_TMP)
        bg_img = bpy.data.images.load(PATH_bg_TMP)

        res[0] = bg_img.size[0]
        res[1] = bg_img.size[1]

        #################################################################
        if lower_visibility_line is None:
            ob, obname, arm_ob, cam_ob, plane_ob, teximage, lower_visibility_line = init_scene(gender[i], i, lower_visibility_line=None)
        else:
            ob, obname, arm_ob, cam_ob, plane_ob, teximage, lower_visibility_line = init_scene(gender[i], i,lower_visibility_line=lower_visibility_line)
        obs.append( ob )
        obnames.append( obname )
        arm_obs.append( arm_ob )
        if not(teximage is None):
            texim = teximage

        ####################################
        intrinsic = cam_compute_intrinsic()
        ####################################


        split_proportion_TRAIN = 0.8
        split_proportion_VALID = 0.1
        split_proportion_TESTT = 0.1
        assert (split_proportion_TRAIN + split_proportion_VALID + split_proportion_TESTT) == 1.0

        logger.info('load body data')
        #################################################################################################
        cmu_parms, fshapes, cmu_frames, cmu_stepsize = load_body_data(smpl_data_MOCAP, smpl_data_OTHER, ob, obname, gender=gender[i], i=i)
        shuffle_idx = np.load('./resources/random_mocap_order_'+mocapDataName+'.npy')

        #################################################################################################
        fshapes_NUMB = fshapes.shape[0]
        xxxShapesTRN = int(round(fshapes_NUMB * (split_proportion_TRAIN)))
        xxxShapesVLD = int(round(fshapes_NUMB * (split_proportion_TRAIN + split_proportion_VALID)))
       #
        if split == 'train':
            fshapes = fshapes[:xxxShapesTRN            , :]
        elif split == 'valid':
            fshapes = fshapes[xxxShapesTRN:xxxShapesVLD, :]
        elif split == 'test':
            fshapes = fshapes[xxxShapesVLD:            , :]
        else:
            print('\n\n\n')
            print('not defined - split !@#')
            print('\n\n\n')
            exit()

        cmu_parms_keyz = np.array(sorted(cmu_parms.keys()))[shuffle_idx] #just to make sure that mocap data is randomly distributed across training and test sets
        cmu_parms_NUMB = len(cmu_parms_keyz)
        #
        xxxCmuTRN = int(round(cmu_parms_NUMB * (split_proportion_TRAIN)))
        xxxCmuVLD = int(round(cmu_parms_NUMB * (split_proportion_TRAIN + split_proportion_VALID)))

        if split == 'train':
            names = cmu_parms_keyz[:xxxCmuTRN]
        elif split == 'valid':
            names = cmu_parms_keyz[xxxCmuTRN:xxxCmuVLD]
        elif split == 'test':
            names = cmu_parms_keyz[xxxCmuVLD:]
        else:
            print('\n\n\n')
            print('not defined - split !@#')
            print('\n\n\n')
            exit()
        #
        if i == 0 :
            name = sorted(names)[idx % len(names)]
            filename = str(idx)
        else:
            frame_ar = np.array([cmu_frames[key]/cmu_stepsize[key] for key in names])
            csum = np.cumsum(frame_ar / np.sum(frame_ar))
            dice = np.random.rand(1)
            rand_seq = np.argmax(csum > dice) # get first occurance of this condition

            name = names[rand_seq]
        dataNames.append(name)


        if DBG_print_MOCAP_keys:
            for idd, kk in enumerate(sorted(cmu_parms.keys())):
                print('%04d  -  %-20s' % (idd, kk))
            print(' ')
            print('gender         = %s' % [i])
            print('len(cmu_parms) = %s' % len(cmu_parms))
            print('idx            = %s' % idx)
            print('idx %% len     = %s' % (idx % len(cmu_parms)))
            # for nn in name:
            #     print(nn)
            print('\n\n\n')
            print('DBG_print_MOCAP_keys = True  -->  Exiting now !!!')
            print('\n\n\n')
            exit(1)
        ################################################################################################################
        ################################################################################################################

        ###############################################
        ###############################################
        hand_poses = load_hand_poses(PATH_MoCap_SMPLH)
        ###############################################
        ###############################################
        hand_poses_NUMB = len(hand_poses)
        #
        xxxHPosesTRN = int(round(hand_poses_NUMB * (split_proportion_TRAIN)))
        xxxHPosesVLD = int(round(hand_poses_NUMB * (split_proportion_TRAIN + split_proportion_VALID)))

        if split == 'train':
            hand_poses = hand_poses[:xxxHPosesTRN]
        elif split == 'valid':
            hand_poses = hand_poses[xxxHPosesTRN:xxxHPosesVLD]
        elif split == 'test':
            hand_poses = hand_poses[xxxHPosesVLD:]
        else:
            print('\n\n\n')
            print('not defined - split @#$')
            print('\n\n\n')
            exit()
        ###############################################
        ###############################################


        caesar_txt_paths = sorted(glob((PATHs_texture_participants.replace('<CEASAR>', 'grey')).replace('<GENDER>', gender[i])))
        noncaesar_txt_paths = sorted(glob((PATHs_texture_participants.replace('<CEASAR>', 'nongrey')).replace('<GENDER>', gender[i])))
        #
        ncaesar = len(caesar_txt_paths)
        nnoncaesar = len(noncaesar_txt_paths)
        print(noncaesar_txt_paths)
        #
        xxxCeasarTRN = int(round(ncaesar * (split_proportion_TRAIN)))
        xxxCeasarVLD = int(round(ncaesar * (split_proportion_TRAIN + split_proportion_VALID)))

        #
        xxxNonCeasarTRN = int(round(nnoncaesar * (split_proportion_TRAIN)))
        xxxNonCeasarVLD = int(round(nnoncaesar * (split_proportion_TRAIN + split_proportion_VALID)))

        non_ceasar_ceasar_ratio = 4
        if ncaesar >= nnoncaesar:
            if split == 'train':
                txt_paths.append(caesar_txt_paths[:xxxCeasarTRN]             + (non_ceasar_ceasar_ratio*(ncaesar // nnoncaesar) * noncaesar_txt_paths[:xxxNonCeasarTRN]) )
            elif split == 'valid':
                txt_paths.append( caesar_txt_paths[xxxCeasarTRN:xxxCeasarVLD] + (non_ceasar_ceasar_ratio*(ncaesar // nnoncaesar) * noncaesar_txt_paths[xxxNonCeasarTRN:xxxNonCeasarVLD]) )
            elif split == 'test':
                txt_paths.append( caesar_txt_paths[xxxCeasarVLD:]             + (non_ceasar_ceasar_ratio*(ncaesar // nnoncaesar) * noncaesar_txt_paths[xxxNonCeasarVLD:]) )
            else:
                print('\n\n\n')
                print('not defined - split $%^')
                print('\n\n\n')
                exit()
        else:
            if split == 'train':
                txt_paths.append( int(np.round(1/non_ceasar_ceasar_ratio*(nnoncaesar // ncaesar))) * caesar_txt_paths[:xxxCeasarTRN] + noncaesar_txt_paths[:xxxNonCeasarTRN] )
            elif split == 'valid':
                txt_paths.append( int(np.round(1/non_ceasar_ceasar_ratio*(nnoncaesar // ncaesar))) * caesar_txt_paths[xxxCeasarTRN:xxxCeasarVLD] + noncaesar_txt_paths[xxxNonCeasarTRN:xxxNonCeasarVLD] )
            elif split == 'test':
                txt_paths.append( int(np.round(1/non_ceasar_ceasar_ratio*(nnoncaesar // ncaesar))) * caesar_txt_paths[xxxCeasarVLD:] + noncaesar_txt_paths[xxxNonCeasarVLD:] )
            else:
                print('\n\n\n')
                print('not defined - split $%^')
                print('\n\n\n')
                exit()

        #
        ####################################################
        # PUT INITIAL CLOTHING
        cloth_img_name = choice(txt_paths[i])
        cloth_img.append(  bpy.data.images.load(cloth_img_name) )
        ####################################################
        # grab random textures from existing participants
        real_img.append( cloth_img[i] )
        # this texture holds the part segmentation.
        # Ideally it should be directly done with face colors
        ####################################################
        ####################################################

        ####################################################
        mat_tree = bpy.data.materials['Material'].node_tree
        ####################################################
        # Create copy-spher.harm. directory if not exist

        ####################################################
        PATH_OUT_sh = join(PATH_out, filename, 'spher_harm', 'sh.osl')  #
        ####################################################
        if not exists(dirname(PATH_OUT_sh)):
            makedirs(dirname(PATH_OUT_sh))
        system('cp %s %s' % (FILE_sh_original.replace(' ', '\ '), PATH_OUT_sh.replace(' ', '\ ')))
        ####################################################
        create_sh_material(mat_tree, PATH_OUT_sh, img=real_img[i])
        ####################################################
        res_paths = create_composite_nodes(scene.node_tree, filename, img=bg_img)
        ####################################################

        setState0()
        ob.select = True
        bpy.context.scene.objects.active = ob
        #
        # create material segmentation
        if segmented_materials:

            mat, matID_2_part = create_segmentation(ob, i)
            materials.append( mat )
            materialID_2_part.append( matID_2_part )


            prob_dressed = {'global':           .01,
                            'leftThigh':        .9,
                            'rightThigh':       .9,
                            'spine':            .9,
                            'leftCalf':         .5,
                            'rightCalf':        .5,
                            'spine1':           .9,
                            'leftFoot':         .9,
                            'rightFoot':        .9,
                            'spine2':           .9,
                            'leftToes':         .9,
                            'rightToes':        .9,
                            'neck':             .01,
                            'leftShoulder':     .8,
                            'rightShoulder':    .8,
                            'head':             .01,
                            'leftUpperArm':     .5,
                            'rightUpperArm':    .5,
                            'leftForeArm':      .5,
                            'rightForeArm':     .5,
                            'leftHand':         .01,
                            'rightHand':        .01,
                            'lIndex0':          .01,
                            'lMiddle0':         .01,
                            'lPinky0':          .01,
                            'lRing0':           .01,
                            'lThumb0':          .01,
                            'rIndex0':          .01,
                            'rMiddle0':         .01,
                            'rPinky0':          .01,
                            'rRing0':           .01,
                            'rThumb0':          .01,
                            'lIndex1':          .01,
                            'lMiddle1':         .01,
                            'lPinky1':          .01,
                            'lRing1':           .01,
                            'lThumb1':          .01,
                            'rIndex1':          .01,
                            'rMiddle1':         .01,
                            'rPinky1':          .01,
                            'rRing1':           .01,
                            'rThumb1':          .01,
                            'lIndex2':          .01,
                            'lMiddle2':         .01,
                            'lPinky2':          .01,
                            'lRing2':           .01,
                            'lThumb2':          .01,
                            'rIndex2':          .01,
                            'rMiddle2':         .01,
                            'rPinky2':          .01,
                            'rRing2':           .01,
                            'rThumb2':          .01}
        else:
            materials.append( {'FullBody': bpy.data.materials['Material']} )
            prob_dressed = {'FullBody': 1.}

        ##########################################################################################################################
        orig_pelvis_loc = (arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_'+part_match['bone_00']].head.copy()) - Vector((-1., 1., 1.))
        ##########################################################################################################################
        orig_cam_loc = cam_ob.location.copy()
        ##########################################################################################################################

        ###################################################################
        ###################################################################
        beta_stds = np.load(join(data_folder, ('%s_beta_stds.npy' % gender[i])))  #
        ####################################################
        ####################################################
        random_shapes.append( lambda std:np.concatenate((np.random.uniform(-std, std, size=(shape_ndofs,))*beta_stds[:shape_ndofs], np.zeros(shape_totalCoeffs-shape_ndofs)))  )# 64

        scene.objects.active = arm_obs[i]

        orig_trans = np.asarray(arm_obs[i].pose.bones[obname+'_'+part_match['bone_00']].location).copy()

        # The spherical harmonics material needs a script to be loaded
        scs = []
        for mname, material in materials[i].items():
            scs.append(material.node_tree.nodes['Script'])
            scs[-1].filepath = PATH_OUT_sh
            scs[-1].update()
        harmonics_scripts.append(scs)
        flip_coin = lambda prob: np.random.rand() < prob
        data.append( cmu_parms[name] )

        stepsize[i] = int(cmu_stepsize[name]) * stepsizeFactor
        N[i] = len(data[i]['poses'][::stepsize[i]])

        print('================= %d ======================' % N[i])
        irot2name = {i:'txt_images_%03d' % (i*90) for i in range(1)}
        #####################################################

        get_real_frame = lambda ifr: ifr
        random_camera_trans = Vector((0.,0.,0.))
        random_camera_rotX = 0
        random_camera_rotY = 0
        random_camera_rotZ = 0
        random_camera_rotX_ACCUM = 0.
        random_camera_rotY_ACCUM = 0.
        random_camera_rotZ_ACCUM = 0.
        reset_loc = False
        flip_coin_perCHUNK = False
    ############################################################
    ############################################################

    # create a sequence of frames_per_shape with a single shape and rotation, with keyframes
    # keyframe animation is important to obtain the flow
    min_N_idx   = np.argmin(N)
    min_N       = N[min_N_idx]

    ishapes     = range(int(np.ceil(float(nr_of_frames_per_bg) / float(frames_per_shape))))
    fbegin      = np.zeros((len(ishapes), nrOfPeople))
    fend        = np.zeros((len(ishapes), nrOfPeople))
    reset_loc   = np.zeros((nrOfPeople,1), dtype=bool)
    init_trans  = []
    rand_start_frame = np.zeros((nrOfPeople,))

    for i, n in enumerate(N):
        if n == min_N:
            rand_start_frame[i] = 0
        else:
            rand_start_frame[i] = np.random.randint(0, n - min_N)

    print(N)
    print(dataNames)
    ############################################################
    logger.info('start ishapes loop')
    for ishape in ishapes:

        for i in range(nrOfPeople):
            bpy.ops.object.select_all(action='DESELECT')
            arm_obs[i].select = True
            bpy.context.scene.objects.active = arm_obs[i]
            if RANDOM_SHAPE_per_CHUNK:  # ONLY PER CHUNK !!!
                if i == 0:
                    shapes = []
                shapes.append( set_shape(RANDOM_SHAPE, RANDOM_SHAPE_mode, i) )     # RANDOM shape

            #####################################################
            curr_shape = reset_joint_positions(shapes[i], obs[i], arm_obs[i], obnames[i], scene, cam_ob, smpl_data___regression_verts, smpl_data___joint_regressorr, n_bones, total_pose_DOFs)
            if RANDOM_ROT_Z:        # RANDOM rot
                random_zrot[i] = 2 * np.pi * np.random.rand()
        bpy.ops.object.select_all(action='DESELECT')

        def degree2rad(deg):
            rad = deg * np.pi / 180.
            return rad


        if RANDOM_CAMERA_JITTER:
            flip_coin_perCHUNK = flip_coin(.3)
            random_camera_trans = .005*np.random.randn(3)
        if RANDOM_POSITION_perCHUNCK and not(ishape == ishapes[0]):
            x_range = get_x_range(cam_ob, plane_ob)
            for person in arm_obs:
                bpy.ops.object.select_all(action='DESELECT')
                person.select = True
                bpy.context.scene.objects.active = person
                translation = random_placement_in_visibl_area(cam_ob, x_range)
                vector_move(person, translation)

        for i in range(nrOfPeople):
            if N[i] == min_N:
                fbegin[ishape, i] = ishape * stepsize[i] * frames_per_shape
                fend[ishape, i ] = min((ishape + 1) * stepsize[i] * frames_per_shape, len(data[i]['poses']))
            else:
                fbegin[ishape, i] = ishape * stepsize[i] * frames_per_shape + rand_start_frame[i]
                fend[ishape, i] = (ishape + 1) * stepsize[i] * frames_per_shape + rand_start_frame[i]

        for arm_ob in arm_obs:
            arm_ob.animation_data_clear()
        cam_ob.animation_data_clear()

        for i in range(nrOfPeople):
            if RANDOM_TXT_per_CHUNK:
                cloth_img[i], cloth_img_PATH = set_txt(RANDOM_TXT, cloth_img[i], txt_paths[i], i)  # handles also NON-RANDOM case
                cloth_img_PATHs.append(cloth_img_PATH)


        if RANDOM_LIGHTS_per_CHUNK:
            sh_coeffs = set_lights(RANDOM_LIGHTS)

        #############################################
        if MODE_NODES == 'NEW':
            blur = scene.node_tree.nodes['Blur']
            if RANDOM_Pxl_BLUR_SIZE:
                blur.size_x = int(np.round(1.0 * np.random.randn()))
                blur.size_y = int(np.round(1.0 * np.random.randn()))
            else:
                blur.size_x = 0.
                blur.size_y = 0.


        ##########################################################################
        ##########################################################################
        vblur_factor = np.random.normal(vblur_factor_Miu, vblur_factor_Std)
        if not USE_MOTION_BLUR:
            vblur_factor = 0.
        ##########################################################################
        scene.node_tree.nodes['Vector Blur'].factor = vblur_factor
        ##########################################################################
        ##########################################################################

        allHandsOnly_poses = []
        handsOnly_poses = []
        for personIDX in range(nrOfPeople):
            handSeqID = choice(range(len(hand_poses)))
            lennn = len(hand_poses[handSeqID])
            #
            sampledHandFrameIDs = range(lennn)[::stepsize_hands]
            startSamplingFrID = choice(range(len(sampledHandFrameIDs) - frames_per_shape + 1))
            #
            for sampledHandFrameID in sampledHandFrameIDs:
                if RANDOM_HAND_POSE:
                    curr_hand_pose = hand_poses[handSeqID][sampledHandFrameID]
                else:
                    handSeqID = 0
                    curr_hand_pose = hand_poses[handSeqID][0]
                if DBG_MODE_ENFORCE_FLAT_HAND:
                    curr_hand_pose = np.zeros_like(hand_poses[0][0])
                #
                handsOnly_poses.append(curr_hand_pose)
            allHandsOnly_poses.append( handsOnly_poses )

        ###########################
        STORED_ITER_seq_frame = []
        STORED_ITER_pose      = []
        STORED_ITER_trans     = []
        ###########################

        N_min = np.min(N)
        mins = np.zeros((nrOfPeople,))
        for i in range(nrOfPeople):
            mins[i] = len(np.arange(fbegin[ishape, i], fend[ishape, i], stepsize[i]))
        N_min = min(mins)
        ###########################

        print('done with allocating - searching for a valid configuration without collisions')
        print(fbegin)
        print(fend)
        print(stepsize)
        print(mins)
        ###########################
        collision = True
        # repeat until configuration without collisions is found
        combs = np.linspace(0, nrOfPeople - 1, nrOfPeople, dtype=int)
        pairs = list(combinations(combs, 2))

        logger.info('start collision free assembling')
        while collision:
            logger.info('not found yet')
            collisions_per_person       = np.zeros((nrOfPeople, 1), dtype=int)
            collisions_per_pair         = np.zeros((len(pairs), 1), dtype=int)
            plane_collision             = np.zeros((nrOfPeople, 1), dtype=bool)
            STORED_ITER_seq_frame_temp  = []
            STORED_ITER_pose_temp       = []
            STORED_ITER_trans_temp      = []
            print(N_min)
            print(stepsize)
            for seq_frame in range(int(N_min)):
                # get empty lists for collision detection of meshes
                meshes          = []
                mesh_vertices   = []
                start_frame     = np.zeros((len(arm_obs),), dtype=int)
                pose_list       = []
                trans_list      = []
                print('seqFrame=' +str(seq_frame))
                ###############################################
                iframe = seq_frame + ishape * frames_per_shape
                ###############################################
                scene.frame_set(get_real_frame(seq_frame))
                ###############################################

                for i, (arm_ob, pose_handsOnly) in enumerate(zip(arm_obs, allHandsOnly_poses)):
                    bpy.ops.object.select_all(action='DESELECT')
                    arm_ob.select                       = True
                    bpy.context.scene.objects.active    = arm_ob
                    obs[i].select                       = True
                    bpy.context.scene.objects.active    = obs[i]
                    pose_handsOnly                      = np.array(pose_handsOnly[i])
                    pose_list.append( data[i]['poses'][int(fbegin[ishape, i]):int(fend[ishape, i]):stepsize[i]][seq_frame] )
                    trans_list.append( data[i]['trans'][int(fbegin[ishape, i]):int(fend[ishape, i]):stepsize[i]][seq_frame] )

                    if len(pose_list[i]) < total_pose_DOFs:
                        pose_list[i] = np.hstack((pose_list[i][:body_pose_dofs], pose_handsOnly))
                        #
                        assert(pose_list[i].shape[0] == body_pose_dofs + ncomps)
                        #
                        if DBG_MODE_ENFORCE_POSE_ZERO:
                            pose_list[i][3:] = 0

                    if seq_frame == 0:
                        init_trans.append( Vector(trans_list[i]) + arm_ob.pose.bones[0].tail )
                    init_trans[i] = apply_trans_pose_shape(Vector(trans_list[i]), pose_list[i], shapes[i], obs[i], arm_obs[i], obnames[i], scene, cam_ob, n_bones, get_real_frame(seq_frame), init_trans[i], True)
                    ###############################

                    arm_obs[i].pose.bones[obnames[i]+'_'+part_match['root']].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot[i]), 'XYZ'))
                    arm_obs[i].pose.bones[obnames[i]+'_'+part_match['root']].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))

                    meshes.append( obs[i].to_mesh(scene, True, 'PREVIEW') )
                    mesh_vertices.append( [obs[i].matrix_world * vert.co for vert in meshes[i].vertices] )

                    scene.update()

                if RANDOM_CAMERA_JITTER and flip_coin_perCHUNK:
                    cam_ob_origloc = cam_ob.location.copy()
                    cam_ob.location = cam_ob_origloc + Vector(
                        random_camera_trans)
                    cam_ob.keyframe_insert('location',
                                           frame=get_real_frame(seq_frame))
                    cam_ob.location = cam_ob_origloc


                STORED_ITER_seq_frame_temp.append(seq_frame)
                STORED_ITER_pose_temp.append(pose_list)
                STORED_ITER_trans_temp.append(trans_list)
                #########################################
                #########################################
                #check for collisions
                x_range = get_x_range(cam_ob, plane_ob)

                frame_collisions, collisions_per_ob, plane_collision_seq = collisionDetector.mesh_collision(mesh_vertices, meshes, pairs, min_max_x=(x_range[0], x_range[1]))
                collisions_per_pair += np.array(collisions_per_ob, dtype=int)
                plane_collision += plane_collision_seq
                print(frame_collisions)
                print(plane_collision)
                #delete all meshes created for collision detection
                for mesh in meshes:
                    bpy.data.meshes.remove(mesh)
            #########################################
            collision_count = np.sum(collisions_per_pair)
            #########################################

            #resolve colision if any collision between a person with a person or a person with the background plane was detected
            if collision_count > 0 or any(plane_collision):
                for seq_frame in range(int(N_min)):
                    scene.frame_set(get_real_frame(seq_frame))
                    cam_ob.animation_data_clear()
                    for arm_ob in arm_obs:
                        arm_ob.animation_data_clear()

                ######################################
                ################################
                #loop over pairs of people and replace them until no more collisions from last animation
                while (collision_count > 0 or any(plane_collision)):
                    print(collisions_per_pair)
                    # select one person from pair with largest number of collisions
                    if any( plane_collision ):
                        for j, pl_col in enumerate(plane_collision):
                            if pl_col:
                                max_idx = j
                    else:
                        pair_idx            = np.argmax(collisions_per_pair)
                        max_idx             = pairs[pair_idx][np.random.randint(0, 2)]

                    colliding_person = arm_obs[max_idx]
                    bpy.ops.object.select_all(action='DESELECT')
                    colliding_person.select             = True
                    bpy.context.scene.objects.active    = colliding_person

                    # set new initial pose to a new random position in the visible area
                    x_range = get_x_range(cam_ob, plane_ob)
                    translation = random_placement_in_visibl_area(cam_ob, x_range)
                    vector_move(colliding_person, translation)
                    print(colliding_person.name)

                    ################################
                    #update collision counts
                    for j, tpl in enumerate(pairs):
                        if max_idx in tpl:
                            collision_count         -= collisions_per_pair[j]
                            collisions_per_pair[j]   = 0
                            plane_collision[max_idx] = False
            else:
                collision = False


        for temp_frame, temp_pose, temp_trans in zip(STORED_ITER_seq_frame_temp,
                                                     STORED_ITER_pose_temp,
                                                     STORED_ITER_trans_temp):
            STORED_ITER_seq_frame.append(temp_frame)
            STORED_ITER_pose.append(temp_pose)
            STORED_ITER_trans.append(temp_trans)
        #########################################

        print(STORED_ITER_seq_frame)
    #####################################################
    #####################################################
    # iterate over the keyframes (here set manually) and render
        for _idx_ in range(len(STORED_ITER_seq_frame)):
            ###################################################    ***
            seq_frame = STORED_ITER_seq_frame[_idx_]
            pose      = STORED_ITER_pose[_idx_]
            trans     = STORED_ITER_trans[_idx_]

            ###############################################
            iframe = seq_frame + ishape * frames_per_shape
            ###############################################
            scene.frame_set(get_real_frame(seq_frame))
            ###############################################


            PATH_INN_depth_EXR          = join(           res_paths['depth'],               '%05d.exr'  % (seq_frame))
            PATH_OUT_depth_EXR          = join(PATH_out, join(filename, 'depth_EXR',            '%05d.exr'  % (iframe)))

            PATH_INN_normals_EXR        = join(           res_paths['normal'],              '%05d.exr'  % (seq_frame))
            PATH_OUT_normals_EXR        = join(PATH_out, join(filename, 'normal_EXR',           '%05d.exr'  % (iframe)))
            PATH_OUT_normals_VIZ        = join(PATH_out, join(filename, 'normal_VIZ',           '%05d.png'  % (iframe)))

            PATH_INN_flow_EXR           = join(           res_paths['flow'],                '%05d.exr'  % (seq_frame))
            PATH_OUT_flow_FLOW_fake     = join(PATH_out, join(filename, 'flow_fake',            '%05d.flo'  % (iframe)))
            PATH_OUT_flow_FLOW_real     = join(PATH_out, join(filename, 'flow',                 '%05d.flo'  % (iframe)))
            PATH_OUT_flow_VIZ           = join(PATH_out, join(filename, 'flow_VIZ',             '%05d.png'  % (iframe)))

            PATH_INN_segm_EXR           = join(           res_paths['segm'],                '%05d.exr'  % (seq_frame))
            PATH_OUT_segm_EXR           = join(PATH_out, join(filename, 'segm_EXR',             '%05d.exr'  % (iframe)))

            PATH_INN_objectID_EXR       = join(           res_paths['object_Id'],           '%05d.exr' % (seq_frame))
            PATH_OUT_objectID_EXR       = join(PATH_out, join(filename, 'objectId_EXR',         '%05d.exr' % (iframe)))

            PATH_OUT_objectID_obname = join(PATH_out, join(filename, 'objectId_obname',         '%05d.exr' % (iframe)))

            PATH_OUT_depth_VIZ          = join(PATH_out, join(filename, 'depth_VIZ',            '%05d.png'  % (iframe)))
            PATH_OUT_segm_PNG           = join(PATH_out, join(filename, 'segm_PNG',             '%05d.png'  % (iframe)))
            PATH_OUT_segm_VIZ           = join(PATH_out, join(filename, 'segm_VIZ',             '%05d.png'  % (iframe)))  # (name, iframe)))
            PATH_OUT_full_textBG        = join(PATH_out, join(filename, 'composition',          '%05d.png'  % (iframe)))  # (name, iframe)))
            PATH_OUT_pose_joints_2d     = join(PATH_out, join(filename, 'pose_joints_2d',       '%05d.npy'  % (iframe)))  # (name, iframe)))
            PATH_OUT_pose_joints_2d_VIZ = join(PATH_out, join(filename, 'pose_joints_2d_VIZ',   '%05d.png'  % (iframe)))  # (name, iframe)))
            PATH_OUT_pose_joints_3d     = join(PATH_out, join(filename, 'pose_joints_3d',       '%05d.npy'  % (iframe)))  # (name, iframe)))
            PATH_OUT_pose_joints_3d_VIZ = join(PATH_out, join(filename, 'pose_joints_3d_VIZ',   '%05d.png'  % (iframe)))  # (name, iframe)))
            PATH_OUT_pose_joints_VSBL   = join(PATH_out, join(filename, 'pose_joints_VSBL',     '%05d.npy'  % (iframe)))  # (name, iframe)))
            PATH_OUT_shape              = join(PATH_out, join(filename, 'shapes',         '%05d.npy'  % (iframe)))  # (name, iframe)))
            PATH_OUT_pose_fullPCA       = join(PATH_out, join(filename, 'pose_coeffs_fullPCA',  '%05d.npy'  % (iframe)))  # (name, iframe)))
            PATH_OUT_pose_fullFull      = join(PATH_out, join(filename, 'pose_coeffs_fullFull', '%05d.npy'  % (iframe)))  # (name, iframe)))

            PATH_OUT_BB_handsBody_crn   = join(PATH_out, join(filename, 'cropHandsBody_BBs',    '%05d.npy'  % (iframe)))  # FUSE-SCRIPTS  # out_BBcrn_BASEPATH =

            PATH_OUT_gender             = join(PATH_out, join(filename, 'gender',               '%05d.txt'  % (iframe)))

           #PATH_OUT_HandVisibilityLRtxt= join(PATH_out, join(name, 'hand_visibility',      '%05d.txt'  % (iframe)))
            PATH_OUT_HandVisibilityLR   = join(PATH_out, join(filename, 'hand_visibility',      '%05d.npy'  % (iframe)))

            PATH_OUT_DBG                = join(PATH_out, join(filename, 'azzzDBG',              '%05d.txt'  % (iframe)))

            PATH_OUT_Cam_RT_4x4         = join(PATH_out, join(filename, 'camera_RT_4x4',        '%05d.npy'  % (iframe)))
            PATH_OUT_Cam_RT_4x4_txt     = join(PATH_out, join(filename, 'camera_RT_4x4',        '%05d.txt'  % (iframe)))

            PATH_OUT_Subj_Pelvis_T      = join(PATH_out, join(filename, 'subj_pelvis_T',        '%05d.npy'  % (iframe)))
            PATH_OUT_Subj_ZRot          = join(PATH_out, join(filename, 'subj_ZRot',            '%05d.npy'  % (iframe)))

            PATH_OUT_BLUR_parms         = join(PATH_out, join(filename, 'blur_parms',           '%05d.npy'  % (iframe)))
            PATH_OUT_BLUR_parmsTxt      = join(PATH_out, join(filename, 'blur_parms',           '%05d.txt'  % (iframe)))

            PATH_OUT_bg_img_PATH        = join(PATH_out, join(filename, 'img_path_BGround',     '%05d.txt'  % (iframe)))
            PATH_OUT_txt_img_PATH       = join(PATH_out, join(filename, 'img_path_Texture',     '%05d.txt'  % (iframe)))

            PATH_OUT_scale              = join(PATH_out, join(filename, 'scale',                 '%05d' % (iframe)))
            PATH_OUT_bbx_head           = join(PATH_out, join(filename, 'bbx_head', '%05d' % (iframe)))
            PATH_OUT_bbx_head_img       = join(PATH_out, join(filename, 'bbx_head', '%05d' % (iframe)))
            PATH_OUT_ANNOLIST           = join(PATH_out, join(filename, 'annolist'))
            PATH_OUT_ANNOLIST_HEAD      = join(PATH_out, join(filename, 'annolist_head'))
            PATH_OUT_ANNOLIST_img       = join(PATH_out, join(filename, 'annolist_img'))
            PATH_OUT_ANNOLIST_objpos = join(PATH_out, join(filename, 'annolist_objpos'))
            PATH_OUT_ANNOLIST_scale = join(PATH_out, join(filename, 'annolist_scale'))

            PATH_OUT_joint_visibility = join(PATH_out, join(filename, 'joint_visibility', '%05d.jpg' % (iframe)))
            #

            if not exists(dirname(PATH_OUT_full_textBG)):
                makedirs(dirname(PATH_OUT_full_textBG))

            if not exists(dirname(PATH_OUT_pose_joints_2d)):
                makedirs(dirname(PATH_OUT_pose_joints_2d))
            if not exists(dirname(PATH_OUT_pose_joints_2d_VIZ)):
                makedirs(dirname(PATH_OUT_pose_joints_2d_VIZ))

            if not exists(dirname(PATH_OUT_pose_joints_3d)):
                makedirs(dirname(PATH_OUT_pose_joints_3d))
            if not exists(dirname(PATH_OUT_pose_joints_3d_VIZ)):
                makedirs(dirname(PATH_OUT_pose_joints_3d_VIZ))

            if not exists(dirname(PATH_OUT_pose_joints_VSBL)):
                makedirs(dirname(PATH_OUT_pose_joints_VSBL))

            if not exists(dirname(PATH_OUT_segm_PNG)):
                makedirs(dirname(PATH_OUT_segm_PNG))
            if not exists(dirname(PATH_OUT_segm_VIZ)):
                makedirs(dirname(PATH_OUT_segm_VIZ))

            if not exists(dirname(PATH_OUT_depth_VIZ)):
                makedirs(dirname(PATH_OUT_depth_VIZ))

            if not exists(dirname(PATH_OUT_shape)):
                makedirs(dirname(PATH_OUT_shape))
            if not exists(dirname(PATH_OUT_pose_fullPCA)):
                makedirs(dirname(PATH_OUT_pose_fullPCA))
            if not exists(dirname(PATH_OUT_pose_fullFull)):
                makedirs(dirname(PATH_OUT_pose_fullFull))

            if not exists(dirname(PATH_OUT_depth_EXR)):
                makedirs(dirname(PATH_OUT_depth_EXR))
            if not exists(dirname(PATH_OUT_normals_EXR)):
                makedirs(dirname(PATH_OUT_normals_EXR))
            if not exists(dirname(PATH_OUT_normals_VIZ)):
                makedirs(dirname(PATH_OUT_normals_VIZ))
            if not exists(dirname(PATH_OUT_flow_FLOW_fake)):
                makedirs(dirname(PATH_OUT_flow_FLOW_fake))
            if not exists(dirname(PATH_OUT_flow_FLOW_real)):
                makedirs(dirname(PATH_OUT_flow_FLOW_real))
            if not exists(dirname(PATH_OUT_flow_VIZ)):
                makedirs(dirname(PATH_OUT_flow_VIZ))
            if not exists(dirname(PATH_OUT_segm_EXR)):
                makedirs(dirname(PATH_OUT_segm_EXR))
            if not exists(dirname(PATH_OUT_objectID_EXR)):
                makedirs(dirname(PATH_OUT_objectID_EXR))
            if not exists(dirname(PATH_OUT_objectID_obname)):
                makedirs(dirname(PATH_OUT_objectID_obname))

            if not exists(dirname(PATH_OUT_gender)):
                makedirs(dirname(PATH_OUT_gender))

            if not exists(dirname(PATH_OUT_HandVisibilityLR)):
                makedirs(dirname(PATH_OUT_HandVisibilityLR))

            if not exists(dirname(PATH_OUT_DBG)) and DBG_FLAG_writeOnDisk:
                makedirs(dirname(PATH_OUT_DBG))

            if not exists(dirname(PATH_OUT_BLUR_parms)):
                makedirs(dirname(PATH_OUT_BLUR_parms))

            if not exists(dirname(PATH_OUT_Cam_RT_4x4)):
                makedirs(dirname(PATH_OUT_Cam_RT_4x4))

            if not exists(dirname(PATH_OUT_Subj_Pelvis_T)):
                makedirs(dirname(PATH_OUT_Subj_Pelvis_T))
            if not exists(dirname(PATH_OUT_Subj_ZRot)):
                makedirs(dirname(PATH_OUT_Subj_ZRot))

            if not exists(dirname(PATH_OUT_bg_img_PATH)):
                makedirs(dirname(PATH_OUT_bg_img_PATH))
            if not exists(dirname(PATH_OUT_txt_img_PATH)):
                makedirs(dirname(PATH_OUT_txt_img_PATH))

            if not exists(dirname(PATH_OUT_scale)):
                makedirs(dirname(PATH_OUT_scale))

            if not exists(dirname(PATH_OUT_bbx_head)):
                makedirs(dirname(PATH_OUT_bbx_head))

            if not(exists(dirname(PATH_OUT_ANNOLIST))):
                makedirs(dirname(PATH_OUT_ANNOLIST))

            if not(exists(dirname(PATH_OUT_ANNOLIST_HEAD))):
                makedirs(dirname(PATH_OUT_ANNOLIST_HEAD))

            if not(exists(dirname(PATH_OUT_ANNOLIST_img))):
                makedirs(dirname(PATH_OUT_ANNOLIST_img))

            if not(exists(dirname(PATH_OUT_ANNOLIST_objpos))):
                makedirs(dirname(PATH_OUT_ANNOLIST_objpos))

            if not(exists(dirname(PATH_OUT_ANNOLIST_scale))):
                makedirs(dirname(PATH_OUT_ANNOLIST_scale))

            if not(exists(dirname(PATH_OUT_joint_visibility))):
                makedirs(dirname(PATH_OUT_joint_visibility))


            for i in range(nrOfPeople):
                #not tested for multi person
                if RANDOM_TXT_per_FRAME:                                                                                #
                    cloth_img[i], cloth_img_PATH = set_txt(RANDOM_TXT, cloth_img[i], txt_paths[i], i)  # handles also NON-RANDOM case

            scene.node_tree.nodes['Image'].image = bg_img

            #####################
            texim.image = bg_img
            #####################

            color = None
            for i in range(nrOfPeople):
                for mname, material in materials[i].items():
                    mtree = material.node_tree
                    if color is not None:
                        print(' - color is not None - EXITING')
                        exit(1)
                        mtree.links.new(mtree.nodes['RGB'].outputs[0], mtree.nodes['Script'].inputs[0])
                        mtree.nodes['RGB'].outputs[0].default_value = color
                    else:
                        mtree.links.new(mtree.nodes['Image Texture'].outputs[0], mtree.nodes['Script'].inputs[0])

            #####################################################
            if RANDOM_LIGHTS_per_FRAME:                                                                             #
                sh_coeffs = set_lights(RANDOM_LIGHTS)
            #####################################################
            for scss in harmonics_scripts:
                for ish, coeff in enumerate(sh_coeffs):
                    for sc in scs:
                        sc.inputs[ish+1].default_value = coeff

            # rendered the textured body
            scene.render.use_antialiasing = True
            for i in range(nrOfPeople):
                for part, mat in materials[i].items():
                    if True:
                        mat.node_tree.nodes['Image Texture'].image = cloth_img[i]
                    else:
                        print(0)

            #############################################
            #############################################
            scene.render.filepath = PATH_OUT_full_textBG
            #############################################
            #############################################
            bpy.ops.render.render(write_still=True)
            ########################################
            ########################################

            ##############################################################
            ##############################################################
            # bone locations should be saved AFTER rendering so that the bones are updated
            pose_joints_2d = {}
            pose_joints_3d = {}
            for ob, arm_ob, obname in zip(obs, arm_obs, obnames):
                bpy.ops.object.select_all(action='DESELECT')
                arm_ob.select = True
                bpy.context.scene.objects.active = arm_ob
                pose_joints_2d[obname], pose_joints_3d[obname], bone_loc_names = get_bone_locs(arm_ob, ob, obname, scene, cam_ob, n_bones, UPPER_HEAD)
                #######################################################################################################
            np.save(PATH_OUT_pose_joints_3d, pose_joints_3d)
            np.save(PATH_OUT_pose_joints_2d, pose_joints_2d)
            ################################################################

            pose_dict = {}
            pose_BodyHands_fullFull_dict = {}
            shape_dict = {}
            object_id_obname = {}
            for personIDX, (ob, obname) in enumerate(zip(obs, obnames)):
                pose_dict[obname] = pose[personIDX] #pose contains for time point _idx_ a list of poses for each person

                ########################################################
                pose_body       = pose_list[personIDX][:body_pose_dofs]
                pose_handsPCA   = pose_list[personIDX][body_pose_dofs:(body_pose_dofs+ncomps)]
                proj_pca_2_full = pose_handsPCA.dot(selected_components)
                pose_handsFULL  = proj_pca_2_full + hands_mean
                pose_BodyHands_fullFull_dict[obname] = np.concatenate((pose_body, pose_handsFULL))

                ########################################################
                shape_dict[obname] = shapes[personIDX]

                ########################################################
                object_id_obname[obname] = ob.pass_index

            ################################################################
            np.save(PATH_OUT_pose_fullPCA,  pose_dict)
            ################################################################
            np.save(PATH_OUT_pose_fullFull, pose_BodyHands_fullFull_dict)
            ################################################################
            np.save(PATH_OUT_shape, shape_dict)
            ################################################################
            np.save(PATH_OUT_objectID_obname, object_id_obname)

            system('cp %s %s' % (PATH_INN_segm_EXR.replace(' ', '\ '),    PATH_OUT_segm_EXR.replace(' ', '\ ')))
            system('cp %s %s' % (PATH_INN_objectID_EXR.replace(' ', '\ '),    PATH_OUT_objectID_EXR.replace(' ', '\ ')))

            ####################################################################
            ####################################################################

            if save_flow:
                PATH_OUT_flow_EXR = PATH_OUT_flow_FLOW_fake.replace('.flo', '.exr')
                flow = cv2.imread(PATH_INN_flow_EXR, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                flow[:, :, 0] *= -1
                cv2.imwrite(PATH_OUT_flow_EXR, flow)
                system('mv %s %s' % (PATH_OUT_flow_EXR.replace(' ', '\ '), PATH_OUT_flow_FLOW_fake.replace(' ', '\ ')))
                flow_img = flow_2_img_fromFlowRawImg(flow)
                cv2.imwrite(PATH_OUT_flow_VIZ, flow_img * 255.0)
                flow_write(PATH_OUT_flow_FLOW_real, flow[:, :, :2].astype(np.float32))  # same !!!


            ####################################################################

            normals = cv2.imread(PATH_INN_normals_EXR, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            normals[:, :, 1] *= -1    # OK
            cv2.imwrite(PATH_OUT_normals_EXR, normals)
            if save_dbug_Imgs:
                cv2.imwrite(PATH_OUT_normals_VIZ, normals * 255.)

            #################################################################
            #################################################################
            render_2d_pose(PATH_OUT_pose_joints_2d_VIZ.replace('.png', '___bw.png'), pose_joints_2d, (res[0], res[1]))
            ################################################
            segm_PNG = render_segmentation(PATH_INN_segm_EXR, PATH_OUT_segm_PNG, PATH_OUT_segm_VIZ, save_dbug_Imgs)
            # internally writes on disk !!! the return img is for later use for visibility of joints !!!
            ################################################
            ###############################################################
            #################################################################

            #########################
            jointIDs_2_labelIDs = []
            ###########################################################################
            ###########################################################################
            for ii in range(21 + 1):
                jointIDs_2_labelIDs.append(ii + 1)
            #

            jointIDs = range(62)  # 0..61
            #
            jointIDs_2_labelIDs.append(23)
            jointIDs_2_labelIDs.append(33)
            jointIDs_2_labelIDs.append(43)
            jointIDs_2_labelIDs.append(43)
            #
            jointIDs_2_labelIDs.append(24)
            jointIDs_2_labelIDs.append(34)
            jointIDs_2_labelIDs.append(44)
            jointIDs_2_labelIDs.append(44)
            #
            jointIDs_2_labelIDs.append(25)
            jointIDs_2_labelIDs.append(35)
            jointIDs_2_labelIDs.append(45)
            jointIDs_2_labelIDs.append(45)
            #
            jointIDs_2_labelIDs.append(26)
            jointIDs_2_labelIDs.append(36)
            jointIDs_2_labelIDs.append(46)
            jointIDs_2_labelIDs.append(46)
            #
            jointIDs_2_labelIDs.append(27)
            jointIDs_2_labelIDs.append(37)
            jointIDs_2_labelIDs.append(47)
            jointIDs_2_labelIDs.append(47)
            #
            jointIDs_2_labelIDs.append(28)
            jointIDs_2_labelIDs.append(38)
            jointIDs_2_labelIDs.append(48)
            jointIDs_2_labelIDs.append(48)
            #
            jointIDs_2_labelIDs.append(29)
            jointIDs_2_labelIDs.append(39)
            jointIDs_2_labelIDs.append(49)
            jointIDs_2_labelIDs.append(49)
            #
            jointIDs_2_labelIDs.append(30)
            jointIDs_2_labelIDs.append(40)
            jointIDs_2_labelIDs.append(50)
            jointIDs_2_labelIDs.append(50)
            #
            jointIDs_2_labelIDs.append(31)
            jointIDs_2_labelIDs.append(41)
            jointIDs_2_labelIDs.append(51)
            jointIDs_2_labelIDs.append(51)
            #
            jointIDs_2_labelIDs.append(32)
            jointIDs_2_labelIDs.append(42)
            jointIDs_2_labelIDs.append(52)
            jointIDs_2_labelIDs.append(52)

            if UPPER_HEAD:
                jointIDs_2_labelIDs.append(jointIDs_2_labelIDs[15])
            comp_rgb = cv2.imread(PATH_OUT_full_textBG)
            scale_dict = {}
            head_bb_dict = {}

            print('\n\n\n')
            comp_rgb_VSBL = comp_rgb.copy()
            segm_EXR = cv2.imread(PATH_INN_segm_EXR, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            objectID_EXR = cv2.imread(PATH_INN_objectID_EXR, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            joints_2d_VSBL_FLAGs = {}
            for keyNr, key in enumerate(obnames):
                ii_max = np.min([np.max(pose_joints_2d[key][:,1]), res[1]])
                ii_min = np.max([np.min(pose_joints_2d[key][:, 1]), 0]) #use lowest/highest joint position, or, if outside use resolution
                jj_max = np.min([np.max(pose_joints_2d[key][:,0]), res[0]])
                jj_min = np.max([np.min(pose_joints_2d[key][:,0]), 0])
                bbx = get_head_boundingbox(arm_obs[keyNr], obs[keyNr], scene)
                head_bb_dict[key] = bbx
                scale_dict[key] = np.sqrt( (bbx['x2'] - bbx['x1'])**2 + (bbx['y2'] - bbx['y1'])**2) / 42
                cv2.rectangle(comp_rgb_VSBL, (int(np.round(bbx['x1'])),int(np.round(bbx['y1']))),(int(np.round(bbx['x2'])),int(np.round(bbx['y2']))),(255,0,0), 2)
                joints_2d_VSBL_FLAGs[key] = np.zeros((pose_joints_2d[list(pose_joints_2d.keys())[0]].shape[0],))
                for jID in range(pose_joints_2d[key].shape[0]):
                    centerr = tuple(np.round(pose_joints_2d[key][jID, :]).astype(int))
                    #print(centerr)
                    jj = np.round(pose_joints_2d[key][jID, 0]).astype(int)  # xx
                    ii = np.round(pose_joints_2d[key][jID, 1]).astype(int)  # yy
                    strrr = ''
                    visibilityFLAG = False
                    #
                    if UPPER_HEAD:
                        labelll = jointIDs_2_labelIDs[jID]
                    else:
                        labelll = jointIDs_2_labelIDs[jID]
                    #
                    ff =  (scale_dict[key]*21)/2
                    #
                    iiMin = np.round(np.max((ii-ff, 0)))
                    iiMin = int(iiMin)
                    iiMax = np.round(np.min((ii+ff+1, res[1])))
                    iiMax = int(iiMax)
                    jjMin = np.round(np.max((jj-ff, 0)))
                    jjMin = int(jjMin)
                    jjMax = np.round(np.min((jj+ff+1, res[0])))
                    jjMax = int(jjMax)
                    print(iiMin)
                    print(iiMax)
                    print(jjMin)
                    print(jjMax)
                    testArea = segm_EXR[iiMin:iiMax, jjMin:jjMax].flatten()
                    testArea_objectID = objectID_EXR[iiMin:iiMax, jjMin:jjMax].flatten()
                    #
                    if any(np.logical_and(labelll == testArea, object_id_obname[key] == testArea_objectID)):
                        strrr = '----------' + '----------'
                        visibilityFLAG = True
                    if visibilityFLAG:
                        cv2.circle(comp_rgb_VSBL, centerr, 0, (0, 255, 0))
                    else:
                        cv2.circle(comp_rgb_VSBL, centerr, 0, (0, 0, 255))
                    #
                    joints_2d_VSBL_FLAGs[key][jID] = int(visibilityFLAG)
            #
            rgb_VSBL_PATH = PATH_OUT_pose_joints_2d_VIZ.replace('.png', '___visibility.png')
            cv2.imwrite(rgb_VSBL_PATH, comp_rgb_VSBL)
            #
            np.save(PATH_OUT_pose_joints_VSBL, joints_2d_VSBL_FLAGs)

            np.save(PATH_OUT_scale, scale_dict)
            np.save(PATH_OUT_bbx_head, head_bb_dict)

            camera_RT_4_4 = np.array(cam_ob.matrix_world)
            extrinsic = cam_compute_extrinsic(camera_RT_4_4)
            intrinsic = cam_compute_intrinsic()

            annolist, head_anno, img_anno, objectpos_anno, scale_anno = get_annorect(annolist, head_anno, image_anno, objectpos_anno, scale_anno, pose_joints_2d, joints_2d_VSBL_FLAGs, head_bb_dict, PATH_OUT_full_textBG, scale_dict )
            annolist, head_anno, img_anno, objectpos_anno, scale_anno = get_annorect_from_real_annolist(annolist, head_anno, img_anno, objectpos_anno, scale_anno, annolist_real[idx], comp_rgb.copy(), plane_ob, intrinsic, extrinsic, 1, res,
                                                        cam_ob, scene,PATH_OUT_joint_visibility,PATH_INN_segm_EXR)
            savemat(PATH_OUT_ANNOLIST, mdict={'annolist':annolist})
            savemat(PATH_OUT_ANNOLIST_HEAD , mdict={'annolist_head': head_anno})
            savemat(PATH_OUT_ANNOLIST_img, mdict={'annolist_img': img_anno})
            savemat(PATH_OUT_ANNOLIST_objpos, mdict={'annolist_objpos': objectpos_anno})
            savemat(PATH_OUT_ANNOLIST_scale, mdict={'annolist_scale': scale_anno})

            with open(PATH_OUT_bg_img_PATH, "w") as text_file:
                text_file.write(annolist_real[idx].image.name)

            with open(PATH_OUT_txt_img_PATH, "w") as text_file:
                for i, obname in enumerate(obnames):
                    text_file.write(obname + ': ' + cloth_img_PATHs[i][cloth_img_PATHs[i].find('smpl_data'):] + '\n')


            #save gener as dictionary for better interpretability of outpu
            gender_dict = {}
            for i, obname in enumerate(obnames):
                gender_dict[obname] = gender[i]
            np.save(PATH_OUT_gender, gender_dict)

            for i, obname in enumerate(obnames):
                reset_loc[i] = (pose_joints_2d[obname].max(axis=1) > res[0]).any() or (pose_joints_2d[obname].min(axis=0) < 0).any()

            ########################################################
            rot0_pelvis_loc = {}
            for i, (arm_ob, obname) in enumerate(zip(arm_obs, obnames)):
                arm_ob.select = True
                bpy.context.scene.objects.active = arm_ob
                rot0_pelvis_loc[obname] = arm_ob.pose.bones[obname+'_'+part_match['bone_00']].head.copy()
                rot0_root_loc   = arm_ob.pose.bones[obname+'_'+part_match['root']].location.copy()
                for irot, extra_rot in enumerate([0]):
                    if irot != 0:
                        arm_ob.pose.bones[obname+'_'+part_match['root']].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot[i] + extra_rot), 'XYZ'))
                        arm_ob.pose.bones[obname+'_'+part_match['root']].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))

                        arm_ob.pose.bones[obname+'_'+part_match['root']].location = rot0_root_loc + ((rot0_pelvis_loc[obname] - rot0_root_loc) - Quaternion(Euler((0, 0, extra_rot), 'XYZ')) * (rot0_pelvis_loc[obname] - rot0_root_loc))
                        arm_ob.pose.bones[obname+'_'+part_match['root']].keyframe_insert('location', frame=get_real_frame(seq_frame))

                        PATH_OUT_full_textBG_ROT = PATH_OUT_full_textBG.replace('.png', '%d.png' % irot)
                        scene.render.filepath = PATH_OUT_full_textBG_ROT

                        bpy.ops.render.render(write_still=True)

                #######################################################
                arm_ob.pose.bones[obname+'_'+part_match['root']].location = rot0_root_loc
                arm_ob.pose.bones[obname+'_'+part_match['root']].keyframe_insert('location', frame=get_real_frame(seq_frame))

                arm_ob.pose.bones[obname+'_'+part_match['root']].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot[i]), 'XYZ'))
                arm_ob.pose.bones[obname+'_'+part_match['root']].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))

                arm_ob.pose.bones[obname+'_'+part_match['root']].rotation_quaternion = Quaternion((1, 0, 0, 0))
                ###################################################

                ###################################################
                rot0_pelvis_loc[obname] = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_'+part_match['bone_00']].head.copy()
                rot0_root_loc   = arm_ob.pose.bones[obname+'_'+part_match['root']].location.copy()
            #
            if DBG_FLAG_writeOnDisk:
                #
                cam_ob_matrix_world_RRR = np.array(cam_ob.matrix_world)[:3,:3]
                cam_ob_matrix_world_TTT = np.array(cam_ob.matrix_world)[:3,3]
                cam_ob_matrix_world_RR2 = np.array(Euler(cam_ob.rotation_euler).to_matrix())
                cam_ob_matrix_world_TT2 = np.array(      cam_ob.location)
                #
                tstRRR = np.allclose(cam_ob_matrix_world_RRR, cam_ob_matrix_world_RR2)
                tstTTT = np.allclose(cam_ob_matrix_world_TTT, cam_ob_matrix_world_TT2)
                 #
                assert tstRRR
                assert tstTTT
                #
                if DBG_FLAG_writeOnDisk:
                    with open(PATH_OUT_DBG, "w") as text_file:
                        text_file.write('cam_ob.matrix_world\n\n')
                        text_file.write('%s' % np.array(cam_ob.matrix_world))
                        text_file.write('\n\n\n\n\n')
                        text_file.write('cam_ob_matrix_world_RRR\n\n')
                        text_file.write('%s' % cam_ob_matrix_world_RRR)
                        text_file.write('\n\n')
                        text_file.write('cam_ob_matrix_world_TTT\n\n')
                        text_file.write('%s' % cam_ob_matrix_world_TTT)
                        text_file.write('\n\n')
                        text_file.write('tstRRR, tstTTT\n\n')
                        text_file.write('%s\n%s' % (tstRRR, tstTTT))
                        text_file.write('\n\n\n\n\n')
                        text_file.write('rot0_pelvis_loc\n\n')
                        text_file.write('%s' % rot0_pelvis_loc)
                        text_file.write('\n\n')
                        text_file.write('random_zrot\n\n')
                        text_file.write('%s' % np.array(random_zrot))
                        text_file.write('\n\n\n\n\n')
                        text_file.write('RANDOM_CAMERA_JITTER - random_camera_rotX, random_camera_rotY, random_camera_rotZ, random_camera_trans\n\n')
                        text_file.write('%s %s %s %s' % (random_camera_rotX, random_camera_rotY, random_camera_rotZ, random_camera_trans))
                        #
                        text_file.write('\n\n\n\n\n')
                        for kk in bpy.data.objects.keys():
                            text_file.write(kk)
                            text_file.write('\n')
            #
            camera_RT_4_4 = np.array(cam_ob.matrix_world)
            np.save(PATH_OUT_Cam_RT_4x4, camera_RT_4_4)

            random_zrot_dict         = {}
            for i, obname in enumerate(obnames):
                rot0_pelvis_loc[obname] = np.array(rot0_pelvis_loc[obname])
                random_zrot_dict[obname]        = random_zrot[i]
            np.save(PATH_OUT_Subj_Pelvis_T, rot0_pelvis_loc)
            np.save(PATH_OUT_Subj_ZRot,     random_zrot_dict)   # be careful if there is extra_rot !!!

            project_joints3d_imwrite(pose_joints_3d, comp_rgb_VSBL, imgPATH=PATH_OUT_pose_joints_3d_VIZ, cam_ob=cam_ob, scene=scene, save_dbug_Imgs=save_dbug_Imgs )


    print('FINITO')
