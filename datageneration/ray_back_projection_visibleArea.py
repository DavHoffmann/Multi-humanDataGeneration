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

import bpy
import bpy_extras
from mathutils import Matrix
import numpy as np
import scipy.linalg as lin
from bpy_extras.view3d_utils import region_2d_to_origin_3d
from bpy_extras.view3d_utils import region_2d_to_vector_3d

# ---------------------------------------------------------------
# 3x4 P matrix from Blender camera
# ---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew, u_0),
         (0, alpha_v, v_0),
         (0, 0, 1)))
    return K


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv =  R_world2bcam
    T_world2cv =  T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
    ))
    return RT


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K * RT, K, RT


def get_vec(p1, cam_ob):
    #using blender cam (img plane corresponds to view through blender cam in gui)
    #[0,0] right top img plane
    #[640,0] left top img plane
    #[640,640] left bottom
    #[0, 640] right bottom
    #[320, 640] bottom middle
    P, K, RT = get_3x4_P_matrix_from_blender(cam_ob)

    X = np.dot(lin.pinv(P), p1)
    X = X / X[3]
    XX = np.copy(X[0:3])

    return XX


def get_xrange(cam_ob, res_x, res_y):
    #returns x value fro intersection with the ground plane and ray originating at bottom middle pixel
    # project bottom middle back to 3D z position is arbitrary
    p_bot = [res_x/2, res_y, 1]
    # get vector intersecting with camera and image plane at bottom middle of image plane
    XX = get_vec(p_bot, cam_ob)

    # construct ray from cam_ob.location in direction XX
    vec = XX[0:3] - cam_ob.location
    # scale ray vector, such that its length of y equals y of camera
    vec = vec / np.abs(vec[1]) * cam_ob.location[1]
    # subtracting vector from cam_ob.location will give intersection with y=0 plane
    intersection = np.array(cam_ob.location) - vec[0:3]

    return intersection[0]


def get_yrange(cam_ob, x_fixed, res_x, res_y):
    #z in blender coordinate system
    P, K, RT = get_3x4_P_matrix_from_blender(cam_ob)
    # get pixel where p=[x_fixed,0,0] project to
    px = np.dot(P, [x_fixed,0, 0, 1])
    px = px / px[-1]

    # use y value on image plane that p projects to, to construct extreme x pixels for x_fixed
    px_extremes = [[res_x, px[1], 1], # leftmost pixel
                   [0,     px[1], 1]] # rightmost pixel

    #get ray from extreme pixels in 3D
    ray1 = get_vec(px_extremes[0], cam_ob)[0:3]
    ray2 = get_vec(px_extremes[1], cam_ob)[0:3]

    # get vector from cam_ob.location to point in space (arbitrary points, but project to px_extremes)
    vec1 = ray1 - cam_ob.location
    vec2 = ray2 - cam_ob.location

    #scale x component of vec to 1
    vec1 = vec1 / np.abs(vec1[0])
    vec2 = vec2 / np.abs(vec2[0])

    #make sure that x component has same length as distance between cam_ob and x_fixed
    vec1 = vec1 * (cam_ob.location[0] - x_fixed)
    vec2 = vec2 * (cam_ob.location[0] - x_fixed)

    minY = vec1[2]
    maxY = vec2[2]

    y_range = [minY, maxY]

    return y_range
