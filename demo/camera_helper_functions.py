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
import numpy as np


def get_3x4_RT_matrix_from_blender(cam):

    from mathutils import Matrix #Todo: additional dependency! not listed in readme
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    cam = Matrix(cam)
    location, rotation = cam.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*np.dot(R_world2bcam, location)

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.dot(R_bcam2cv,R_world2bcam)
    T_world2cv = np.dot(R_bcam2cv,T_world2bcam)

    # put into 3x4 matrix
    RT = np.concatenate((np.array(R_world2cv), np.array(T_world2cv).reshape(3,1)), axis=1)

    return RT


def point_in_camera_coords(cam, points,intcam_mat):

    transformed =np.zeros_like(points)
    projected_2d = np.zeros((points.shape[0], 2))

    cam_mat_world =np.linalg.inv(cam)
    RT = np.concatenate((np.diag([1., 1., 1.]), np.zeros((3, 1))), axis=1)

    #for each point get homogenous representation and apply coordinate system transformation
    for i in range(points.shape[0]):
        hom_p = np.concatenate([points[i,:], np.array([1])], axis=0)
        #get point incamera coordinate system
        transformed[i,:] = np.dot(cam_mat_world, hom_p)[0:3]
        transformed[i,-1] = -transformed[i,-1]
        transformed[i, 1] = -transformed[i, 1]
        #project to 2d image plane (use default extrinsic matrix, since points are in camera coordinate frame)
        projected_2d[i, :] = project_point(np.concatenate([transformed[i,:],np.array([1])]), RT, intcam_mat)

    return transformed, projected_2d


def project_point(joint, RT, KKK):

    P = np.dot(KKK,RT)
    joints_2d = np.dot(P, joint)
    joints_2d = joints_2d[0:2] / joints_2d[2]

    return joints_2d


def cam_compute_intrinsic(res):
    # These are set in Blender
    res_x_px = res[0]  # *scn.render.resolution_x
    res_y_px = res[1]  # *scn.render.resolution_y
    f_mm = 60  # *cam_ob.data.lens
    sensor_w_mm = 32  # cam_sensor
    sensor_h_mm = 32  # sensor_w_mm * res_y_px / res_x_px

    scale = 1  # *scn.render.resolution_percentage/100
    skew = 0  # only use rectangular pixels
    pixel_aspect_ratio = 1

    # From similar triangles:
    # sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
    fx_px = f_mm * res_x_px * scale / sensor_w_mm
    fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

    # Center of the image
    u = np.round(res_x_px * scale / 2)
    v = np.round(res_y_px * scale / 2)

    # Intrinsic camera matrix
    K = np.array([[fx_px, skew, u],
                  [0, fy_px, v],
                  [0, 0, 1]])
    return K


def project_vertices(points, intrinsic, extrinsic):
    homo_coords = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).transpose()
    proj_coords = np.dot(intrinsic, np.dot(extrinsic, homo_coords))
    proj_coords = proj_coords / proj_coords[2]
    proj_coords = proj_coords[:2].transpose()
    return proj_coords
