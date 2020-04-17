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
from mathutils import Vector
import scipy
from mathutils.geometry import intersect_ray_tri

def mesh_collision(mesh_vertices, meshes, tups, precise=False, min_max_x=None):
    # If resolution is too low (large triangles few points) it will produce misses
    # if using precise make sure to reduce nr of triangles before
    collisions              = {}
    boundingBoxes           = []
    collisions_per_person   = np.zeros((len(meshes), 1))
    plane_collision         = np.zeros((len(meshes), 1), dtype=bool)

    for i, mesh in enumerate(mesh_vertices):
        box = get_axis_aligned_boundingBox_mesh(mesh)
        boundingBoxes.append( box )

        if not(min_max_x is None) and ((box['x_min'] <= min_max_x[0]) or  box['x_max'] >= min_max_x[1]):
            plane_collision[i] = True
    nr_of_obs = len(boundingBoxes)

    if nr_of_obs > 1:
        collisions_per_pair = np.zeros((int(scipy.special.binom(len(meshes), 2)), 1))

        for i, pair in enumerate( tups ):
            collisions[pair], intersection_cube = bb_collision(boundingBoxes[pair[0]],
                                                               boundingBoxes[pair[1]])

            if collisions[pair]:
                collisions_per_person[pair[0]] += 1
                collisions_per_person[pair[1]] += 1
                collisions_per_pair[i] += 1

            #if boundingBoxes overlap check if objects overlap
            if precise:
                intersecting_vertices = points_in_cube(mesh_vertices[pair[0]],
                                                       intersection_cube)
                intersection_counts = np.zeros((np.shape(intersecting_vertices)[0],
                                                1))
                if not(np.shape(intersecting_vertices)[0] == 0):
                    for face in meshes[pair[1]].polygons:
                        face_verts = list( face.vertices[:] )
                        face_vectors = np.array(mesh_vertices[pair[1]])[face_verts,:]
                        mask = pts_in_face_xy_bounds(face_vectors,
                                                     intersecting_vertices)

                        if not(np.sum(mask) == 0):
                            intersection_counts[mask] += face_intersect(
                                                            face_vectors,
                                                            intersecting_vertices[mask])
                if any(intersection_counts % 2 > 0):
                    collisions[i] = True
                else:
                    collisions[i] = False
                    collisions_per_person[pair[0]] -= 1
                    collisions_per_person[pair[1]] -= 1
                    collisions_per_pair[i] -= 1
        return collisions, collisions_per_pair, plane_collision

    else:
        return 0, 0, plane_collision

def get_axis_aligned_boundingBox_mesh(mesh):
    #unused
    v_world = np.array(mesh)

    corners = {
            'x_min' : np.min(v_world[:,0]),
            'x_max' : np.max(v_world[:, 0]),
            'y_min' : np.min(v_world[:,1]),
            'y_max' : np.max(v_world[:, 1]),
            'z_min' : np.min(v_world[:,2]),
            'z_max' : np.max(v_world[:, 2])}

    return corners


def bb_collision(box1, box2):

    intersection_cube = {}
    intersection_cube['minmax_x'] = np.min([box1['x_max'], box2['x_max']])
    intersection_cube['minmax_y'] = np.min([box1['y_max'], box2['y_max']])
    intersection_cube['minmax_z'] = np.min([box1['z_max'], box2['z_max']])
    intersection_cube['maxmin_x'] = np.max([box1['x_min'], box2['x_min']])
    intersection_cube['maxmin_y'] = np.max([box1['y_min'], box2['y_min']])
    intersection_cube['maxmin_z'] = np.max([box1['z_min'], box2['z_min']])

    intersecting = (
            intersection_cube['maxmin_x'] <= intersection_cube['minmax_x'] and
            intersection_cube['maxmin_y'] <= intersection_cube['minmax_y'] and
            intersection_cube['maxmin_z'] <= intersection_cube['minmax_z'])

    return intersecting, intersection_cube


def points_in_cube(mesh, cube):
    mesh = np.array(mesh)
    mask = np.logical_and(mesh[:, 0] >= cube['maxmin_x'],
                          mesh[:, 0] <= cube['minmax_x'])
    mask = np.logical_and(mask,
                          mesh[:, 1] >= cube['maxmin_y'])
    mask = np.logical_and(mask,
                          mesh[:, 1] <= cube['minmax_y'])
    mask = np.logical_and(mask,
                          mesh[:, 2] >= cube['maxmin_z'])
    mask = np.logical_and(mask,
                          mesh[:, 2] <= cube['minmax_z'])

    return mesh[mask,:]


def pts_in_face_xy_bounds(face_verts, verts):

    xmin = np.min(face_verts[:, 0])
    ymin = np.min(face_verts[:, 1])
    xmax = np.max(face_verts[:, 0])
    ymax = np.max(face_verts[:, 1])

    pt_candidate_mask = np.logical_and(verts[:, 0] >= xmin, verts[:, 0] <= xmax)
    pt_candidate_mask = np.logical_and(pt_candidate_mask, verts[:, 1] >= ymin)
    pt_candidate_mask = np.logical_and(pt_candidate_mask, verts[:, 1] <= ymax)

 #   return verts[pt_candidate_mask, :]
    return  pt_candidate_mask


def face_intersect(face, pts):
    ray = Vector((0,0,1))
    nrIntersections = np.zeros((np.shape(pts)[0],1), dtype=bool)
    for i in range(np.shape(pts)[0]):
        isect = intersect_ray_tri(face[0], face[1], face[2], ray, pts[i,:], 1)

        nrIntersections[i] = bool( isect and isect.z > pts[i, 2])

    return np.array(nrIntersections, dtype=int)
