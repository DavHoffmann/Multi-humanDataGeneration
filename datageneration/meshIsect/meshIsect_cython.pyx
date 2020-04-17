# -*- coding: utf-8 -*-
# Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is
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

import cython

cimport numpy as np
from cpython cimport PyObject
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t

np.import_array()

cdef extern from "meshIsect.h":
  void meshIsectF(  const uint32_t* faces1,
               const size_t n_faces1,
               const double* verts1,
               const size_t n_verts1,
               const uint32_t* faces2,
               const size_t n_faces2,
               const double* verts2,
               const size_t n_verts2,
               vector[uint32_t] mesh_intersections);


def get_intersections_indices(  np.ndarray[double, ndim=2, mode='c'] verts1 not None,
                                np.ndarray[uint32_t, ndim=2, mode='c'] faces1 not None,
                                np.ndarray[double, ndim=2, mode='c'] verts2 not None,
                                np.ndarray[uint32_t, ndim=2, mode='c'] faces2 not None):
  cdef int n_verts1, dims1, n_verts2, dims2;

  n_verts1, dims_v1 = verts1.shape[0], verts1.shape[0];
  n_verts2, dims_v2 = verts2.shape[0], verts2.shape[0];

  n_faces1, dims_f1 = faces1.shape[0], verts1.shape[0];
  n_faces2, dims_f2 = faces2.shape[0], verts2.shape[0];

  cdef vector[uint32_t] mesh_intersections;
  meshIsectF( &faces1[0,0], n_faces1, &verts1[0,0], n_verts1, &faces2[0,0], n_faces2, &verts2[0,0], n_verts2, mesh_intersections);

  return mesh_intersections
