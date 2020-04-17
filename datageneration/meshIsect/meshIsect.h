/*
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
*/

#ifndef MESHISECT_H
#define MESHISECT_H
#include <string>
#define CGAL_CFG_NO_CPP0X_VARIADIC_TEMPLATES 1

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <list>
#include <boost/cstdint.hpp>
using boost::uint32_t;
// #include <boost/array.hpp>


// #include "numpy/arrayobject.h"

using std::vector;

// #include <boost/cstdint.hpp>
// #include <boost/array.hpp>
// #include <vector>
// #include <Python.h>
// #include <CGAL/Simple_cartesian.h>
// #include <list>

// typedef CGAL::Simple_cartesian<double>::Point_3 Point;
// typedef unsigned int UType;
// std::vector<uint32_t> meshIsectF(const uint32_t* faces1,
//                                  const size_t n_faces1,
//                                  const double* verts1,
//                                  const size_t n_verts1,
//                                  const uint32_t* faces2,
//                                  const size_t n_faces2,
//                                  const double* verts2,
//                                  const size_t n_verts2
//                                  ,std::vector<uint32_t> &mesh_intersections
//                                );
// std::vector<uint32_t> meshIsectF(const uint32_t* faces1,
//                                  const size_t n_faces1,
//                                  const double* verts1,
//                                  const size_t n_verts1,
//                                  const uint32_t* faces2,
//                                  const size_t n_faces2,
//                                  const double* verts2,
//                                  const size_t n_verts2
//                                  ,std::vector<uint32_t> &mesh_intersections
                               // );
// void meshIsectF(const uint32_t* faces1,
//                  const size_t n_faces1,
//                  const double* verts1,
//                  const size_t n_verts1,
//                  const uint32_t* faces2,
//                  const size_t n_faces2,
//                  const double* verts2,
//                  const size_t n_verts2
//                  ,std::vector<uint32_t> &mesh_intersections

void meshIsectF(const uint32_t* faces1,
                 const size_t n_faces1,
                 const double* verts1,
                 const size_t n_verts1,
                 const uint32_t* faces2,
                 const size_t n_faces2,
                 const double* verts2,
                 const size_t n_verts2
                 ,std::vector<uint32_t> &mesh_intersections

                               );
// void meshIsectF(
//                                );
// PyObject* meshIsectF(const uint32_t* faces1,
//                                  const size_t n_faces1,
//                                  const double* verts1,
//                                  const size_t n_verts1,
//                                  const uint32_t* faces2,
//                                  const size_t n_faces2,
//                                  const double* verts2,
//                                  const size_t n_verts2
//                                  ,std::vector<uint32_t> &mesh_intersections
//                                );
// class Mesh_IntersectionsException: public std::exception {
// public:
//     Mesh_IntersectionsException(std::string m="Mesh_IntersectionsException!"):msg(m) {}
//     ~Mesh_IntersectionsException() throw() {}
//     const char* what() const throw() { return msg.c_str(); }
// private:
//     std::string msg;
// };
#endif // MESH_INTERSECTIONS_H
