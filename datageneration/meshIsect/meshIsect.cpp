/*
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

*/

#include "meshIsect.h"

typedef CGAL::Simple_cartesian<double> K;
typedef CGAL::AABB_triangle_primitive<K,std::vector<K::Triangle_3>::iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

typedef std::vector<K::Point_3> point_vec;
typedef std::vector<K::Triangle_3> triang_vec;

#ifdef _OPENMP
#include <omp.h>
#endif

void meshIsectF(const uint32_t* faces1,
               const size_t n_faces1,
               const double* verts1,
               const size_t n_verts1,
               const uint32_t* faces2,
               const size_t n_faces2,
               const double* verts2,
               const size_t n_verts2
               ,std::vector<uint32_t> &mesh_intersections
                             ){

   int (*q_faces_arr)[3] = (int(*)[3]) &faces1[0];
   point_vec verts_vec_1;
   verts_vec_1.reserve(n_verts1);
   for (int i=0; i<n_verts1*3; i+=3){
     verts_vec_1.push_back(K::Point_3(verts1[i],
                                     verts1[i+1],
                                     verts1[i+2]));
   }

   //get n_verts2 x 3 vectors for verts2
   point_vec verts_vec_2;
   verts_vec_2.reserve(n_verts2);

   for (int i=0; i<n_verts2*3; i+=3){
     verts_vec_2.push_back(K::Point_3(verts2[i],
                                     verts2[i+1],
                                     verts2[i+2]));
   }

   // get n_faces x 3 faces for faces1
   triang_vec faces_vec_1;
   faces_vec_1.reserve(n_faces1);
   for (int i=0; i<n_faces1*3; i+=3){
     faces_vec_1.push_back(K::Triangle_3(verts_vec_1[faces1[i]],
                                         verts_vec_1[faces1[i+1]],
                                         verts_vec_1[faces1[i+2]]));
   }

   // get n_faces x 3 faces for faces2
   triang_vec faces_vec_2;
   faces_vec_2.reserve(n_faces2);
   std::cout << n_verts2 << std::endl;
   std::cout << n_faces2 << std::endl;
   for (int i=0; i<n_faces2*3; i+=3){
     std::cout << i<< std::endl;
     faces_vec_2.push_back(K::Triangle_3(verts_vec_2[faces2[i]],
                                         verts_vec_2[faces2[i+1]],
                                         verts_vec_2[faces2[i+2]]));
   }


   // construct tree for mesh 2
   Tree tree(faces_vec_2.begin(), faces_vec_2.end());
   tree.accelerate_distance_queries();


   // USE TREE TO QUERY FACES
   std::cout << "loop5" <<std::endl;
   #pragma omp parallel for
   for(size_t i=0; i<n_faces1; ++i){
     K::Triangle_3 triangle_query( verts_vec_1[q_faces_arr[i][0]],
                                   verts_vec_1[q_faces_arr[i][1]],
                                   verts_vec_1[q_faces_arr[i][2]] );

     if (tree.do_intersect(triangle_query)) {
         mesh_intersections.push_back(i);
     }

   }


 }

int main(){}
