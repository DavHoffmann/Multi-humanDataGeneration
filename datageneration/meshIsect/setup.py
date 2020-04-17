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

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

# python setup.py build_ext --inplace

cgalIncludeDir = '../../cgal/CGAL-4.7/include'

setup(
    name = 'findMeshIntersection',
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("findMeshIntersection",
            sources=["meshIsect_cython.pyx", "meshIsect.cpp"],
            libraries=['CGAL', 'boost_system'],
            include_dirs=[numpy.get_include(), '/usr/local/include', '/usr/include', cgalIncludeDir],
            library_dirs=["/usr/lib/", "/usr/lib/x86_64-linux-gnu"],
            language="c++",
            extra_compile_args=[ "-lboost_system","-lCGAL"]
     )])
