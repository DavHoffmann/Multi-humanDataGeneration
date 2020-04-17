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

# from mesh_intersections import get_intersections_indices

#Todo: add absolute path to mesh package root
path_meshPackage =  ''

import sys
sys.path.append(path_meshPackage)
from mesh import Mesh

from findMeshIntersection import get_intersections_indices
from numpy import linspace, zeros, save, array, unique, reshape, arange, ones

import argparse
from os import listdir
from itertools import combinations
from os.path import join, exists
from scipy.special import binom
import pickle

import imp


lm = imp.load_source(
    'load_model',
    './smpl_data/mano_v1_2/webuser/smpl_handpca_wrapper_multiHuman.py')

# smpl_data/mano_v1_2/webuser
def get_excludeDict():
    exclude = {}
    exclude['rightCalf'] = ['rightThigh', 'rightFoot']
    exclude['leftCalf'] = ['leftThigh', 'leftFoot']
    exclude['head'] = ['neck']
    exclude['rightHand'] = ['rightForeArm']
    exclude['leftHand'] = ['leftForeArm']
    exclude['neck'] = ['head', 'spine2']
    exclude['spine'] = ['global', 'spine1','spine2', 'leftUpperArm', 'rightUpperArm']
    exclude['spine1'] = ['spine', 'spine2', 'leftUpperArm', 'rightUpperArm']
    exclude['rightFoot'] = ['rightCalf']
    exclude['leftFoot'] = ['leftCalf']
    exclude['leftUpperArm'] = ['spine2', 'spine1', 'spine', 'leftForeArm', 'leftShoulder']
    exclude['rightUpperArm'] = ['spine2', 'spine1', 'spine', 'rightForeArm', 'rightShoulder']
    exclude['spine2'] = ['leftUpperArm', 'rightUpperArm', 'neck', 'spine1', 'leftShoulder', 'rightShoulder', 'spine', 'spine2']
    exclude['global'] = ['rightThigh', 'leftThigh', 'spine', 'spine2']
    exclude['rightThigh'] = ['global', 'leftThigh', 'rightCalf']
    exclude['leftThigh'] = ['global', 'rightThigh', 'leftCalf']
    exclude['leftShoulder'] = ['leftUpperArm', 'spine2']
    exclude['rightShoulder'] = ['rightUpperArm', 'spine2']
    exclude['leftForeArm'] = ['leftHand', 'leftUpperArm']
    exclude['rightForeArm'] = ['rightHand', 'rightUpperArm']

    return exclude



def visualize(model, self_intersection_ids, default_model, vertices,faces):
    radius = 0.01
    from psbody.mesh.meshviewer import MeshViewer
    # from mesh.mesh.meshviewer import MeshViewer
    mv = MeshViewer(window_width=800, window_height=800)
    mesh =  Mesh(v=default_model.r, f=default_model.f)
    mesh.f = []

    if self_intersection_ids == []:
        return
        mv.set_static_meshes([mesh, meanSphere], blocking=True)
    else:
        m = Mesh(v=model.v, f=model.f, vc='SeaGreen')
        m.vc[m.f[faces][self_intersection_ids].flatten()] = ones(1)
        mv.set_static_meshes([m], blocking=True)

    raw_input('Press Enter to exit')



def find_self_penetration(bodypart2vert, model, default_model, excludeDict, gender, fastFace=None, vis=False):
    face_by_vertex = model.faces_by_vertex()
    self_intersection_ids = []
    bodypart2face_facerest = {}
    for key in bodypart2vert.keys():
        if fastFace is None:
            self_intersection_ids = []
            print(key)
            vertices = bodypart2vert[key]
            vertices = array(vertices, dtype=int)
            faces = []
            for vertex in vertices:
                faces = faces + face_by_vertex[vertex]

            faces = unique( array( faces, dtype=int).reshape(-1,) )

            excludeVertices = []
            for bodypart in excludeDict[key]:
                excludeVertices = excludeVertices + list( bodypart2vert[bodypart] )

            vertices_rest = [vertex for vertex in arange(model.v.shape[0]) if (not(vertex in vertices or vertex in excludeVertices))]
            vertices_rest = array(vertices_rest, dtype=int)
            faces_rest = []
            for vertex in vertices_rest:
                faces_rest = faces_rest + face_by_vertex[vertex]
            faces_rest = unique(array(faces_rest, dtype=int).reshape(-1, ))

            sface = set(faces)
            sface_rest = set(faces_rest)
            sintersec = sface.intersection(sface_rest)
            sface  = sface.difference(sintersec)
            faces = array( list(sface) )

            bodypart2face_facerest[key] = [faces, faces_rest]

        else:
            vertices = None
            faces = fastFace[key][0]
            faces_rest = fastFace[key][1]


        selfIntersec = get_intersections_indices(model.v, model.f[faces], model.v, model.f[faces_rest])

        if not(len(selfIntersec)==0):
            if vis:
                self_intersection_ids = self_intersection_ids + list(selfIntersec)
                visualize(model, self_intersection_ids, default_model, vertices, faces)

            return True
    if fastFace is None:
        with open('./selfPenetrationTest/bodypart2face_facerest_'+gender+'.pkl', 'w+') as outfile:
            pickle.dump(bodypart2face_facerest, outfile)

    return False


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelFolder', type=str)
    parser.add_argument('--obnames', type=str)
    parser.add_argument('--selfPenetrationCheck', type=bool, default=False)
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()

    selfPenetrationCheck = args.selfPenetrationCheck
    obnames = args.obnames
    modelFolder = args.modelFolder
    visualize = args.visualize

    plane_ob_name = 'plane_ob_model.ply'
    male_model = lm.load_model(
        fname_or_dict='./smpl_data/mano_v1_2/models/SMPLH_female.pkl')
    female_model = lm.load_model(
        fname_or_dict='./smpl_data/mano_v1_2/models/SMPLH_female.pkl')

    if selfPenetrationCheck:
        with open('./selfPenetrationTest/bodypart2vert_dict_female.pkl','rb') as infile:
            bodypart2vert_dict_female = pickle.load(infile, encoding='latin1')
        with open('./selfPenetrationTest/bodypart2vert_dict_male.pkl','rb') as infile:
            bodypart2vert_dict_male = pickle.load(infile, encoding='latin1')
        if exists('./selfPenetrationTest/bodypart2face_facerest_f.pkl'):
            with open('./selfPenetrationTest/bodypart2face_facerest_f.pkl','rb') as infile:
                fastFace_female = pickle.load(infile, encoding='latin1')
        else:
            fastFace_female = None
        if exists('./selfPenetrationTest/bodypart2face_facerest_m.pkl'):
            with open('./selfPenetrationTest/bodypart2face_facerest_m.pkl','rb') as infile:
                fastFace_male = pickle.load(infile, encoding='latin1')
        else:
            fastFace_male = None

        excludeDict = get_excludeDict()



    print(obnames)

    models_flnms = obnames.split('+')
    print(models_flnms)
    models_filenames = []
    models_flnms.pop(0) #necessary because there is the empty string in first posission!!
    for filename in models_flnms:
        models_filenames.append( filename+'_model.ply')
    print(models_filenames)

    combs = linspace(0, len(models_filenames) - 1, len(models_filenames), dtype=int)
    pairs = list(combinations(combs, 2))
    PATH_tmp = modelFolder
    PATH_tmp = PATH_tmp.split('models')[0]

    models = []
    model_names = []
    intersection_dict = {}
    selfIntersection_dict = {}

    plane_ob_collisions = zeros((len(models_filenames),))
    intersections_per_ob = zeros((len(models_filenames),))
    collisions_per_pair = zeros((int(binom(len(models_filenames), 2)), ))

    for i, model in enumerate(models_filenames):
        model_names.append(model.split('_mo')[0])
        models.append(Mesh(filename= join(modelFolder,model)))
        intersection_dict[model_names[i]] = 0

        if selfPenetrationCheck:
            if 'f_avg' in model_names[i]:
                selfIntersection = find_self_penetration(bodypart2vert_dict_female, models[i], female_model, excludeDict, 'f', fastFace_female, visualize)
            else:
                selfIntersection = find_self_penetration(bodypart2vert_dict_male, models[i], male_model, excludeDict, 'm', fastFace_male, visualize)

            selfIntersection_dict[model_names[i]] = selfIntersection

    plane_ob = Mesh(filename=join(modelFolder,plane_ob_name))


#for counts per pair

    for i, pair in enumerate(pairs):
        collisions = get_intersections_indices(models[pair[0]].v, models[pair[0]].f, models[pair[1]].v, models[pair[1]].f )
        print('-----------')
        print(len(collisions))
        print(len(unique(collisions)))
        if not(len(collisions) == 0):
            collisions_per_pair[i] += 1

    for i, model in enumerate(models):
        collisions = get_intersections_indices(model.v, model.f, plane_ob.v, plane_ob.f)
        if not(len(collisions) ==0):
            plane_ob_collisions[i] += 1

    print(PATH_tmp)
    save(join(PATH_tmp,'intersection.npy'), collisions_per_pair)
    save(join(PATH_tmp, 'plane_collision.npy'), plane_ob_collisions)
    save(join(PATH_tmp, 'self_intersection.npy'), selfIntersection_dict)
