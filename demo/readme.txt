Demo scripts to load and process the data are provided on the associated git repository.

blur_parms:
  contains the blur parameters.
  array of size (3,) containing [blur vector, size_x, size_y] (size in pixels).

camera_RT_4x4:
  camera matrix as returned by blenders camera.matrix_world (see both demo scripts for more instructions on how to use this camera matrix).

composition:
  rendered rgb images

depth_EXR:
  depth maps. Note that the there is a cut off depth. To use the depth maps mask out all values with depth== 1.e+10. In the data generation scripts provided in the git repository the cut of depth is turned off. If you create data with this script use objectID_EXR as mask.

flow:
  Optical flow. Only available for multi-human optical flow dataset.

gender:
  contains a dictionary that maps object name to gender.

normal_EXR:
  surface normals

objectId_EXR:
  each pixel is assigned a value corresponding to background or a unique value for each synthetic human.

pose_coeffs_fullFull:
  dictionary containing SMPL-H pose coefficient for each synthetic human.

pose_coeffs_fullPCA:
  dictionary containing SMPL-H pose coefficient for each synthetic human and PCA components for the hand poses as used as input my MANO. Please refer to the demo script load_smpl_and_render_demo.py for more instructions.

pose_joints_2d:
  2D locations on image of respective joints. For a mapping from index to joint name see below. For visualisation see demo script joint_numbering_2d_3d.py.
  J_names = {
      0: 'Pelvis',

      1: 'L_Hip',
      4: 'L_Knee',
      7: 'L_Ankle',
      10: 'L_Foot',

      2: 'R_Hip',
      5: 'R_Knee',
      8: 'R_Ankle',
      11: 'R_Foot',

      3: 'Spine1',
      6: 'Spine2',
      9: 'Spine3',
      12: 'Neck',
      15: 'Head',

      13: 'L_Collar',
      16: 'L_Shoulder',
      18: 'L_Elbow',
      20: 'L_Wrist',

      14: 'R_Collar',
      17: 'R_Shoulder',
      19: 'R_Elbow',
      21: 'R_Wrist',

   #left hand
      22: 'L_index_0',
      23: 'L_index_1',
      24: 'L_index_2',
      25: 'L_index_3',

      26: 'L_middle_0',
      27: 'L_middle_1',
      28: 'L_middle_2',
      29: 'L_middle_3',

      30: 'L_pinky_0',
      31: 'L_pinky_1',
      32: 'L_pinky_2',
      33: 'L_pinky_3',

      34: 'L_ring_0',
      35: 'L_ring_1',
      36: 'L_ring_2',
      37: 'L_ring_3',

      38: 'L_thumb_0',
      39: 'L_thumb_1',
      40: 'L_thumb_2',
      41: 'L_thumb_3',

  #right hand
      42: 'R_index_0',
      43: 'R_index_1',
      44: 'R_index_2',
      45: 'R_index_3',

      46: 'R_middle_0',
      47: 'R_middle_1',
      48: 'R_middle_2',
      49: 'R_middle_3',

      50: 'R_pinky_0',
      51: 'R_pinky_1',
      52: 'R_pinky_2',
      53: 'R_pinky_3',

      54: 'R_ring_0',
      55: 'R_ring_1',
      56: 'R_ring_2',
      57: 'R_ring_3',

      58: 'R_thumb_0',
      59: 'R_thumb_1',
      60: 'R_thumb_2',
      61: 'R_thumb_3',

  # additional
      62: 'top_head'
  }



pose_joints_3d:
  3D location of joints in a global coordinate frame (positive x points back, positive z points to the right and positive y points down). To see how to transform this into camera coordinate in opencv coordinate system ( positive x points to the right, positive y points down and positive z to the front) see the demo script load_data_demo.py. Numbering is identical as for pose_joints_2d.

pose_joints_VSBL:
  visibility label for each joint. Obtained using heuristic and thus might contain false labels.

scale:
  a measure to compare scales of synthetic humans (based on 2D head bounding box)

segm_EXR:
  exr image in which each pixel is assigned a value corresponding to the SMPL body segmentation (see Figure 3 in multi-human flow)

shapes:
  dictionary containing SMPL-H shapes for each synthetic human (note that the shapes distributions are gender specific). Due to a bug in the data generation code for most models the shape distribution is limited to [0,1]. The bug was removed for the code published here.

subj_pelvis_T:
  translation of each synthetic human (Please refer to the demo script load_smpl_and_render_demo.py for more instructions).

subj_ZRot:
  z rotation. Rotations to be applied to the SMPL body (Please refer to the demo script load_smpl_and_render_demo.py for more instructions).
