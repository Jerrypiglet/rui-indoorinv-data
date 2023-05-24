'''
Works with Mitsuba/Blender scenes
'''
import sys

# host = 'mm1'
host = 'apple'

from lib.global_vars import PATH_HOME_dict, INV_NERF_ROOT_dict, MONOSDF_ROOT_dict, OR_RAW_ROOT_dict
PATH_HOME = PATH_HOME_dict[host]
sys.path.insert(0, PATH_HOME)
OR_RAW_ROOT = OR_RAW_ROOT_dict[host]
INV_NERF_ROOT = INV_NERF_ROOT_dict[host]
MONOSDF_ROOT = MONOSDF_ROOT_dict[host]

from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)
from lib.utils_misc import str2bool, white_magenta
import argparse

from lib.class_realScene3D import realScene3D

from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d
from lib.class_visualizer_scene_3D_plt import visualizer_scene_3D_plt

from lib.class_eval_rad import evaluator_scene_rad
from lib.class_eval_monosdf import evaluator_scene_monosdf
from lib.class_eval_inv import evaluator_scene_inv
from lib.class_eval_scene import evaluator_scene_scene

from lib.class_renderer_mi_mitsubaScene_3D import renderer_mi_mitsubaScene_3D
from lib.class_renderer_blender_mitsubaScene_3D import renderer_blender_mitsubaScene_3D

parser = argparse.ArgumentParser()
# visualizers
parser.add_argument('--vis_3d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to visualize 3D with plt for debugging')
parser.add_argument('--vis_3d_o3d', type=str2bool, nargs='?', const=True, default=True, help='whether to visualize in open3D')
parser.add_argument('--vis_2d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to show (1) pixel-space modalities (2) projection onto one image (e.g. layout, object bboxes), with plt')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=False, help='')
# options for visualizers
parser.add_argument('--pcd_color_mode_dense_geo', type=str, default='rgb', help='colormap for all points in fused geo')
parser.add_argument('--if_set_pcd_color_mi', type=str2bool, nargs='?', const=True, default=False, help='if create color map for all points of Mitsuba; required: input_colors_tuple')
# parser.add_argument('--if_add_rays_from_renderer', type=str2bool, nargs='?', const=True, default=False, help='if add camera rays and emitter sample rays from renderer')

parser.add_argument('--split', type=str, default='train', help='')

# renderer (mi/blender)
parser.add_argument('--render_2d', type=str2bool, nargs='?', const=True, default=False, help='render 2D modalities')
parser.add_argument('--renderer', type=str, default='blender', help='mi, blender')

# ==== Evaluators
parser.add_argument('--if_add_rays_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if add rays from evaluating MLPs (e.g. emitter radiance rays')
parser.add_argument('--if_add_est_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if add estimations from evaluating MLPs (e.g. ennvmaps)')
parser.add_argument('--if_add_color_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if colorize mesh vertices with values from evaluator')
# evaluator for rad-MLP
parser.add_argument('--eval_rad', type=str2bool, nargs='?', const=True, default=False, help='eval trained rad-MLP')
# evaluator for inv-MLP
parser.add_argument('--eval_inv', type=str2bool, nargs='?', const=True, default=False, help='eval trained inv-MLP')
# evaluator over scene shapes
parser.add_argument('--eval_scene', type=str2bool, nargs='?', const=True, default=False, help='eval over scene (e.g. shapes for coverage)')
parser.add_argument('--eval_scene_sample_type', type=str, default='vis_count', help='')
# evaluator for MonoSDF
parser.add_argument('--eval_monosdf', type=str2bool, nargs='?', const=True, default=False, help='eval trained MonoSDF')

# debug
parser.add_argument('--if_debug_info', type=str2bool, nargs='?', const=True, default=False, help='if show debug info')

# utils
parser.add_argument('--if_sample_poses', type=str2bool, nargs='?', const=True, default=False, help='if sample camera poses instead of loading from pose file')
parser.add_argument('--export', type=str2bool, nargs='?', const=True, default=False, help='if export entire scene to mitsubaScene data structure')
parser.add_argument('--export_format', type=str, default='monosdf', help='')
parser.add_argument('--export_appendix', type=str, default='', help='')
parser.add_argument('--force', type=str2bool, nargs='?', const=True, default=False, help='if force to overwrite existing files')

opt = parser.parse_args()

dataset_root = Path(PATH_HOME) / 'data/real'
xml_root = Path(PATH_HOME) / 'data/real'

'''
default
'''
eval_models_dict = {}
# monosdf_shape_dict = {}
shape_file = ''
frame_id_list = []
invalid_frame_id_list = []
invalid_frame_idx_list = []
hdr_radiance_scale = 1.
sdr_radiance_scale = 1.
if_rc = False; pcd_file = ''; 

if_reorient_y_up = False  #  images/demo_realScene_after_center_scale_reorient.png
reorient_blender_angles = [] # images/demo_blender_rotate.png; Open the input .ply/.obj file in Blender, rotate object axes to align with world, write down the angles
if_reorient_y_up_skip_shape = False # do not transform shape; only transform posesa


# scene_name = 'IndoorKitchen_v1'; hdr_radiance_scale = 10.
# # if_rc = False; pcd_file = 'reconstuction_auto/dense/2/fused.ply'; pose_file = ('json', 'transforms.json')
# if_rc = True; pcd_file = 'RealityCapture/real_kitchen.ply'; pose_file = ('bundle', 'RealityCapture/real_kitchen_bundle.out')
# frame_id_list = [5, 6, 7]

# scene_name = 'IndoorKitchen_v2'; hdr_radiance_scale = 3.
# if_rc = False; pcd_file = ''; pose_file = ('json', 'transforms.json')
# shape_file = dataset_root / 'RESULTS_monosdf/20230301-135857-mm1-EVAL-20230301-035029IndoorKitchen_v2_HDR_grids_trainval.ply'

# scene_name = 'IndoorKitchen_v2_2'; hdr_radiance_scale = 2.
# if_rc = False; pcd_file = ''; pose_file = ('json', 'transforms.json')
# shape_file = dataset_root / 'RESULTS_monosdf/20230303-013146-mm1-EVAL-IndoorKitchen_v2_2_HDR_grids_trainval_tmp.ply'

# scene_name = 'IndoorKitchen_v2_3'; hdr_radiance_scale = 0.5
# if_rc = False; pcd_file = ''; pose_file = ('json', 'transforms.json')
# shape_file = dataset_root / 'RESULTS_monosdf/20230303-233627-mm3-IndoorKitchen_v2_3_RE_HDR_grids_trainval_gamma2_L1_Lr1e-4S25.ply'

# scene_name = 'IndoorKitchen_v2_3_Dark'; hdr_radiance_scale = 1.
# scene_name = 'IndoorKitchen_v2_3_Dark_v2'; hdr_radiance_scale = 1.
# if_rc = False; pcd_file = ''; pose_file = ('json', 'transforms.json')
# shape_file = dataset_root / 'RESULTS_monosdf/20230303-233627-mm3-IndoorKitchen_v2_3_RE_HDR_grids_trainval_gamma2_L1_Lr1e-4S25.ply'

# scene_name = 'IndoorKitchen_v2_merged'; hdr_radiance_scale = 0.5
# if_rc = False; pcd_file = ''; pose_file = ('json', 'transforms.json')
# # shape_file = dataset_root / 'RESULTS_monosdf/20230305-180337-mm3-IndoorKitchen_v2_MERGED_HDR_grids_trainval_gamma2_L1_Lr1e-4S25.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/20230305-180337-mm3-IndoorKitchen_v2_MERGED_HDR_grids_trainval_gamma2_L1_Lr1e-4S25_2.ply'

# scene_name = 'DormRoom'; hdr_radiance_scale = 0.5
# if_rc = False; pcd_file = ''; pose_file = ('json', 'transforms.json')
# # shape_file = dataset_root / 'RESULTS_monosdf/20230305-142814-mm1-EVAL-20230304-173919DormRoom_v1_SDR_grids_trainval.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/20230305-141754-mm1-EVAL-20230304-135016DormRoom_v1_HDR_grids_trainval.ply'

# scene_name = 'ConferenceRoom'; hdr_radiance_scale = 0.7
# if_rc = False; pcd_file = ''; pose_file = ('json', 'transforms.json')

'''
------ final
'''
# scene_name = 'IndoorKitchen_v2_final_supergloo'; hdr_radiance_scale = 0.5
# pose_file = ('json', 'transforms_superglue.json')

# scene_name = 'DormRoom_v2_final'; hdr_radiance_scale = 0.5
# pose_file = ('json', 'transforms_colmap.json')
# shape_file = dataset_root / 'RESULTS_monosdf/20230306-022845-K-DormRoom_v2_final_supergloo_HDR_grids_trainval_tmp.ply'

# scene_name = 'DormRoom_v2_final_supergloo'; hdr_radiance_scale = 0.5
# pose_file = ('json', 'transforms.json'); 
# invalid_frame_idx_list = [187, 188, 189, 190, 197, 198, 199, 200, 201, 202, 222, 223, 225, 215, 214, 213, 212, 211, 210, 204, 203, 205]
# # shape_file = dataset_root / 'RESULTS_monosdf/20230306-022845-K-DormRoom_v2_final_supergloo_HDR_grids_trainval_tmp.ply'
# # shape_file = dataset_root / 'RESULTS_monosdf/20230306-123126-K-DormRoom_v2_final_supergloo_FIXED_SDR_grids_trainval.ply'
# # shape_file = dataset_root / 'RESULTS_monosdf/20230306-174427-mm1-DormRoom_v2_final_supergloo_FIXED2_SDR_grids_trainval_gamma2_L1_Lr1e-4S25.ply' # experimental
# # shape_file = dataset_root / 'RESULTS_monosdf/20230306-234504-K-DormRoom_v2_final_supergloo_FIXED3_SDR_grids_trainval_tmp.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/20230307-044433-mm1-CONT20230306-234504-DormRoom_v2_final_supergloo_FIXED3_SDR_grids_trainval_tmp.ply'

# scene_name = 'IndoorKitchenV3_final'; hdr_radiance_scale = 0.5
# pose_file = ('json', 'transforms_colmap.json')
# shape_file = dataset_root / 'RESULTS_monosdf/20230306-042253-K-IndoorKitchenV3_final_HDR_grids_trainval.ply'

# scene_name = 'IndoorKitchenV3_final_supergloo'; hdr_radiance_scale = 0.5
# pose_file = ('json', 'transforms_superglue.json'); 
# invalid_frame_idx_list = [243, 105, 158, 198, 200, 209, 208, 217, 218]
# shape_file = dataset_root / 'RESULTS_monosdf/20230306-040256-K-IndoorKitchenV3_final_supergloo_HDR_grids_trainval.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/20230306-052812-K-IndoorKitchenV3_final_supergloo_HDR_grids_trainval_FIXlast.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/20230306-040343-K-IndoorKitchenV3_final_supergloo_SDR_grids_trainval.ply'

# scene_name = 'IndoorKitchenV3_final_supergloo_RE'; hdr_radiance_scale = 0.5
# pose_file = ('json', 'transforms_superglue_RE.json'); 
# invalid_frame_idx_list = [243, 105, 158, 198, 200, 209, 208, 217, 218]
# shape_file = dataset_root / 'RESULTS_monosdf/20230306-040256-K-IndoorKitchenV3_final_supergloo_HDR_grids_trainval.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/20230306-052812-K-IndoorKitchenV3_final_supergloo_HDR_grids_trainval_FIXlast.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/20230306-040343-K-IndoorKitchenV3_final_supergloo_SDR_grids_trainval.ply'

# scene_name = 'ConferenceRoomV2_final'; hdr_radiance_scale = 0.5
# pose_file = ('json', 'transforms_colmap.json')
# shape_file = dataset_root / 'RESULTS_monosdf/'

# +++++ ConferenceRoomV2_final_supergloo_aligned +++++ [SUPP]
scene_name = 'ConferenceRoomV2_final_supergloo'; hdr_radiance_scale = 2.
pose_file = ('json', 'transforms_superglue.json')
# shape_file = dataset_root / 'RESULTS_monosdf/20230306-060630-K-ConferenceRoomV2_final_supergloo_HDR_grids_trainval.ply'
shape_file = dataset_root / 'RESULTS_monosdf/20230306-072825-K-ConferenceRoomV2_final_supergloo_SDR_grids_trainval.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/20230306-152848-mm1-EVAL-20230306-072825ConferenceRoomV2_final_supergloo_SDR_grids_trainval.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/conference-old.obj'
# shape_file = '/Volumes/RuiT7/ICCV23/real/EXPORT_mitsuba/scene_milo.obj'
# if_reorient_y_up = True; reorient_blender_angles = [-175, -140, 2.85] # images/demo_blender_rotate.png
# if_reorient_y_up_skip_shape = True
# emitter_thres = 4.
# # frame_id_list = [9, 161] # BRDF
# # frame_id_list = [180, 68] # re-rendering + relighting

'''
Supplementary
'''
# scene_name = 'IndoorKitchenV4'; hdr_radiance_scale = 1 # BETTER intrinsics
# # # pose_file = ('json', 'transforms_superglue/transforms_bright.json'); invalid_frame_idx_list = [261, 254, 255, 230, 231, 180]
# # shape_file = dataset_root / 'RESULTS_monosdf/20230306-190430-mm1-IndoorKitchenV4_SDR_grids_trainval.ply'
# pose_file = ('json', 'transforms_colmap/transforms.json'); invalid_frame_idx_list = [232, 241]; invalid_frame_id_list = list(range(265, 285)) # need to exclude 265 from both...
# shape_file = dataset_root / 'RESULTS_monosdf/20230307-005359-mm1-IndoorKitchenV4_COLMAP_SDR_grids_trainval.ply'

# ------------>>
# +++++ IndoorKitchenV4_2_aligned +++++
# scene_name = 'IndoorKitchenV4_2'; hdr_radiance_scale = 1
# pose_file = ('json', 'transforms_bright.json') # colmap
# # shape_file = dataset_root / 'RESULTS_monosdf/20230309-170742-mm3-IndoorKitchenV4_2_SDR_grids_trainval.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/20230311-014753-K-IndoorKitchenV4_2_aligned_SDR_grids_trainval.ply'
# if_reorient_y_up = True; reorient_blender_angles = [-11.2, -43, -181] # images/demo_blender_rotate.png
# if_reorient_y_up_skip_shape = True
# <<------------

# ------------>>
# +++++ DormRoom_v2_1_betterK_supergloo_old_aligned +++++ [SUPP]
# scene_name = 'DormRoom_v2_1_betterK_supergloo'; hdr_radiance_scale = 0.5 # BETTER intrinsics
# invalid_frame_idx_list = [10, 11, 12, 13, 14, 16, 17, 18, 84]
# pose_file = ('json', 'transforms.json'); 
# # # shape_file = dataset_root / 'RESULTS_monosdf/20230309-185710-mm1-DormRoom_v2_1_betterK_supergloo_SDR_grids_trainval_gamma2_L1_Lr1e-4S25.ply'
# # # shape_file = dataset_root / 'RESULTS_monosdf/20230309-200858-mm1-DormRoom_v2_1_betterK_supergloo_SDR_grids_trainval_gamma2_L1_Lr1e-4S25.ply'
# # shape_file = dataset_root / 'RESULTS_monosdf/20230309-232118-mm1-DormRoom_v2_1_betterK_supergloo_FIXED_SDR_grids_trainval_gamma2_L1_Lr1e-4S25.ply'
# # shape_file = dataset_root / 'RESULTS_monosdf/20230311-020104-K-DormRoom_v2_1_betterK_supergloo_aligned_SDR_grids_trainval.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/20230310-035459-K-DormRoom_v2_1_betterK_supergloo_FIXED_SDR_grids_trainval.ply' # OLD
# if_reorient_y_up = True; reorient_blender_angles = [165, 36.5, -3.1] # images/demo_blender_rotate.png
# if_reorient_y_up_skip_shape = True
# <<------------

# ------------>>
# invalid_frame_id_list = list(range(267, 273)) + list(range(274, 295)) # original lighting
# # invalid_frame_id_list = list(range(267)) + [273] + list(range(288, 305)) # blackboard lighting
# # invalid_frame_id_list = list(range(288)) + list(range(295, 305)) # lamp lighting

# # # +++++ ClassRoom_aligned +++++ [SUPP]
# scene_name = 'ClassRoom'; hdr_radiance_scale = 3 # BETTER intrinsics
# assert invalid_frame_id_list != []
# # shape_file = dataset_root / 'RESULTS_monosdf/20230310-035028-mm1-ClassRoom_SDR_grids_trainval.ply'
# pose_file = ('json', 'transforms_colmap.json'); 
# # shape_file = dataset_root / 'RESULTS_monosdf/classroom.obj'; 
# shape_file = dataset_root / 'RESULTS_monosdf/20230310-162753-K-ClassRoom_aligned_SDR_grids_trainval.ply'; 
# if_reorient_y_up = True; reorient_blender_angles = [-184, -19.7, -0.757] # images/demo_blender_rotate.png
# if_reorient_y_up_skip_shape = True
# # frame_id_list = [120, 235, 27, 92]
# # frame_id_list = [278, 275] # for relighting
# emitter_thres = 2.

# +++++ ClassRoom_supergloo_aligned +++++
# scene_name = 'ClassRoom_supergloo'; hdr_radiance_scale = 3 # BETTER intrinsics
# invalid_frame_id_list = list(range(267, 273)) + list(range(274, 295)) # original lighting
# pose_file = ('json', 'transforms_superglue.json')
# shape_file = dataset_root / 'RESULTS_monosdf/20230310-035030-mm1-ClassRoom_supergloo_SDR_grids_trainval_gamma2_L1_Lr1e-4S25.ply'
# if_reorient_y_up = True; reorient_blender_angles = [-181, 16.5, -2.05] # images/demo_blender_rotate.png
# if_reorient_y_up_skip_shape = True
# <<------------

# ------------>>
# scene_name = 'ConferenceRoomV2_betterK'; hdr_radiance_scale = 0.5
# pose_file = ('json', 'transforms_colmap.json')
# shape_file = dataset_root / 'RESULTS_monosdf/'

# scene_name = 'ConferenceRoomV2_betterK_supergloo'; hdr_radiance_scale = 0.5
# pose_file = ('json', 'transforms_supergloo.json')
# shape_file = dataset_root / 'RESULTS_monosdf/'
# <<------------

# ------------>>
# +++++ Bedroom_supergloo_aligned +++++
# scene_name = 'Bedroom_supergloo'; hdr_radiance_scale = 8; sdr_radiance_scale = 4
# invalid_frame_id_list = [198, 199, 200] # original lighting
# invalid_frame_idx_list = [7, 8, 171, 172, 173, 174, 175] # bad poses
# pose_file = ('json', 'transforms_supergloo.json')
# shape_file = dataset_root / 'RESULTS_monosdf/20230311-164356-K-Bedroom_supergloo_aligned_SDR_grids_trainval.ply'
# if_reorient_y_up = True; reorient_blender_angles = [172, 55.3, -1.07] # images/demo_blender_rotate.png
# if_reorient_y_up_skip_shape = True

# scene_name = 'Bedroom'; hdr_radiance_scale = 8; sdr_radiance_scale = 4
# invalid_frame_id_list = [198, 199, 200] # original lighting
# invalid_frame_idx_list = [18, 22, 23, 24, 25, 26] # bad poses
# pose_file = ('json', 'transforms_colmap.json')
# shape_file = dataset_root / 'RESULTS_monosdf/20230311-164210-K-Bedroom_aligned_SDR_grids_trainval.ply'
# if_reorient_y_up = True; reorient_blender_angles = [171, 177, -361] # images/demo_blender_rotate.png
# if_reorient_y_up_skip_shape = True

# +++++ Bedroom_MORE_aligned +++++ [SUPP]
# scene_name = 'Bedroom_MORE'; hdr_radiance_scale = 1; sdr_radiance_scale = 2
# invalid_frame_id_list = [198, 199, 200, 202, 203, 206, 207, 209, 217, ] # original lighting
# invalid_frame_idx_list = [18, 22, 23, 24, 25, 26] # bad poses
# pose_file = ('json', 'transforms_colmap.json')
# shape_file = dataset_root / 'RESULTS_monosdf/20230312-132325-mm3-Bedroom_MORE_aligned_HDR_grids_trainval.ply'
# if_reorient_y_up = True; reorient_blender_angles = [-197, 177, -10.2]
# if_reorient_y_up_skip_shape = True

# scene_name = 'Bedroom_MORE_supergloo'; hdr_radiance_scale = 1; sdr_radiance_scale = 2
# invalid_frame_id_list = [198, 199, 200, 202, 203, 206, 207, 209, 217, ] # original lighting
# invalid_frame_idx_list = [7, 8, 171, 172, 173, 174, 175] # bad poses
# pose_file = ('json', 'transforms_supergloo.json')
# <<------------

# scene_name = 'CSEKitchen'; hdr_radiance_scale = 4
# pose_file = ('json', 'transforms_colmap.json') # colmap
# invalid_frame_id_list = [1, 13]
# invalid_frame_id_list += [165, 166, 167, 170, 171, 172, 173, 185] # novel lighting ALL; original lighting: 164, 168, 169, 174, 175, 176; 185: lamp off
# shape_file = dataset_root / 'RESULTS_monosdf/20230313-214417-mm3-RE-CSEKitchen_HDR_grids_trainval.ply'
# # shape_file = dataset_root / 'RESULTS_monosdf/20230313-220318-mm3-CSEKitchen_HDR_grids_trainval_2.ply'
# if_reorient_y_up = True; reorient_blender_angles = [-24, 30.1, -178]
# # if_reorient_y_up_skip_shape = True

# scene_name = 'CSEKitchen_supergloo'; hdr_radiance_scale = 4; sdr_radiance_scale = 2
# pose_file = ('json', 'transforms_supergloo.json') # supergloo
# invalid_frame_id_list = [1, 13]
# invalid_frame_id_list += [165, 166, 167, 170, 171, 172, 173, 185] # novel lighting ALL; original lighting: 164, 168, 169, 174, 175, 176; 185: lamp off
# shape_file = dataset_root / 'RESULTS_monosdf/20230313-220314-mm3-CSEKitchen_supergloo_HDR_grids_trainval.ply'
# # shape_file = dataset_root / 'RESULTS_monosdf/20230313-222459-mm3-CSEKitchen_supergloo_HDR_grids_trainval.ply'
# if_reorient_y_up = True; reorient_blender_angles = [-25.4, 30, -179]
# if_reorient_y_up_skip_shape = False

'''
OBSELETE
'''
# scene_name = 'ConferenceRoomV2_final_MORE'; hdr_radiance_scale = 1.; sdr_radiance_scale = 0.5
# # pose_file = ('json', 'transforms_colmap.json')
# pose_file = ('json', 'transforms_superglue.json')
# # shape_file = dataset_root / 'RESULTS_monosdf/20230307-022305-mm1-ConferenceRoomV2_final_MORE_SDR_grids_trainval.ply'
# shape_file = dataset_root / 'RESULTS_monosdf/20230307-030111-mm1-ConferenceRoomV2_final_MORE_HDR_grids_trainval.ply'
# if opt.export_format == 'mitsuba': invalid_frame_idx_list = list(range(190, 227))

frame_id_list = [0]

im_params_dict={
    'hdr_radiance_scale': hdr_radiance_scale, 
    'sdr_radiance_scale': sdr_radiance_scale, 
    # V2_2/V2_3
    'im_H_load_hdr': 512, 'im_W_load_hdr': 768, 
    'im_H_load_sdr': 512, 'im_W_load_sdr': 768, 
    'im_H_load': 512, 'im_W_load': 768, 
}
if opt.export_format == 'mitsuba':
    im_params_dict.update({
        'im_H_resize': 360, 'im_W_resize': 540, # inv-nerf
    })
elif opt.export_format == 'lieccv22':
    im_params_dict.update({
        'im_H_resize': 240, 'im_W_resize': 320, 
    })
else:
    im_params_dict.update({
        'im_H_resize': 512, 'im_W_resize': 768, # monosdf
    })

scene_obj = realScene3D(
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'dataset_root': dataset_root}, 
    scene_params_dict={
        'scene_name': scene_name, 
        'frame_id_list': frame_id_list, 
        # 'mitsuba_version': '3.0.0', 
        # 'intrinsics_path': Path(PATH_HOME) / 'data/real' / scene_name / 'intrinsic_mitsubaScene.txt', 
        'axis_up': 'y+', # WILL REORIENT TO y+
        # 'extra_transform': np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32), # z=y, y=x, x=z # convert from y+ (native to indoor synthetic) to z+
        'invalid_frame_id_list': invalid_frame_id_list, 
        'invalid_frame_idx_list': invalid_frame_idx_list,
        'pose_file': pose_file, 
        'pcd_file': pcd_file, 
        'shape_file': shape_file, 
        'if_rc': if_rc, 
        
        'if_autoscale_scene': False, # not doing this for exporting, to avoid potential bugs (export to monosdf will handling scale)
        
        'if_reorient_y_up': if_reorient_y_up,  #  images/demo_realScene_after_center_scale_reorient.png
        'reorient_blender_angles': reorient_blender_angles, # images/demo_blender_rotate.png; Open the input .ply/.obj file in Blender, rotate object axes to align with world, write down the angles
        'if_reorient_y_up_skip_shape': if_reorient_y_up_skip_shape, # do not transform shape; only transform posesa
        
        # 'normal_up_frame_info': {'frame_id': 3, 'normal_up_hw_1': (0.5, 0.35), 'normal_up_hw_2': (1., 0.6)}, # find one image with mostly floor within the desginated region
        # 'normal_left_frame_info': {'frame_id': 8, 'normal_left_hw_1': (0., 0.), 'normal_left_hw_2': (0.5, 0.5)}, # find one image with mostly floor within the desginated region
        }, 
    mi_params_dict={
        # 'if_also_dump_xml_with_lit_area_lights_only': True,  # True: to dump a second file containing lit-up lamps only
        'debug_render_test_image': True if shape_file != '' else False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'debug_dump_mesh': False, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        # 'if_sample_rays_pts': True if shape_file != '' else False, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': False, # [depend on if_sample_rays_pts] True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
    modality_list = [
        'poses', 
        'im_hdr', 
        'im_sdr', 
        'shapes', # objs + emitters, geometry shapes + emitter properties
        ], 
    modality_filename_dict = {
        # 'poses', 
        'im_hdr': 'merged_images/img_%04d.exr', 
        'im_sdr': 'png_images/img_%04d.png', 
        # 'shapes', # objs + emitters, geometry shapes + emitter properties
    }, 
    im_params_dict=im_params_dict, 
    cam_params_dict={
        'near': 0.2, 'far': 2., # [in a unit box]
        # 'near': 0.5, 'far': 20., # [in a unit box]
        # ==> if sample poses and render images 
        'if_sample_poses': False, # True to generate camera poses following Zhengqin's method (i.e. walking along walls)
        # 'sample_pose_num': 200 if 'train' in opt.split else 20, # Number of poses to sample; set to -1 if not sampling
        # 'sample_pose_if_vis_plt': True, # images/demo_sample_pose.png, images/demo_sample_pose_bathroom.png
    }, 
    lighting_params_dict={
    }, 
    shape_params_dict={
        },
    emitter_params_dict={
        },
)

eval_return_dict = {}

'''
Evaluator for scene
'''
if opt.eval_scene:
    evaluator_scene = evaluator_scene_scene(
        host=host, 
        scene_object=scene_obj, 
    )

    '''
    sample visivility to camera centers on vertices
    [!!!] set 'mesh_color_type': 'eval-vis_count'
    '''
    if opt.export:
        eval_scene_sample_type = 'face_normal'
    _ = evaluator_scene.sample_shapes(
        sample_type=opt.eval_scene_sample_type, # ['']
        # sample_type='vis_count', # ['']
        # sample_type='t', # ['']
        # sample_type='face_normal', # ['']
        shape_params={
        }
    )
    for k, v in _.items():
        if k in eval_return_dict:
            eval_return_dict[k].update(_[k])
        else:
            eval_return_dict[k] = _[k]
            
    if 'face_normals_flipped_mask' in eval_return_dict and opt.export:
        face_normals_flipped_mask = eval_return_dict['face_normals_flipped_mask']
        assert face_normals_flipped_mask.shape[0] == scene_obj.faces_list[0].shape[0]
        if np.sum(face_normals_flipped_mask) > 0:
            validate_idx = np.where(face_normals_flipped_mask)[0][0]
            print(validate_idx, scene_obj.faces_list[0][validate_idx])
            scene_obj.faces_list[0][face_normals_flipped_mask] = scene_obj.faces_list[0][face_normals_flipped_mask][:, [0, 2, 1]]
            print(white_magenta('[FLIPPED] %d/%d inward face normals'%(np.sum(face_normals_flipped_mask), scene_obj.faces_list[0].shape[0])))
            print(validate_idx, '->', scene_obj.faces_list[0][validate_idx])

if opt.export:
    from lib.class_exporter import exporter_scene
    exporter = exporter_scene(
        scene_object=scene_obj,
        format=opt.export_format, 
        modality_list = [
            'poses', 
            'im_hdr', 
            'im_sdr', 
            # 'im_mask', 
            # 'shapes', 
            # 'mi_normal', 
            # 'mi_depth', 
            ], 
        if_force=opt.force, 
        # convert from y+ (native to indoor synthetic) to z+
        # extra_transform = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),  # y=z, z=x, x=y
        # extra_transform = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32),  # z=y, y=x, x=z
        
    )
    if opt.export_format == 'monosdf':
        exporter.export_monosdf_fvp_mitsuba(
            # split=opt.split, 
            format='monosdf',
            modality_list = [
                'poses', 
                'im_hdr', 
                'im_sdr', 
                'im_mask', 
                'shapes', 
                'mi_normal' if shape_file != '' else '', 
                'mi_depth' if shape_file != '' else '', 
                ], 
            appendix=opt.export_appendix, 
            )
    if opt.export_format == 'mitsuba':
        exporter.export_monosdf_fvp_mitsuba(
            # split=opt.split, 
            format='mitsuba',
            modality_list = [
                'poses', 
                'im_hdr', 
                'im_sdr', 
                'im_mask', 
                'shapes', 
                'mi_normal' if shape_file != '' else '', 
                'mi_depth' if shape_file != '' else '', 
                ], 
            appendix=opt.export_appendix, 
            )
    if opt.export_format == 'fvp':
        exporter.export_monosdf_fvp_mitsuba(
            # split=opt.split, 
            format='fvp',
            modality_list = [
                'poses', 
                'im_hdr', 
                'im_sdr', 
                'im_mask', 
                'shapes', 
                'lighting', # new lights
                ], 
            appendix=opt.export_appendix, 
        )
    if opt.export_format == 'lieccv22':
        exporter.export_lieccv22(
            modality_list = [
            'im_sdr', 
            'mi_depth', 
            'mi_seg', 
            'lighting', # ONLY available after getting BRDFLight result from testRealBRDFLight.py
            'emission', 
            ], 
            split='real', 
            assert_shape=(240, 320),
            window_area_emitter_id_list=[], # not available for real images
            merge_lamp_id_list=[], # not available for real images
            emitter_thres = emitter_thres, # same as offered to fvp
            BRDF_results_folder='BRDFLight_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000', # transfer this back once get BRDF results
            # BRDF_results_folder='BRDFLight_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000_optimize', # transfer this back once get BRDF results
            # center_crop_HW=(240, 320), 
            if_no_gt_appendix=True, 
            appendix=opt.export_appendix, 
        )
    
'''
Matploblib 2D viewer
'''
if opt.vis_2d_plt:
    visualizer_2D = visualizer_scene_2D(
        scene_obj, 
        modality_list_vis=[
            'im', 
            # 'layout', 
            # 'shapes', 
            # 'albedo', 
            # 'roughness', 
            # 'emission', 
            # 'depth', 
            # 'normal', 
            'mi_depth' if shape_file != '' else '', 
            'mi_normal' if shape_file != '' else '', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            # 'lighting_SG', # convert to lighting_envmap and vis: images/demo_lighting_SG_envmap_2D_plt.png
            # 'lighting_envmap', # renderer with mi/blender: images/demo_lighting_envmap_mitsubaScene_2D_plt.png
            # 'seg_area', 'seg_env', 'seg_obj', 
            # 'mi_seg_area' if shape_file != '' else '', 
            # 'mi_seg_env' if shape_file != '' else '', 
            # 'mi_seg_obj' if shape_file != '' else '', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
            ], 
        frame_idx_list=[0, 1, 2, 3, 4], 
        # frame_idx_list=[0], 
        # frame_idx_list=[6, 10, 12], 
    )
    if opt.if_add_est_from_eval:
        for modality in ['lighting_envmap']:
            if modality in eval_return_dict:
                scene_obj.add_modality(eval_return_dict[modality], modality, 'EST')

    visualizer_2D.vis_2d_with_plt(
        lighting_params={
            'lighting_scale': 1., # rescaling the brightness of the envmap
            }, 
        other_params={
            # 'mi_normal_vis_coords': 'world-blender', 
            'mi_normal_vis_coords': 'opencv', 
            'mi_depth_if_sync_scale': False, 
            }, 
    )

'''
Open3D 3D viewer
'''
if opt.vis_3d_o3d:
    visualizer_3D_o3d = visualizer_scene_3D_o3d(
        scene_obj, 
        modality_list_vis=[
            'poses', 
            'shapes' if shape_file != '' else '', # bbox and (if loaded) meshs of shapes (objs + emitters SHAPES); CTRL + 9
            'mi', # mitsuba sampled rays, pts
            ], 
        if_debug_info=opt.if_debug_info, 
    )

    lighting_params_vis={
        }

    if opt.if_add_rays_from_eval:
        for key in eval_return_dict.keys():
            if 'rays_list' in key:
            # assert opt.eval_rad
                lpts, lpts_end = eval_return_dict[key]['v'], eval_return_dict[key]['v']+eval_return_dict[key]['d']*eval_return_dict[key]['l'].reshape(-1, 1)
                visualizer_3D_o3d.add_extra_geometry([
                    ('rays', {
                        'ray_o': lpts, 'ray_e': lpts_end, 'ray_c': np.array([[0., 0., 1.]]*lpts.shape[0]), # BLUE for EST
                    }),
                ]) 
        
    if opt.if_add_color_from_eval:
        if 'samples_v_dict' in eval_return_dict:
            assert opt.eval_rad or opt.eval_inv or opt.eval_scene or opt.eval_monosdf
            visualizer_3D_o3d.extra_input_dict['samples_v_dict'] = eval_return_dict['samples_v_dict']
        
    visualizer_3D_o3d.run_o3d(
        if_shader=opt.if_shader, # set to False to disable faycny shaders 
        cam_params={
            'if_cam_axis_only': False, 
            'cam_vis_scale': 0.6, 
            'if_cam_traj': False, 
            }, 
        lighting_params=lighting_params_vis, 
        shapes_params={
            # 'simply_mesh_ratio_vis': 1., # simply num of triangles to #triangles * simply_mesh_ratio_vis
            'if_meshes': True, # [OPTIONAL] if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': True, # [OPTIONAL] if show labels (False: only show bboxes)
            'if_voxel_volume': False, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png; USEFUL WHEN NEED TO CHECK SCENE SCALE (1 voxel = 1 meter)
            # 'if_ceiling': True if opt.eval_scene else False, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            # 'if_walls': True if opt.eval_scene else False, # [OPTIONAL] remove wall meshes to better see the furniture 
            'if_ceiling': True, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            'if_walls': True, # [OPTIONAL] remove wall meshes to better see the furniture 
            'if_sampled_pts': False, # [OPTIONAL] is show samples pts from scene_obj.sample_pts_list if available
            'mesh_color_type': 'eval-', # ['obj_color', 'face_normal', 'eval-' ('rad', 'emission_mask', 'vis_count', 't')]
        },
        emitter_params={
        },
        mi_params={
            'if_pts': False, # if show pts sampled by mi; should close to backprojected pts from OptixRenderer depth maps
            'if_pts_colorize_rgb': True, 
            'pts_subsample': 10,

            'if_cam_rays': False, 
            'cam_rays_if_pts': shape_file != '', # if cam rays end in surface intersections; set to False to visualize rays of unit length
            'cam_rays_subsample': 10, 
            
            'if_normal': False, 
            'normal_subsample': 50, 
            'normal_scale': 0.2, 

        }, 
    )
