'''
Load and visualize exported scenes in 'monosdf' format.
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
from pyhocon import ConfigFactory, ConfigTree
import argparse
from lib.utils_misc import str2bool, white_magenta, check_exists

from lib.class_monosdfScene3D import monosdfScene3D
from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d

from lib.class_eval_scene import evaluator_scene_scene

parser = argparse.ArgumentParser()
# visualizers
# parser.add_argument('--vis_3d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to visualize 3D with plt for debugging')
parser.add_argument('--vis_3d_o3d', type=str2bool, nargs='?', const=True, default=True, help='whether to visualize in open3D')
parser.add_argument('--vis_2d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to show (1) pixel-space modalities (2) projection onto one image (e.g. layout, object bboxes), with plt')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=False, help='')
# options for visualizers
parser.add_argument('--pcd_color_mode_dense_geo', type=str, default='rgb', help='colormap for all points in fused geo')
parser.add_argument('--if_set_pcd_color_mi', type=str2bool, nargs='?', const=True, default=False, help='if create color map for all points of Mitsuba; required: input_colors_tuple')

# evaluator over scene shapes
parser.add_argument('--eval_scene', type=str2bool, nargs='?', const=True, default=False, help='eval over scene (e.g. shapes for coverage)')
parser.add_argument('--if_add_color_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if colorize mesh vertices with values from evaluator')

# debug
parser.add_argument('--if_debug_info', type=str2bool, nargs='?', const=True, default=False, help='if show debug info')

# === after refactorization
parser.add_argument('--scene', type=str, default='ConferenceRoomV2_final_supergloo', help='load conf file: confs/real/\{opt.scene\}.conf')

opt = parser.parse_args()

DATASET = 'monosdf'
conf_base_path = Path('confs/%s.conf'%DATASET); check_exists(conf_base_path)
CONF = ConfigFactory.parse_file(str(conf_base_path))
CONF.im_params_dict.update({
    'im_H_load': 480, 'im_W_load': 640, 
    'im_H_resize': 480, 'im_W_resize': 640, 
})

'''
change those according to the scene you want to load
'''
dataset_root = Path(PATH_HOME) / 'data/i2-sdf-dataset/EXPORT_monosdf'
CONF.scene_params_dict.update({
    'scene_name': 'scan332_bedroom_relight_0', 
    'if_normalize_shape_depth_from_pose': True, # [!!!!!] True: normalize shape with loaded scale_mat
    })

CONF.mi_params_dict.update({
    'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
    'if_get_segs': True, # [depend on if_sample_rays_pts=True] True: to generate segs similar to those in openroomsScene2D.load_seg()
    })

mitsuba_scene = monosdfScene3D(
    CONF = CONF, 
    if_debug_info = opt.if_debug_info, 
    host = host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'dataset_root': dataset_root}, 
    modality_list = [
        'im_sdr', 
        'im_mask', 
        'poses', 
        'shapes', 
        'depth', 
        'normal', 
        'mi_depth', 
        'mi_normal', 
        
        # 'im_hdr', 
        ], 
)

eval_return_dict = {}

'''
Evaluator for scene
'''
if opt.eval_scene:
    evaluator_scene = evaluator_scene_scene(
        host=host, 
        scene_object=mitsuba_scene, 
    )

    '''
    sample visivility to camera centers on vertices
    '''
    _ = evaluator_scene.sample_shapes(
        sample_type='vis_count', # ['']
        # sample_type='t', # ['']
        shape_params={
        }
    )
    for k, v in _.items():
        if k in eval_return_dict:
            eval_return_dict[k].update(_[k])
        else:
            eval_return_dict[k] = _[k]

'''
Matploblib 2D viewer
'''
if opt.vis_2d_plt:
    visualizer_2D = visualizer_scene_2D(
        mitsuba_scene, 
        modality_list_vis=[
            'im', 
            'depth', 
            'normal', 
            'mi_depth', 
            'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            'mi_seg_obj', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
            ], 
        frame_idx_list=[0, 1, 2, 3, 4], 
        # frame_idx_list=[0], 
    )

    visualizer_2D.vis_2d_with_plt(
        other_params={
            'mi_normal_vis_coords': 'opencv', 
            'mi_depth_if_sync_scale': False, 
            }, 
    )

'''
Open3D 3D viewer
'''
if opt.vis_3d_o3d:
    visualizer_3D_o3d = visualizer_scene_3D_o3d(
        mitsuba_scene, 
        modality_list_vis=[
            'poses', 
            # 'lighting_SG', # images/demo_lighting_SG_o3d.png; arrows in blue
            # 'lighting_envmap', # images/demo_lighting_envmap_o3d.png; arrows in pink
            # 'layout', 
            'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters SHAPES)
            # 'emitters', # emitter PROPERTIES (e.g. SGs, half envmaps)
            # 'mi', # mitsuba sampled rays, pts
            ], 
        if_debug_info=opt.if_debug_info, 
    )

    if opt.if_add_color_from_eval:
        if 'samples_v_dict' in eval_return_dict:
            assert opt.eval_rad or opt.eval_inv or opt.eval_scene
            visualizer_3D_o3d.extra_input_dict['samples_v_dict'] = eval_return_dict['samples_v_dict']
        

    visualizer_3D_o3d.run_o3d(
        if_shader=opt.if_shader, # set to False to disable faycny shaders 
        cam_params={
            'if_cam_axis_only': False, 
            }, 
        shapes_params={
            # 'simply_mesh_ratio_vis': 1., # simply num of triangles to #triangles * simply_mesh_ratio_vis
            'if_meshes': True, # [OPTIONAL] if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': False, # [OPTIONAL] if show labels (False: only show bboxes)
            'if_voxel_volume': False, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png
            'if_ceiling': True, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            'if_walls': True, # [OPTIONAL] remove wall meshes to better see the furniture 
            'if_sampled_pts': False, # [OPTIONAL] is show samples pts from mitsuba_scene.sample_pts_list if available
            'mesh_color_type': 'eval-vis_count', # ['obj_color', 'face_normal', 'eval-rad', 'eval-emission_mask', 'eval-vis_count', 'eval-t']
            # 'mesh_color_type': 'eval-t', # ['obj_color', 'face_normal', 'eval-rad', 'eval-emission_mask', 'eval-vis_count]
        },
        mi_params={
            'if_pts': False, # if show pts sampled by mi; should close to backprojected pts from OptixRenderer depth maps
            'if_pts_colorize_rgb': True, 
            'pts_subsample': 10,
            # 'if_ceiling': False, # [OPTIONAL] remove ceiling points to better see the furniture 
            # 'if_walls': False, # [OPTIONAL] remove wall points to better see the furniture 

            'if_cam_rays': True, 
            'cam_rays_if_pts': True, # if cam rays end in surface intersections; set to False to visualize rays of unit length
            'cam_rays_subsample': 10, 
            
            'if_normal': False, 
            'normal_subsample': 50, 
            'normal_scale': 0.2, 
        }, 
    )

