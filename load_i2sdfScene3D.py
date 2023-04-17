'''
Works with I^2-SDF scenes
'''
import sys
from pathlib import Path

# host = 'mm1'
host = 'apple'

from lib.global_vars import PATH_HOME_dict# , INV_NERF_ROOT_dict, MONOSDF_ROOT_dict, OR_RAW_ROOT_dict
PATH_HOME = Path(PATH_HOME_dict[host])
sys.path.insert(0, str(PATH_HOME))

import numpy as np
np.set_printoptions(suppress=True)
import argparse
from pyhocon import ConfigFactory, ConfigTree
from lib.utils_misc import str2bool, white_magenta, check_exists

from lib.class_i2sdfScene3D import i2sdfScene3D

from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d

from lib.class_eval_scene import evaluator_scene_scene

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

# === after refactorization
parser.add_argument('--scene', type=str, default='scan332_bedroom_relight_0', help='load conf file: confs/i2sdf/\{opt.scene\}.conf')

opt = parser.parse_args()

DATASET = 'i2sdf'
conf_base_path = Path('confs/%s.conf'%DATASET); check_exists(conf_base_path)
CONF = ConfigFactory.parse_file(str(conf_base_path))
conf_scene_path = Path('confs/%s/%s.conf'%(DATASET, opt.scene)); check_exists(conf_scene_path)
conf_scene = ConfigFactory.parse_file(str(conf_scene_path))
CONF = ConfigTree.merge_configs(CONF, conf_scene)

dataset_root = Path(PATH_HOME) / CONF.data.dataset_root

frame_id_list = CONF.scene_params_dict.frame_id_list
invalid_frame_id_list = CONF.scene_params_dict.invalid_frame_id_list

# [debug] override
# frame_id_list = [12]
frame_id_list = list(range(12))
# frame_id_list = list(np.arange(25, 50, 1))

'''
update confs
'''

CONF.scene_params_dict.update({
    'split': opt.split, # train, val, train+val
    'frame_id_list': frame_id_list, 
    })

'''
create scene obj
'''
scene_obj = i2sdfScene3D(
    CONF = CONF, 
    if_debug_info = opt.if_debug_info, 
    host = host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'dataset_root': dataset_root}, 
    modality_list = [
        # 'im_hdr', 
        'im_sdr', 
        'poses', 
        # 'ks', 'kd',  
        # 'roughness', 
        'depth', 'normal', 
        'im_mask', 
        'tsdf', 
        
        'shapes', # load from tsdf shape
        'layout', # load from tsdf shape
        
        # 'layout', 
        # 'emission', 
        ], 
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
    _ = evaluator_scene.sample_shapes(
        sample_type='vis_count', # ['']
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


if opt.export:
    from lib.class_exporter import exporter_scene
    exporter = exporter_scene(
        scene_object=scene_obj,
        format=opt.export_format, 
        if_force=opt.force, 
        
    )
    if opt.export_format == 'monosdf':
        exporter.export_monosdf_fvp_mitsuba(
            # split=opt.split, 
            format='monosdf',
            modality_list = [
                # 'poses', 
                # 'im_hdr', 
                # 'im_sdr', 
                # 'im_mask', 
                # 'shapes', 
                'normal', 
                # 'depth', 
                'mi_normal', 
                # 'mi_depth', 
                ], 
            )
    if opt.export_format == 'fvp':
        exporter.export_monosdf_fvp_mitsuba(
            # split=opt.split, 
            format='fvp',
            modality_list = [
                'poses', 
                # 'im_hdr', 
                # 'im_sdr', 
                # 'im_mask', 
                'shapes', 
                ], 
            appendix=opt.export_appendix, 
        )
    if opt.export_format == 'lieccv22':
        exporter.export_lieccv22(
            modality_list = [
            'im_sdr', 
            'mi_seg', 
            'mi_depth', 
            'lighting', # ONLY available after getting BRDFLight result from testRealBRDFLight.py
            ], 
            split=opt.split, 
            assert_shape=(240, 320),
            window_area_emitter_id_list=CONF.scene_params_dict.window_area_emitter_id_list, # need to manually specify in XML: e.g. <emitter type="area" id="lamp_oven_0">
            merge_lamp_id_list=CONF.scene_params_dict.merge_lamp_id_list,  # need to manually specify in XML
            BRDF_results_folder='BRDFLight_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000', # transfer this back once get BRDF results
            # center_crop_HW=(240, 320), 
            if_no_gt_appendix=True, # do not append '_gt' to the end of the file name
        )
    
'''
Matploblib 2D viewer
'''
if opt.vis_2d_plt:
    visualizer_2D = visualizer_scene_2D(
        scene_obj, 
        modality_list_vis=[
            'im', 
            # 'ks', 
            # 'kd', 
            # 'roughness', 
            # 'im_mask', 
            'depth', 
            'normal', 
            'mi_depth', 
            'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            
            # 'emission', 
            # 'seg_area', 'seg_env', 'seg_obj', 
            ], 
        frame_idx_list=[0, 1, 2, 3, 4], 
    )

    visualizer_2D.vis_2d_with_plt(
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
            'tsdf',
            'shapes', # tsdf shape
            'layout', # from tsdf shape
            
            # 'mi', # mitsuba sampled rays, pts
            # 'dense_geo', # fused from 2D
            # 'lighting_envmap', # images/demo_lighting_envmap_o3d.png; arrows in pink
            # 'emitters', # emitter PROPERTIES (e.g. SGs, half envmaps)
            ], 
        if_debug_info=opt.if_debug_info, 
    )

    lighting_params_vis={
        'if_use_mi_geometry': True, 
        'if_use_loaded_envmap_position': True, # assuming lighting envmap endpoint position dumped by Blender renderer
        'subsample_lighting_pts_rate': 1, # change this according to how sparse the lighting arrows you would like to be (also according to num of frame_id_list)
        'subsample_lighting_wi_rate': 500, # subsample on lighting directions: too many directions (e.g. 128x256)
        # 'lighting_keep_ratio': 0.05, 
        # 'lighting_further_clip_ratio': 0.1, 
        'lighting_scale': 2, 
        # 'lighting_keep_ratio': 0.2, # - good for lighting_SG
        # 'lighting_further_clip_ratio': 0.3, 
        'lighting_keep_ratio': 0., # - good for lighting_envmap
        'lighting_further_clip_ratio': 0., 
        # 'lighting_keep_ratio': 0., # - debug
        # 'lighting_further_clip_ratio': 0., 
        'lighting_autoscale': False, 
        'lighting_if_show_hemisphere': True, # mainly to show hemisphere and local axes: images/demo_lighting_envmap_hemisphere_axes_o3d.png
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
        if 'lighting_fused_list' in eval_return_dict:
            assert opt.eval_rad
            for lighting_fused_dict in eval_return_dict['lighting_fused_list']:
                geometry_list = visualizer_3D_o3d.process_lighting(
                    lighting_fused_dict, 
                    lighting_params=lighting_params_vis, 
                    lighting_source='lighting_envmap', 
                    lighting_color=[0., 0., 1.], 
                    if_X_multiplied=True, 
                    if_use_pts_end=True,
                    )
                visualizer_3D_o3d.add_extra_geometry(geometry_list, if_processed_geometry_list=True)
        
    if opt.if_add_color_from_eval:
        if 'samples_v_dict' in eval_return_dict:
            assert opt.eval_rad or opt.eval_inv or opt.eval_scene or opt.eval_monosdf
            visualizer_3D_o3d.extra_input_dict['samples_v_dict'] = eval_return_dict['samples_v_dict']
        
    visualizer_3D_o3d.run_o3d(
        if_shader=opt.if_shader, # set to False to disable faycny shaders 
        cam_params={
            'if_cam_axis_only': False, 
            }, 
        lighting_params=lighting_params_vis, 
        shapes_params={
            # 'simply_mesh_ratio_vis': 1., # simply num of triangles to #triangles * simply_mesh_ratio_vis
            'if_meshes': True, # [OPTIONAL] if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': False, # [OPTIONAL] if show labels (False: only show bboxes)
            'if_voxel_volume': False, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png; USEFUL WHEN NEED TO CHECK SCENE SCALE (1 voxel = 1 meter)

            'if_ceiling': True if opt.eval_scene else False, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            'if_walls': True if opt.eval_scene else False, # [OPTIONAL] remove wall meshes to better see the furniture 
            # 'if_ceiling': False, 
            # 'if_walls': False, 
            # 'if_ceiling': True, 
            # 'if_walls': True, 

            'if_sampled_pts': False, # [OPTIONAL] is show samples pts from scene_obj.sample_pts_list if available
            'mesh_color_type': 'eval-', # ['obj_color', 'face_normal', 'eval-' ('rad', 'emission_mask', 'vis_count', 't')]
        },
        emitter_params={
            # 'if_half_envmap': False, # [OPTIONAL] if show half envmap as a hemisphere for window emitters (False: only show bboxes)
            # 'scale_SG_length': 2., 
            'if_sampling_emitter': True, 
            'scene_radiance_scale': CONF.scene_params_dict.scene_radiance_scale, 
            'max_plate': 32, 
        },
        mi_params={
            'if_pts': False, # if show pts sampled by mi; should close to backprojected pts from OptixRenderer depth maps
            'if_pts_colorize_rgb': True, 
            'pts_subsample': 10,
            # 'if_ceiling': True, # [OPTIONAL] remove ceiling points to better see the furniture 
            # 'if_walls': True, # [OPTIONAL] remove wall points to better see the furniture 

            'if_cam_rays': False, 
            'cam_rays_if_pts': True, # if cam rays end in surface intersections; set to False to visualize rays of unit length
            'cam_rays_subsample': 10, 
            
            'if_normal': False, 
            'normal_subsample': 50, 
            'normal_scale': 0.2, 

        }, 
    )

