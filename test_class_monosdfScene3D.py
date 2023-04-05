'''
work with Mitsuba/Blender scenes
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

from lib.class_monosdfScene3D import monosdfScene3D
from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d
from lib.class_eval_rad import evaluator_scene_rad
from lib.class_eval_inv import evaluator_scene_inv
from lib.class_eval_scene import evaluator_scene_scene
# from lib.class_renderer_mi_mitsubaScene_3D import renderer_mi_mitsubaScene_3D
# from lib.class_renderer_blender_mitsubaScene_3D import renderer_blender_mitsubaScene_3D
from lib.utils_misc import str2bool
import argparse

parser = argparse.ArgumentParser()
# visualizers
# parser.add_argument('--vis_3d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to visualize 3D with plt for debugging')
parser.add_argument('--vis_3d_o3d', type=str2bool, nargs='?', const=True, default=True, help='whether to visualize in open3D')
parser.add_argument('--vis_2d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to show (1) pixel-space modalities (2) projection onto one image (e.g. layout, object bboxes), with plt')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=False, help='')
# options for visualizers
parser.add_argument('--pcd_color_mode_dense_geo', type=str, default='rgb', help='colormap for all points in fused geo')
parser.add_argument('--if_set_pcd_color_mi', type=str2bool, nargs='?', const=True, default=False, help='if create color map for all points of Mitsuba; required: input_colors_tuple')
# parser.add_argument('--if_add_rays_from_renderer', type=str2bool, nargs='?', const=True, default=False, help='if add camera rays and emitter sample rays from renderer')

parser.add_argument('--split', type=str, default='train', help='')

# differential renderer
# parser.add_argument('--render_diff', type=str2bool, nargs='?', const=True, default=False, help='differentiable surface rendering')
# parser.add_argument('--renderer_option', type=str, default='PhySG', help='differentiable renderer option')

# renderer (mi/blender)
parser.add_argument('--render_2d', type=str2bool, nargs='?', const=True, default=False, help='render 2D modalities')
parser.add_argument('--renderer', type=str, default='blender', help='mi, blender')

# evaluator for rad-MLP
parser.add_argument('--eval_rad', type=str2bool, nargs='?', const=True, default=False, help='eval trained rad-MLP')
parser.add_argument('--if_add_rays_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if add rays from evaluating MLPs (e.g. emitter radiance rays')
parser.add_argument('--if_add_est_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if add estimations from evaluating MLPs (e.g. ennvmaps)')
parser.add_argument('--if_add_color_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if colorize mesh vertices with values from evaluator')
# evaluator for inv-MLP
parser.add_argument('--eval_inv', type=str2bool, nargs='?', const=True, default=False, help='eval trained inv-MLP')
# evaluator over scene shapes
parser.add_argument('--eval_scene', type=str2bool, nargs='?', const=True, default=False, help='eval over scene (e.g. shapes for coverage)')

# debug
parser.add_argument('--if_debug_info', type=str2bool, nargs='?', const=True, default=False, help='if show debug info')

# utils
parser.add_argument('--export', type=str2bool, nargs='?', const=True, default=False, help='if export entire scene to mitsubaScene data structure')
parser.add_argument('--export_format', type=str, default='monosdf', help='')
parser.add_argument('--export_appendix', type=str, default='', help='')
parser.add_argument('--force', type=str2bool, nargs='?', const=True, default=False, help='if force to overwrite existing files')

opt = parser.parse_args()

radiance_scale = 1.
# '''
# scannet scan 1~4 from MonoSDF
# '''
# base_root = Path(PATH_HOME) / 'data/scannet'
# xml_root = Path(PATH_HOME) / 'data/scannet'
# scan_id = 1
# scene_name = 'scan%d'%scan_id
# # frame_id_list = list(range(465))
# frame_id_list = list(range(0, 465, 40))
# # frame_id_list = [0]
# # shape_file = ('not-normalized', 'GTmesh/scene0050_00_vh_clean_2.ply')
# shape_file = ('normalized', 'ESTmesh/scan1.ply')

# dumped Monosdf scenes from ICCV23
base_root = Path(PATH_HOME) / 'data/real/EXPORT_monosdf'
scene_name = 'IndoorKitchenV4_2_aligned'
# frame_id_list = list(range(255))
frame_id_list = [0]
shape_file = ('not-normalized', 'scene_aligned.obj')

scene_obj = monosdfScene3D(
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root}, 
    scene_params_dict={
        'scene_name': scene_name, 
        # 'split': opt.split, 
        'frame_id_list': frame_id_list, 
        'axis_up': 'y+', 
        'pose_file': ('npz', 'cameras.npz'), # requires scaled Blender scene! in comply with Liwen's IndoorDataset (https://github.com/william122742/inv-nerf/blob/bake/utils/dataset/indoor.py)
        'shape_file': shape_file, 
        }, 
    mi_params_dict={
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': True, # [depend on if_sample_rays_pts] True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
    modality_list = [
        'im_hdr', 
        'im_sdr', 
        'shapes', # single scene file in ScanNet's case; objs + emitters, geometry shapes + emitter properties
        'depth', 
        'normal', 
        ], 
    modality_filename_dict = {
        'im_hdr': 'Image/%03d_0001.exr', 
        # 'im_sdr': 'Image/%03d_0001.png', 
        'im_sdr': '%03d_0001_rgb.png', 
        # 'depth': 'Depth/%03d_0001.exr', 
        'depth': '%03d_0001_depth.npy', 
        # 'normal': 'Normal/%03d_0001.exr', 
        'normal': '%03d_0001_normal.npy', 
    }, 
    im_params_dict={
        'im_H_load': 512, 'im_W_load': 768, 
        'im_H_resize': 512, 'im_W_resize': 768, 
        }, 
    cam_params_dict={
        'center_crop_type': 'no_crop', 
        'near': 0.1, 'far': 10., 
        'sample_pose_if_vis_plt': False, # images/demo_sample_pose.png
    }, 
    shape_params_dict={
        'if_load_obj_mesh': True, # set to False to not load meshes for objs (furniture) to save time
        },
)

eval_return_dict = {}

'''
Matploblib 2D viewer
'''
if opt.vis_2d_plt:
    visualizer_2D = visualizer_scene_2D(
        scene_obj, 
        modality_list_vis=[
            'im', 
            'depth', 
            'normal', 
            'mi_depth', 
            'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            'mi_seg_obj', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png

            # 'layout', 
            # 'shapes', 
            # 'albedo', 
            # 'roughness', 
            # 'emission', 
            # 'lighting_SG', # convert to lighting_envmap and vis: images/demo_lighting_SG_envmap_2D_plt.png
            # 'lighting_envmap', # renderer with mi/blender: images/demo_lighting_envmap_mitsubaScene_2D_plt.png
            # 'seg_area', 'seg_env', 'seg_obj', 
            # 'mi_seg_area', 'mi_seg_env', 
            ], 
        frame_idx_list=[0, 1, 2, 3, 4], 
        # frame_idx_list=[0], 
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
            'mi_normal_vis_coords': 'opencv', 
            'mi_depth_if_sync_scale': False, 
            }, 
    )


if opt.export:
    from lib.class_exporter import exporter_scene
    exporter = exporter_scene(
        scene_object=scene_obj,
        format=opt.export_format, 
        modality_list = [
            'poses', 
            'im_hdr', 
            'im_sdr', 
            'im_mask', 
            'shapes', 
            # 'mi_normal', 
            # 'mi_depth', 
            'normal', 
            'depth', 
            ], 
        if_force=opt.force, 
        
    )
    assert opt.export_format == 'monosdf'
    if opt.export_format == 'monosdf':
        exporter.export_monosdf_fvp_mitsuba(
            # split=opt.split, 
            format='monosdf',
            appendix='_REEXPORT', 
            )

'''
Open3D 3D viewer
'''
if opt.vis_3d_o3d:
    visualizer_3D_o3d = visualizer_scene_3D_o3d(
        scene_obj, 
        modality_list_vis=[
            # 'dense_geo', # fused from 2D
            'poses', 
            # 'lighting_SG', # images/demo_lighting_SG_o3d.png; arrows in blue
            # 'lighting_envmap', # images/demo_lighting_envmap_o3d.png; arrows in pink
            # 'layout', 
            'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters SHAPES)
            # 'emitters', # emitter PROPERTIES (e.g. SGs, half envmaps)
            'mi', # mitsuba sampled rays, pts
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
        if 'emitter_rays_list' in eval_return_dict:
            assert opt.eval_rad
            for _ in eval_return_dict['emitter_rays_list']:
                lpts, lpts_end = _['v'], _['v']+_['d']*_['l']
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
            assert opt.eval_rad or opt.eval_inv or opt.eval_scene
            visualizer_3D_o3d.extra_input_dict['samples_v_dict'] = eval_return_dict['samples_v_dict']
        
    visualizer_3D_o3d.run_o3d(
        if_shader=opt.if_shader, # set to False to disable faycny shaders 
        cam_params={
            'if_cam_axis_only': False, 
            }, 
        # dense_geo_params={
        #     'subsample_pcd_rate': 1, # change this according to how sparse the points you would like to be (also according to num of frame_id_list)
        #     'if_ceiling': False, # [OPTIONAL] remove ceiling points to better see the furniture 
        #     'if_walls': False, # [OPTIONAL] remove wall points to better see the furniture 
        #     'if_normal': False, # [OPTIONAL] turn off normals to avoid clusters
        #     'subsample_normal_rate_x': 2, 
        #     'pcd_color_mode': opt.pcd_color_mode_dense_geo, 
        #     }, 
        lighting_params=lighting_params_vis, 
        shapes_params={
            # 'simply_mesh_ratio_vis': 1., # simply num of triangles to #triangles * simply_mesh_ratio_vis
            'if_meshes': True, # [OPTIONAL] if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': False, # [OPTIONAL] if show labels (False: only show bboxes)
            'if_voxel_volume': False, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png
            'if_ceiling': True, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            'if_walls': True, # [OPTIONAL] remove wall meshes to better see the furniture 
            'if_sampled_pts': False, # [OPTIONAL] is show samples pts from scene_obj.sample_pts_list if available
            'mesh_color_type': 'eval-vis_count', # ['obj_color', 'face_normal', 'eval-rad', 'eval-emission_mask', 'eval-vis_count', 'eval-t']
            # 'mesh_color_type': 'eval-t', # ['obj_color', 'face_normal', 'eval-rad', 'eval-emission_mask', 'eval-vis_count]
        },
        emitter_params={
            # 'if_half_envmap': False, # [OPTIONAL] if show half envmap as a hemisphere for window emitters (False: only show bboxes)
            # 'scale_SG_length': 2., 
            'if_sampling_emitter': True, 
            'radiance_scale': radiance_scale, 
            'max_plate': 32, 
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

