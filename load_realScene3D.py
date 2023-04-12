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
import argparse
from pyhocon import ConfigFactory, ConfigTree
from lib.utils_misc import str2bool, white_magenta, check_exists

from lib.class_realScene3D import realScene3D

from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d
from lib.class_visualizer_scene_3D_plt import visualizer_scene_3D_plt

# from lib.class_eval_rad import evaluator_scene_rad
# from lib.class_eval_monosdf import evaluator_scene_monosdf
# from lib.class_eval_inv import evaluator_scene_inv
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

# === after refactorization
parser.add_argument('--scene', type=str, default='ConferenceRoomV2_final_supergloo', help='load conf file: confs/real/\{opt.scene\}.conf')

opt = parser.parse_args()

DATASET = 'real'
conf_base_path = Path('confs/%s.conf'%DATASET); check_exists(conf_base_path)
CONF = ConfigFactory.parse_file(str(conf_base_path))
conf_scene_path = Path('confs/%s/%s.conf'%(DATASET, opt.scene)); check_exists(conf_scene_path)
conf_scene = ConfigFactory.parse_file(str(conf_scene_path))
CONF = ConfigTree.merge_configs(CONF, conf_scene)

dataset_root = Path(PATH_HOME) / CONF.data.dataset_root

'''
default
'''
frame_id_list = CONF.scene_params_dict.frame_id_list
invalid_frame_id_list = CONF.scene_params_dict.invalid_frame_id_list
invalid_frame_idx_list = CONF.scene_params_dict.invalid_frame_idx_list

'''
------ final
'''
# +++++ ConferenceRoomV2_final_supergloo_aligned +++++ [SUPP]
# scene_name = 'ConferenceRoomV2_final_supergloo'

# [debug] override
# frame_id_list = [0]

'''
modify confs
'''

if opt.export:
    if opt.export_format == 'mitsuba':
        CONF.im_params_dict.update({
            'im_H_resize': 360, 'im_W_resize': 540, # inv-nerf
        })
    elif opt.export_format == 'lieccv22':
        CONF.im_params_dict.update({
            'im_H_resize': 240, 'im_W_resize': 320, 
        })
    elif opt.export_format == 'monosdf':
        CONF.im_params_dict.update({
            'im_H_resize': 512, 'im_W_resize': 768, # monosdf
        })
    
CONF.scene_params_dict.update({
    'split': opt.split, # train, val, train+val
    'frame_id_list': frame_id_list, 
    # 'extra_transform': np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32), # z=y, y=x, x=z # convert from y+ (native to indoor synthetic) to z+
    'invalid_frame_id_list': invalid_frame_id_list, 
    })

CONF.mi_params_dict.update({
    'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
    'if_get_segs': False, # [depend on if_sample_rays_pts=True] True: to generate segs similar to those in openroomsScene2D.load_seg()
    })

CONF.shape_params_dict.update({
    'tsdf_path': 'fused_tsdf.ply', # 'test_files/tmp_tsdf.ply', 
    })

scene_obj = realScene3D(
    CONF = CONF, 
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'dataset_root': dataset_root}, 
    modality_list = [
        'poses', 
        'im_hdr', 
        'im_sdr', 
        'shapes', # objs + emitters, geometry shapes + emitter properties
        'tsdf', 
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
                'mi_normal', 
                'mi_depth', 
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
                'mi_normal', 
                'mi_depth', 
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
            'mi_depth', 
            'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            # 'mi_seg_area', 
            # 'mi_seg_env', 
            # 'mi_seg_obj', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
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
            # 'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters SHAPES); CTRL + 9
            'mi', # mitsuba sampled rays, pts
            'tsdf', 
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
            # 'if_ceiling': False if opt.eval_scene else False, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            # 'if_walls': False if opt.eval_scene else False, # [OPTIONAL] remove wall meshes to better see the furniture 
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
            'cam_rays_if_pts': True, # if cam rays end in surface intersections; set to False to visualize rays of unit length
            'cam_rays_subsample': 10, 
            
            'if_normal': False, 
            'normal_subsample': 50, 
            'normal_scale': 0.2, 

        }, 
    )

