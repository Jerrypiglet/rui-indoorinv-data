'''
Works with
- Indoor Synthetic scenes;
- Evermotion scenes exported to BlendeMitsuba format (via Blender).

Examples:

Load and export indoor_synthetic scenes:

- Load and export indoor_synthetic scenes train and val to MonoSDF format:

> python load_mitsubaScene3D.py --scene kitchen_mi --export --export_format monosdf --export_appendix _fipt --split train --vis_3d_o3d False --force           

- Load Evermotion scenes:

> python load_mitsubaScene3D.py --DATASET Evermotion --scene AI55_004

'''

import sys
from pathlib import Path

host = 'r4090'
# host = 'apple'

from lib.global_vars import PATH_HOME_dict# , INV_NERF_ROOT_dict, MONOSDF_ROOT_dict, OR_RAW_ROOT_dict
PATH_HOME = Path(PATH_HOME_dict[host])
sys.path.insert(0, str(PATH_HOME))

import numpy as np
np.set_printoptions(suppress=True)
import argparse
from pyhocon import ConfigFactory, ConfigTree
from lib.utils_misc import str2bool, white_magenta, check_exists

from lib.class_mitsubaScene3D import mitsubaScene3D

from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d

# from lib.class_eval_rad import evaluator_scene_rad
# from lib.class_eval_monosdf import evaluator_scene_monosdf
# from lib.class_eval_inv import evaluator_scene_inv
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

# differential renderer
# parser.add_argument('--render_diff', type=str2bool, nargs='?', const=True, default=False, help='differentiable surface rendering')
# parser.add_argument('--renderer_option', type=str, default='PhySG', help='differentiable renderer option')

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
parser.add_argument('--scene', type=str, default='kitchen', help='load conf file: confs/indoor_synthetic/\{opt.scene\}.conf')
parser.add_argument('--DATASET', type=str, default='indoor_synthetic', help='load conf file: confs/\{DATASET\}')

opt = parser.parse_args()

conf_base_path = Path('confs/%s.conf'%opt.DATASET); check_exists(conf_base_path)
CONF = ConfigFactory.parse_file(str(conf_base_path))
conf_scene_path = Path('confs/%s/%s.conf'%(opt.DATASET, opt.scene)); check_exists(conf_scene_path)
conf_scene = ConfigFactory.parse_file(str(conf_scene_path))
CONF = ConfigTree.merge_configs(CONF, conf_scene)

dataset_root = Path(PATH_HOME) / CONF.data.dataset_root
xml_root = Path(PATH_HOME) / CONF.data.xml_root

frame_id_list = CONF.scene_params_dict.frame_id_list
invalid_frame_id_list = CONF.scene_params_dict.invalid_frame_id_list

# [debug] override
# frame_id_list = [0,1,2,3,4,5]

'''
update confs
'''

CONF.scene_params_dict.update({
    'split': opt.split, # train, val, train+val
    'frame_id_list': frame_id_list, 
    # 'extra_transform': np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32), # z=y, y=x, x=z # convert from y+ (native to indoor synthetic) to z+
    'invalid_frame_id_list': invalid_frame_id_list, 
    })
    
CONF.cam_params_dict.update({
    # ==> if sample poses and render images 
    'if_sample_poses': opt.if_sample_poses, # True to generate camera poses following Zhengqin's method (i.e. walking along walls)
    'sample_pose_num': 200 if 'train' in opt.split else 20, # Number of poses to sample; set to -1 if not sampling
    'sample_pose_if_vis_plt': True, # images/demo_sample_pose.png, images/demo_sample_pose_bathroom.png
    })

CONF.mi_params_dict.update({
    'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
    'if_get_segs': True, # [depend on if_sample_rays_pts=True] True: to generate segs similar to those in openroomsScene2D.load_seg()
    })

CONF.im_params_dict.update({
    'im_H_resize': 480, 'im_W_resize': 640, 
    # 'im_H_resize': 640, 'im_W_resize': 1280, 
    })

CONF.shape_params_dict.update({
    'if_load_obj_mesh': True, # set to False to not load meshes for objs (furniture) to save time
    'if_load_emitter_mesh': True, # default True: to load emitter meshes, because not too many emitters
    })

'''
create scene obj
'''
scene_obj = mitsubaScene3D(
    CONF = CONF, 
    if_debug_info = opt.if_debug_info, 
    host = host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'dataset_root': dataset_root, 'xml_root': xml_root}, 
    modality_list = [
        'shapes', # objs + emitters, geometry shapes + emitter properties``
        # 'layout', 
        'poses', 
        # 'im_hdr', 
        # 'im_sdr', 
        # 'lighting_envmap', 
        # 'albedo', 'roughness', 
        # 'emission', 
        # 'depth', 
        # 'normal', 
        # 'tsdf', 
        ], 
)

'''
Mitsuba/Blender 2D renderer
'''
if opt.render_2d:
    assert opt.renderer in ['mi', 'blender']
    if opt.renderer == 'mi':
        renderer = renderer_mi_mitsubaScene_3D(
            scene_obj, 
            modality_list=[
                'im', # both hdr and sdr
            ], 
            im_params_dict=
            {
                # 'im_H_load': 640, 'im_W_load': 1280, 
                'im_H_load': 480, 'im_W_load': 640, 
                'spp': 4096, 
            }, # override
            cam_params_dict={}, 
            mi_params_dict={},
            if_skip_check=True,
        )
    if opt.renderer == 'blender':
        renderer = renderer_blender_mitsubaScene_3D(
            scene_obj, 
            modality_list=[
                # 'im', 
                'albedo', 
                'roughness', 
                'depth', 
                'normal', 
                'index', 
                'emission', 
                # 'lighting_envmap', 
                ], 
            host=host, 
            FORMAT='OPEN_EXR', 
            # FORMAT='PNG', 
            im_params_dict=
            {
                # 'im_H_load': 640, 'im_W_load': 1280, 
                'im_H_load': 480, 'im_W_load': 640, 
                # 'spp': 32, 
                'spp': 256, 
            }, # override
            cam_params_dict={}, 
            mi_params_dict={},
            # blender_file_name='test_blender_export_reimport.blend', 
            if_skip_check=True,
        )
    host=host, 
    renderer.render(if_force=opt.force)
    
    # compare HDR images from mi/blender if both are available
    # renderer.compare_blender_mi_Image()

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
        if_force=opt.force, 
        # convert from y+ (native to indoor synthetic) to z+
        # extra_transform = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),  # y=z, z=x, x=y
        # extra_transform = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32),  # z=y, y=x, x=z
        
    )
    if opt.export_format == 'monosdf':
        exporter.export_monosdf_fvp_mitsuba(
            split=opt.split, 
            if_mask_from_mi=True, 
            format='monosdf',
            modality_list = [
                # 'poses', 
                # 'im_hdr', 
                # 'im_sdr', 
                # 'im_mask', 
                # 'shapes', 
                'mi_normal', 
                'mi_depth', 
                ], 
            appendix=opt.export_appendix, 
            )
    if opt.export_format == 'fvp':
        exporter.export_monosdf_fvp_mitsuba(
            split=opt.split, 
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
            # 'layout', 
            # 'shapes', 
            # 'albedo', 
            # 'roughness', 
            # 'emission', 
            # 'depth', 
            # 'normal', 
            'mi_depth', 
            'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            # 'lighting_envmap', # renderer with mi/blender: images/demo_lighting_envmap_mitsubaScene_2D_plt.png
            # 'seg_area', 'seg_env', 'seg_obj', 
            # 'mi_seg_area', 'mi_seg_env', 'mi_seg_obj', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
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
            'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters SHAPES); CTRL + 9
            'layout', 
            'mi', # mitsuba sampled rays, pts
            'poses', 
            # 'tsdf', 
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
            'if_labels': True, # [OPTIONAL] if show labels (False: only show bboxes)
            'if_voxel_volume': False, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png; USEFUL WHEN NEED TO CHECK SCENE SCALE (1 voxel = 1 meter)

            # 'if_ceiling': True if opt.eval_scene else False, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            # 'if_walls': True if opt.eval_scene else False, # [OPTIONAL] remove wall meshes to better see the furniture 
            'if_ceiling': True, 
            'if_walls': False, 
            'if_floor': True, 

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

