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
from lib.utils_misc import str2bool, white_magenta
import argparse

from lib.class_simpleScene3D import simpleScene3D

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

base_root = Path(PATH_HOME) / 'data/real/EXPORT_mitsuba'
xml_root = Path(PATH_HOME) / 'data/real/EXPORT_mitsuba'

'''
default
'''
eval_models_dict = {}
monosdf_shape_dict = {}
shape_file = ''
frame_ids = []
invalid_frame_id_list = []
hdr_radiance_scale = 1.

scene_name = 'IndoorKitchen_v2_2'; hdr_radiance_scale = 2.
if_rc = False; pcd_file = ''; pose_file = ('OpenRooms', 'cam.txt')
shape_file = base_root / scene_name / 'scene.obj'

scene_obj = simpleScene3D(
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root}, 
    scene_params_dict={
        'scene_name': scene_name, 
        'frame_id_list': frame_ids, 
        'axis_up': 'y+', # WILL REORIENT TO y+
        'invalid_frame_id_list': invalid_frame_id_list, 
        'pose_file': pose_file, 
        'shape_file': shape_file, 
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
        'im_hdr': 'Image/%03d_0001.exr', 
        'im_sdr': 'Image/%03d_0001.png', 
    }, 
    im_params_dict={
        'hdr_radiance_scale': hdr_radiance_scale, 
        
        # V2_2
        'im_H_load_hdr': 512, 'im_W_load_hdr': 768, 
        'im_H_load_sdr': 512, 'im_W_load_sdr': 768, 
        'im_H_load': 512, 'im_W_load': 768, 
        # 'im_H_resize': 360, 'im_W_resize': 540, # inv-nerf
        'im_H_resize': 512, 'im_W_resize': 768, # monosdf
        }, 
    cam_params_dict={
        'near': 0.1, 'far': 1., # [in a unit box]
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
            ]
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
            ]
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
                ], 
            appendix=opt.export_appendix, 
        )
    if opt.export_format == 'lieccv22':
        exporter.export_lieccv22(
            modality_list = [
            'im_sdr', 
            'mi_depth', 
            'mi_seg', 
            ], 
            # split=opt.split, 
            assert_shape=(240, 320),
            window_area_emitter_id_list=['window_area_emitter'], # need to manually specify in XML: e.g. <emitter type="area" id="lamp_oven_0">
            merge_lamp_id_list=['lamp_oven_0', 'lamp_oven_1', 'lamp_oven_2'],  # need to manually specify in XML
            # center_crop_HW=(240, 320), 
            if_no_gt_appendix=True, 
        )
    
'''
Matploblib 2D viewer
'''
if opt.vis_2d_plt:
    visualizer_2D = visualizer_scene_2D(
        scene_obj, 
        modality_list_vis=[
            'im', 
            'mi_depth' if shape_file != '' else '', 
            'mi_normal' if shape_file != '' else '', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
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
            'cam_vis_scale': 0.3, 
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

            'if_cam_rays': True, 
            'cam_rays_if_pts': shape_file != '', # if cam rays end in surface intersections; set to False to visualize rays of unit length
            'cam_rays_subsample': 10, 
            
            'if_normal': False, 
            'normal_subsample': 50, 
            'normal_scale': 0.2, 

        }, 
    )

