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
from lib.utils_misc import str2bool
import argparse

from lib.class_matterportScene3D import matterportScene3D

from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d
# from lib.class_visualizer_scene_3D_plt import visualizer_scene_3D_plt

# from lib.class_eval_rad import evaluator_scene_rad
# from lib.class_eval_monosdf import evaluator_scene_monosdf
# from lib.class_eval_inv import evaluator_scene_inv
from lib.class_eval_scene import evaluator_scene_scene

# from lib.class_renderer_mi_mitsubaScene_3D import renderer_mi_mitsubaScene_3D
# from lib.class_renderer_blender_mitsubaScene_3D import renderer_blender_mitsubaScene_3D

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
parser.add_argument('--if_convert_poses', type=str2bool, nargs='?', const=True, default=False, help='if sample camera poses instead of loading from pose file')
parser.add_argument('--if_dump_shape', type=str2bool, nargs='?', const=True, default=False, help='if dump shape of entire scene')
parser.add_argument('--if_export', type=str2bool, nargs='?', const=True, default=False, help='if export entire scene to mitsubaScene data structure')

opt = parser.parse_args()

base_root = Path(PATH_HOME) / 'data/Matterport3D'
assert base_root.exists()

'''
conference room with set of lamps and white chairs
https://aspis.cmpt.sfu.ca/scene-toolkit/scans/simple-viewer?condition=mpr3d&modelId=mpr3d.17DRP5sb8fy_5
'''
scene_name = '17DRP5sb8fy'; region_id_list = [5]; hdr_radiance_scale = 10; 
frame_ids = [21, 22, 46, 47]

'''
old bedroom
https://aspis.cmpt.sfu.ca/scene-toolkit/scans/simple-viewer?condition=mpr3d&modelId=mpr3d.2t7WUuJeko7_5
'''
scene_name = '2t7WUuJeko7'; region_id_list = [4, 5]; hdr_radiance_scale = 10; 
frame_ids = [18, 19, 20, 21, 22]

scene_obj = matterportScene3D(
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root}, 
    scene_params_dict={
        'scene_name': scene_name, 
        # 'frame_id_list': frame_ids, # comment out to use all frames
        'axis_up': 'z+', 
        'region_id_list': region_id_list, 
        'if_undist': False, # True to use undistorted images/poses
        # 'pose_file': ('bundle', 'bundle.out'), 
        # 'pose_file': ('OpenRooms', 'cam.txt'), # after dump to cam.txt
        # 'if_scale_scene': True, # whether to scale the scene to metric in meters, with given scale in scale.txt
        }, 
    mi_params_dict={
        'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': True, # [depend on if_sample_rays_pts] True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
    modality_list = [
        'poses', 
        'im_hdr', 
        'im_sdr', 
        # 'depth', 
        # 'im_mask', 
        'shapes', 
        ], 
    modality_filename_dict = {
        'im_hdr': ('matterport_hdr_images', 'j', 'jxr'), 
        'im_sdr': ('matterport_color_images', 'i', 'jpg'), 
        'depth': ('matterport_depth_images', 'd', 'png'), 
        'poses': ('matterport_camera_poses', 'pose_', 'txt'), # https://github.com/niessner/Matterport/blob/master/data_organization.md#matterport_camera_poses
        'im_sdr_undist': ('undistorted_color_images', 'i', 'jpg'), 
        'depth_undist': ('undistorted_depth_images', 'd', 'png'), 
        # 'im_hdr_undist': ('undistorted_hdr_images', 'j', 'jxr'), 
        # 'normal_undist': ('undistorted_normal_images', 'd', 'png'), 
        # 'im_mask': 'images/%08d_mask.png', 

    }, 
    im_params_dict={
        'im_H_load': 1024, 'im_W_load': 1280, 
        'im_H_resize': 512, 'im_W_resize': 640, 
        'hdr_radiance_scale': hdr_radiance_scale, 
        }, 
    cam_params_dict={
        'if_convert': opt.if_convert_poses, # True to convert poses to cam.txt and K_list.txt
    }, 
    shape_params_dict={
        # 'if_load_obj_mesh': True, # set to False to not load meshes for objs (furniture) to save time
        # 'if_load_emitter_mesh': True,  # default True: to load emitter meshes, because not too many emitters
        'if_dump_shape': opt.if_dump_shape, # True to dump fixed shape to obj file

        # 'if_sample_pts_on_mesh': False,  # default True: sample points on each shape -> self.sample_pts_list
        # 'sample_mesh_ratio': 0.1, # target num of VERTICES: len(vertices) * sample_mesh_ratio
        # 'sample_mesh_min': 10, 
        # 'sample_mesh_max': 100, 

        # 'if_simplify_mesh': False,  # default True: simply triangles
        # 'simplify_mesh_ratio': 0.1, # target num of FACES: len(faces) * simplify_mesh_ratio
        # 'simplify_mesh_min': 100, 
        # 'simplify_mesh_max': 1000, 
        # 'if_remesh': True, # False: images/demo_shapes_3D_kitchen_NO_remesh.png; True: images/demo_shapes_3D_kitchen_YES_remesh.png
        # 'remesh_max_edge': 0.15,  
        },
    emitter_params_dict={
        },
)

if opt.if_export:
    scene_obj.export_scene(
        modality_list = [
        'poses', 
        'im_hdr', 
        'im_sdr', 
        'im_mask', 
        'shapes', 
        'mi_normal', 
        'mi_depth', 
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
        scene_obj, 
        modality_list_vis=[
            'im', 
            # 'im_mask', 
            'mi_depth', 
            'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            'mi_seg_area', 'mi_seg_env', 'mi_seg_obj', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
            ], 
        # frame_idx_list=[0], 
    )
    visualizer_2D.vis_2d_with_plt(
        lighting_params={
            }, 
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
        scene_obj, 
        modality_list_vis=[
            # 'dense_geo', # fused from 2D
            'poses', 
            'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters SHAPES)
            # 'mi', # mitsuba sampled rays, pts
            ], 
        if_debug_info=opt.if_debug_info, 
    )

    if opt.if_add_color_from_eval:
        if 'samples_v_dict' in eval_return_dict:
            assert opt.eval_rad or opt.eval_inv or opt.eval_scene or opt.eval_monosdf
            visualizer_3D_o3d.extra_input_dict['samples_v_dict'] = eval_return_dict['samples_v_dict']
        
    visualizer_3D_o3d.run_o3d(
        if_shader=opt.if_shader, # set to False to disable faycny shaders 
        cam_params={
            'if_cam_axis_only': False, 
            }, 
        lighting_params={}, 
        shapes_params={
            'if_meshes': True, # [OPTIONAL] if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': False, # [OPTIONAL] if show labels (False: only show bboxes)
            'if_voxel_volume': True, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png; USEFUL WHEN NEED TO CHECK SCENE SCALE (1 voxel = 1 meter)
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

