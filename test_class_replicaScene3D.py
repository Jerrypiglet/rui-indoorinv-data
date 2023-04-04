'''
work with Mitsuba/Blender scenes
'''
import sys
from lib.class_renderer_blender_mitsubaScene_3D import renderer_blender_mitsubaScene_3D
from lib.class_renderer_mi_mitsubaScene_3D import renderer_mi_mitsubaScene_3D

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

from lib.class_replicaScene3D import replicaScene3D

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
parser.add_argument('--f', type=str2bool, nargs='?', const=True, default=False, help='whether to show (1) pixel-space modalities (2) projection onto one image (e.g. layout, object bboxes), with plt')
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
parser.add_argument('--if_sample_poses', type=str2bool, nargs='?', const=True, default=False, help='if sample camera poses instead of loading from pose file')
parser.add_argument('--if_dump_shape', type=str2bool, nargs='?', const=True, default=False, help='if dump shape of entire scene')
parser.add_argument('--if_export', type=str2bool, nargs='?', const=True, default=False, help='if export entire scene to mitsubaScene data structure')

opt = parser.parse_args()

dataset_root = Path(PATH_HOME) / 'data/replica_v1'
assert dataset_root.exists()
sdr_radiance_scale = 1.

# scene_name = 'apartment_0'
# scene_name = 'apartment_1'
# scene_name = 'apartment_2'
# scene_name = 'frl_apartment_0'
# scene_name = 'frl_apartment_1'
# scene_name = 'frl_apartment_2'
# scene_name = 'frl_apartment_3'
# scene_name = 'frl_apartment_4'
# scene_name = 'frl_apartment_5'
# scene_name = 'hotel_0'
# scene_name = 'office_0'; sdr_radiance_scale = 1.; 
# scene_name = 'office_1'; sdr_radiance_scale = 1.; 
# scene_name = 'office_2'; sdr_radiance_scale = 0.5; 
# scene_name = 'office_3'
# scene_name = 'office_4'
# scene_name = 'room_0'; sdr_radiance_scale = 0.5; 
scene_name = 'room_1'; sdr_radiance_scale = 0.5; 
# scene_name = 'room_2'

# frame_ids = [23]
# frame_ids = [0, 10, 199]
frame_ids = list(range(200))

scene_obj = replicaScene3D(
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'dataset_root': dataset_root}, 
    scene_params_dict={
        'scene_name': scene_name, 
        'frame_id_list': frame_ids, 
        'intrinsics_path': dataset_root / 'intrinsic_mitsubaScene.txt', 

        # convert from z+ (native) to y+ and display in y+
        'axis_up': 'y+', 
        'extra_transform': np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),  # y=z, z=x, x=y
        'pose_file': ('OpenRooms', 'cam.txt', False), # only useful after dumping poses to cam.txt

        # display in z+ (native); load pose from z+
        # 'axis_up': 'z+', 
        # 'pose_file': ('OpenRooms', '/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader/data/replica_v1/scene_export/room_0/cam_extra_transform.txt', True), # only useful after dumping poses to cam.txt
        }, 
    mi_params_dict={
        'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': True, # [depend on if_sample_rays_pts] True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
    modality_list = [
        'poses' if not opt.if_sample_poses else '', 
        'shapes', 
        
        # after rendering
        'im_hdr' if not opt.if_sample_poses else '', 
        'im_sdr' if not opt.if_sample_poses else '', 

        # 'mi_normal', 
        # 'mi_depth', 
        ], 
    modality_filename_dict = {
        'im_hdr': 'frame%06d.exr', 
        'im_sdr': 'sdr%06d.png', 
        'depth': 'depth%06d.jpg', 
        # 'im_mask': 'mask%06d.png', 
    }, 
    im_params_dict={
        'im_H_load': 320, 'im_W_load': 640, 
        'im_H_resize': 320, 'im_W_resize': 640, 
        'sdr_radiance_scale': sdr_radiance_scale, 
        }, 
    cam_params_dict={
        'near': 0.1, 'far': 10., 
        # == params for sample camera poses
        'sampleNum': 3, 
        
        'heightMin' : 0.4, # camera height min
        'heightMax' : 2., # camera height max
        'distMin': 0.2, # to wall distance min
        'distMax': 3., # to wall distance max
        'thetaMin': -60, # theta min: pitch angle; up+ 
        'thetaMax' : 40, # theta max: pitch angle; up+
        'phiMin': -60, # yaw angle min
        'phiMax': 60, # yaw angle max
        'distRaysMin': 0.3, # min dist of all camera rays to the scene; should be relatively relaxed; [!!!] set to -1 to disable checking
        'distRaysMedianMin': 0.6, # median dist of all camera rays to the scene; should be relatively STRICT to avoid e.g. camera too close to walls; [!!!] set to -1 to disable checking

        'heightMin' : 0.7, # camera height min
        'heightMax' : 2., # camera height max
        'distMin': 0.1, # to wall distance min
        'distMax': 2.5, # to wall distance max
        'thetaMin': -60, # theta min: pitch angle; up+ 
        'thetaMax' : 40, # theta max: pitch angle; up+
        'phiMin': -60, # yaw angle min
        'phiMax': 60, # yaw angle max
        'distRaysMin': -1, # min dist of all camera rays to the scene; [!!!] set to -1 to disable checking
        'distRaysMedianMin': 0.2, # median dist of all camera rays to the scene; [!!!] set to -1 to disable checking

        # ==> if sample poses and render images 
        'if_sample_poses': opt.if_sample_poses, # True to generate camera poses following Zhengqin's method (i.e. walking along walls)
        'sample_pose_num': 200, # Number of poses to sample; set to -1 if not sampling
        'sample_pose_if_vis_plt': True, # images/demo_sample_pose.png, images/demo_sample_pose_bathroom.png
    }, 
    shape_params_dict={
        'if_dump_shape': opt.if_dump_shape, # True to dump fixed shape to obj file
        'if_fix_watertight': not opt.if_export, 
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
Mitsuba/Blender 2D renderer
'''
if opt.render_2d:
    assert opt.renderer in ['mi', 'blender']
    modality_list = [
        # 'im', # both hdr and sdr
        # 'poses', 
        # 'seg', 
        # 'albedo', 
        # 'roughness', 
        'depth', 'normal', 
        # 'lighting_envmap', 
        ]
    if opt.renderer == 'mi':
        renderer = renderer_mi_mitsubaScene_3D(
            scene_obj, 
            modality_list=modality_list, 
            im_params_dict={}, 
            cam_params_dict={}, 
            mi_params_dict={},
        )
    if opt.renderer == 'blender':
        renderer = renderer_blender_mitsubaScene_3D(
            scene_obj, 
            modality_list=modality_list, 
            host=host, 
            FORMAT='OPEN_EXR', 
            # FORMAT='PNG', 
            im_params_dict={}, 
            cam_params_dict={}, 
            mi_params_dict={},
        )
    host=host, 
    renderer.render()

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
            # 'im', 
            # 'im_mask', 
            'mi_depth', 
            'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            'mi_seg_env', 'mi_seg_obj', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
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
            'poses' if not opt.eval_scene else '', 
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
            'if_voxel_volume': False, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png; USEFUL WHEN NEED TO CHECK SCENE SCALE (1 voxel = 1 meter)
            # 'if_voxel_volume': True, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png; USEFUL WHEN NEED TO CHECK SCENE SCALE (1 voxel = 1 meter)
            # 'if_ceiling': True, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            'if_ceiling': False if not opt.eval_scene else True, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            'if_walls': False if not opt.eval_scene else True, # [OPTIONAL] remove walls meshes to better see the furniture 
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

