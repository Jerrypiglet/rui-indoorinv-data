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
opt = parser.parse_args()

base_root = Path(PATH_HOME) / 'data/scannet'
xml_root = Path(PATH_HOME) / 'data/scannet'
# intrinsics_path = Path(PATH_HOME) / 'data/scannet/intrinsic_mitsubaScene.txt'

radiance_scale = 1.
'''
scannet scan 1~4 from MonoSDF
'''
scan_id = 1
# frame_id_list = list(range(465))
frame_id_list = list(range(0, 465, 40))
# frame_id_list = [0]
# shape_file = ('not-normalized', 'GTmesh/scene0050_00_vh_clean_2.ply')
shape_file = ('normalized', 'ESTmesh/scan1.ply')

mitsuba_scene = monosdfScene3D(
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root, 'xml_scene_root': xml_root}, 
    scene_params_dict={
        # 'xml_filename': xml_filename, 
        'scene_name': 'scan%d'%scan_id, 
        # 'split': split, 
        'frame_id_list': frame_id_list, 
        # 'mitsuba_version': '3.0.0', 
        # 'intrinsics_path': intrinsics_path, 
        'axis_up': 'z+', 
        'pose_file': ('npz', 'cameras.npz'), # requires scaled Blender scene! in comply with Liwen's IndoorDataset (https://github.com/william122742/inv-nerf/blob/bake/utils/dataset/indoor.py)
        'shape_file': shape_file, 
        }, 
    mi_params_dict={
        # 'if_also_dump_xml_with_lit_area_lights_only': True,  # True: to dump a second file containing lit-up lamps only
        # 'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        # 'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': True, # [depend on if_sample_rays_pts] True: to generate segs similar to those in openroomsScene2D.load_seg()
        # sample poses and render images 
        # 'if_sample_poses': False, # True to generate camera poses following Zhengqin's method (i.e. walking along walls)
        # 'poses_sample_num': 200, # Number of poses to sample; set to -1 if not sampling
        },
    # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
    modality_list = [
        'im_sdr', 
        'shapes', # single scene file in ScanNet's case; objs + emitters, geometry shapes + emitter properties
        'depth', 
        'normal', 

        # 'im_hdr', 
        # 'lighting_envmap', 
        # 'albedo', 'roughness', 
        # 'emission', 
        # 'lighting_SG', 
        # 'layout', 
        ], 
    modality_filename_dict = {
        # 'poses', 
        # 'im_hdr': 'Image/%03d_0001.exr', 
        'im_sdr': '%06d_rgb.png', 
        # 'lighting_envmap', 
        # 'albedo': 'DiffCol/%03d_0001.exr', 
        # 'roughness': 'Roughness/%03d_0001.exr', 
        # 'emission': 'Emit/%03d_0001.exr', 
        'depth': '%06d_depth.npy', 
        'normal': '%06d_normal.npy', 
        # 'lighting_SG', 
        # 'layout', 
        # 'shapes', # objs + emitters, geometry shapes + emitter properties
    }, 
    im_params_dict={
        # 'im_H_resize': 480, 'im_W_resize': 640, 
        'im_H_load': 384, 'im_W_load': 384, 
        'im_H_resize': 384, 'im_W_resize': 384, 
        # 'im_H_resize': 1, 'im_W_resize': 2, 
        # 'im_H_resize': 32, 'im_W_resize': 64, 
        # 'spp': 2048, 
        # 'spp': 16, 
        # 'im_H_resize': 120, 'im_W_resize': 160, # to use for rendering so that im dimensions == lighting dimensions
        # 'im_hdr_ext': 'exr', 
        }, 
    cam_params_dict={
        'center_crop_type': 'no_crop', 
        'near': 0.1, 'far': 10., 
        # 'heightMin' : 0.7,  
        # 'heightMax' : 2.,  
        # 'distMin': 1., # to wall distance min
        # 'distMax': 4.5, 
        # 'thetaMin': -60, 
        # 'thetaMax' : 40, # theta: pitch angle; up+
        # 'phiMin': -60, # yaw angle
        # 'phiMax': 60, 
        'sample_pose_if_vis_plt': False, # images/demo_sample_pose.png
    }, 
    # lighting_params_dict={
    #     'SG_num': 12, 
    #     'env_row': 8, 'env_col': 16, # resolution to load; FIXED
    #     'env_downsample_rate': 2, # (8, 16) -> (4, 8)

    #     # 'env_height': 2, 'env_width': 4, 
    #     # 'env_height': 8, 'env_width': 16, 
    #     # 'env_height': 128, 'env_width': 256, 
    #     'env_height': 256, 'env_width': 512, 
    # }, 
    shape_params_dict={
        'if_load_obj_mesh': True, # set to False to not load meshes for objs (furniture) to save time
        # 'if_load_emitter_mesh': True,  # default True: to load emitter meshes, because not too many emitters

        'if_sample_mesh': False,  # default True: sample points on each shape -> self.sample_pts_list
        'sample_mesh_ratio': 0.1, # target num of VERTICES: len(vertices) * sample_mesh_ratio
        'sample_mesh_min': 10, 
        'sample_mesh_max': 100, 

        'if_simplify_mesh': False,  # default True: simply triangles
        'simplify_mesh_ratio': 0.1, # target num of FACES: len(faces) * simplify_mesh_ratio
        'simplify_mesh_min': 100, 
        'simplify_mesh_max': 1000, 
        'if_remesh': True, # False: images/demo_shapes_3D_kitchen_NO_remesh.png; True: images/demo_shapes_3D_kitchen_YES_remesh.png
        'remesh_max_edge': 0.15,  
        },
    # emitter_params_dict={
    #     },
)

'''
Mitsuba/Blender 2D renderer
'''
if opt.render_2d:
    assert False
    assert opt.renderer in ['mi', 'blender']
    modality_list = [
        # 'im', # both hdr and sdr
        # 'poses', 
        # 'seg', 
        # 'albedo', 
        # 'roughness', 
        # 'depth', 'normal', 
        'lighting_envmap', 
        ]
    if opt.renderer == 'mi':
        renderer = renderer_mi_mitsubaScene_3D(
            mitsuba_scene, 
            modality_list=modality_list, 
            im_params_dict={}, 
            cam_params_dict={}, 
            mi_params_dict={},
        )
    if opt.renderer == 'blender':
        renderer = renderer_blender_mitsubaScene_3D(
            mitsuba_scene, 
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


eval_return_dict = {}
'''
Evaluator for rad-MLP
'''
if opt.eval_rad:
    assert False
    evaluator_rad = evaluator_scene_rad(
        host=host, 
        scene_object=mitsuba_scene, 
        INV_NERF_ROOT = INV_NERF_ROOT, 
        ckpt_path='20230110-132112-rad_kitchen_190-10_specT/last.ckpt', # 110
        dataset_key='-'.join(['Indoor', scene_name]), # has to be one of the keys from inv-nerf/configs/scene_options.py
        split=split, 
        rad_scale=1., 
    )

    '''
    render one image by querying rad-MLP: images/demo_eval_radMLP_render.png
    '''
    # evaluator_rad.render_im(7, if_plt=True) 

    '''
    sample and visualize points on emitter surface; show intensity as vectors along normals (BLUE for EST): images/demo_emitter_o3d_sampling.png
    '''
    eval_return_dict.update(
        evaluator_rad.sample_emitter(
            emitter_params={
                'max_plate': 32, 
                'radiance_scale': radiance_scale, 
                'emitter_type_index_list': emitter_type_index_list, 
                }))
    
    '''
    sample non-emitter locations along envmap (hemisphere) directions radiance from rad-MLP: images/demo_envmap_o3d_sampling.png
    '''
    # eval_return_dict.update(
    #     evaluator_rad.sample_lighting(
    #         # sample_type='rad', # 'rad', 'incident-rad'
    #         sample_type='incident', # 'rad', 'incident-rad'
    #         subsample_rate_pts=1, 
    #         if_use_loaded_envmap_position=True, # assuming lighting envmap endpoint position dumped by Blender renderer
    #     )
    # )

    '''
    sample radiance field on shape vertices
    '''
    eval_return_dict.update(
        evaluator_rad.sample_shapes(
            sample_type='rad', # ['rad']
            shape_params={
                'radiance_scale': 1., 
            }
        )
    )

'''
Evaluator for inv-MLP
'''
if opt.eval_inv:
    assert False
    evaluator_inv = evaluator_scene_inv(
        host=host, 
        scene_object=mitsuba_scene, 
        INV_NERF_ROOT = INV_NERF_ROOT, 
        ckpt_path='20230111-191305-inv_kitchen_190-10_specT/last.ckpt', # 110
        dataset_key='-'.join(['Indoor', scene_name]), # has to be one of the keys from inv-nerf/configs/scene_options.py
        split=split, 
        spec=True, 
    )

    '''
    sample emission mask on shape vertices
    '''
    _ = evaluator_inv.sample_shapes(
        sample_type='emission_mask', # ['']
        shape_params={
        }
    )
    for k, v in _.items():
        if k in eval_return_dict:
            eval_return_dict[k].update(_[k])
        else:
            eval_return_dict[k] = _[k]

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
                mitsuba_scene.add_modality(eval_return_dict[modality], modality, 'EST')

    visualizer_2D.vis_2d_with_plt(
        lighting_params={
            'lighting_scale': 1., # rescaling the brightness of the envmap
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
        mitsuba_scene, 
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
        
        # vertex normals

        # if 'samples_v_dict' in eval_return_dict:
        #     assert opt.eval_rad
        #     for _id in eval_return_dict['samples_v_dict']:
        #         lpts, lpts_d = eval_return_dict['samples_v_dict'][_id]['v'], eval_return_dict['samples_v_dict'][_id]['d']
        #         lpts_end = lpts + 0.1 * lpts_d
        #         visualizer_3D_o3d.add_extra_geometry([
        #             ('rays', {
        #                 'ray_o': lpts, 'ray_e': lpts_end, 'ray_c': np.array([[0., 1., 0.]]*lpts.shape[0]), # green
        #             }),
        #         ]) 

    # for frame_idx, (rays_o, rays_d, _) in enumerate(mitsuba_scene.cam_rays_list):
    #     normal_up = np.cross(rays_d[0][0], rays_d[0][-1])
    #     normal_down = np.cross(rays_d[-1][-1], rays_d[-1][0])
    #     normal_left = np.cross(rays_d[-1][0], rays_d[0][0])
    #     normal_right = np.cross(rays_d[0][-1], rays_d[-1][-1])
    #     normals = np.stack((normal_up, normal_down, normal_left, normal_right), axis=0)
    #     cam_o = rays_o[[0,0,-1,-1],[0,-1,0,-1]] # get cam_d of 4 corners: (4, 3)
    #     visualizer_3D_o3d.add_extra_geometry([
    #         ('rays', {
    #             'ray_o': cam_o[0:1], 'ray_e': cam_o[0:1] + normals[0:1], 'ray_c': np.array([[0., 0., 0.]]*1), # black
    #         }),
    #         ('rays', {
    #             'ray_o': cam_o[1:2], 'ray_e': cam_o[1:2] + normals[1:2], 'ray_c': np.array([[1., 0., 0.]]*1), # r
    #         }),
    #         ('rays', {
    #             'ray_o': cam_o[2:3], 'ray_e': cam_o[2:3] + normals[2:3], 'ray_c': np.array([[0., 1., 0.]]*1), # g
    #         }),
    #         ('rays', {
    #             'ray_o': cam_o[3:4], 'ray_e': cam_o[3:4] + normals[3:4], 'ray_c': np.array([[0., 0., 1.]]*1), # b
    #         }),
    #     ]) 


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
            'if_sampled_pts': False, # [OPTIONAL] is show samples pts from mitsuba_scene.sample_pts_list if available
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

