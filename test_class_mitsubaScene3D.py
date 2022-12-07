'''
work with Mitsuba/Blender scenes
'''
import sys

# host = 'mm1'
host = 'apple'
PATH_HOME = {
    'apple': '/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader', 
    'mm1': '', 
    'qc': '', 
}[host]
sys.path.insert(0, PATH_HOME)
INV_NERF_ROOT = {
    'apple': '/Users/jerrypiglet/Documents/Projects/inv-nerf', 
    'mm1': '/home/ruizhu/Documents/Projects/inv-nerf', 
    'qc': '', 
}[host]
from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)

from lib.class_mitsubaScene3D import mitsubaScene3D
from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d
from lib.class_eval_rad import evaluator_scene_rad
from lib.class_renderer_mi_mitsubaScene_3D import renderer_mi_mitsubaScene_3D
from lib.class_renderer_blender_mitsubaScene_3D import renderer_blender_mitsubaScene_3D

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
parser.add_argument('--if_add_rays_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if add rays from evaluating MLPs')
# debug
parser.add_argument('--if_debug_info', type=str2bool, nargs='?', const=True, default=False, help='if show debug info')
opt = parser.parse_args()

base_root = Path(PATH_HOME) / 'data/indoor_synthetic'
xml_root = Path(PATH_HOME) / 'data/indoor_synthetic'
intrinsics_path = Path(PATH_HOME) / 'data/indoor_synthetic/intrinsic_mitsubaScene.txt'

'''
The kitchen scene: data/indoor_synthetic/kitchen/scene_v3.xml
'''
xml_filename = 'scene_v3.xml'
scene_name = 'kitchen'
emitter_type_index_list = [('lamp', 0)]; radiance_scale = 0.1; 
# split = 'train'; frame_ids = list(range(0, 189, 40))
split = 'train'; frame_ids = list(range(0, 4, 1))
# split = 'train'; frame_ids = [0]

mitsuba_scene = mitsubaScene3D(
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root, 'xml_scene_root': xml_root}, 
    scene_params_dict={
        'xml_filename': xml_filename, 
        'scene_name': scene_name, 
        'split': split, 
        'frame_id_list': frame_ids, 
        'mitsuba_version': '3.0.0', 
        'intrinsics_path': intrinsics_path, 
        'up_axis': 'y+', 
        # 'pose_file': ('Blender', 'train.npy'), # requires scaled Blender scene!
        # 'pose_file': ('OpenRooms', 'cam.txt'), 
        'pose_file': ('json', 'transforms.json'), # requires scaled Blender scene! in comply with Liwen's IndoorDataset (https://github.com/william122742/inv-nerf/blob/bake/utils/dataset/indoor.py)
        }, 
    mi_params_dict={
        'if_also_dump_xml_with_lit_area_lights_only': True,  # True: to dump a second file containing lit-up lamps only
        'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': True, # [depend on if_sample_rays_pts] True: to generate segs similar to those in openroomsScene2D.load_seg()
        # sample poses and render images 
        # 'if_sample_poses': False, # True to generate camera poses following Zhengqin's method (i.e. walking along walls)
        'poses_sample_num': 200, # Number of poses to sample; set to -1 if not sampling
        },
    # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
    modality_list = [
        # 'im_hdr', 
        # 'im_sdr', 
        'poses', 
        # 'seg', 
        # 'albedo', 'roughness', 
        # 'depth', 'normal', 
        # 'lighting_SG', 
        # 'lighting_envmap', 
        # 'layout', 
        # 'shapes', # objs + emitters, geometry shapes + emitter properties
        ], 
    im_params_dict={
        # 'im_H_resize': 480, 'im_W_resize': 640, 
        'im_H_load': 320, 'im_W_load': 640, 
        # 'im_H_resize': 160, 'im_W_resize': 320, 
        'im_H_resize': 1, 'im_W_resize': 2, 
        # 'im_H_resize': 32, 'im_W_resize': 64, 
        # 'spp': 2048, 
        'spp': 16, 
        # 'im_H_resize': 120, 'im_W_resize': 160, # to use for rendering so that im dimensions == lighting dimensions
        }, 
    cam_params_dict={
        'near': 0.1, 'far': 10., 
        'heightMin' : 0.7,  
        'heightMax' : 2.,  
        'distMin': 1., # to wall distance min
        'distMax': 4.5, 
        'thetaMin': -60, 
        'thetaMax' : 40, # theta: pitch angle; up+
        'phiMin': -60, # yaw angle
        'phiMax': 60, 
        'if_vis_plt': False, # images/demo_sample_pose.png
    }, 
    lighting_params_dict={
        'env_row': 120, 'env_col': 160, 'SG_num': 12, 
        # 'env_height': 2, 'env_width': 4, 
        'env_height': 8, 'env_width': 16, 
        # 'env_height': 64, 'env_width': 128, 
    }, 
    shape_params_dict={
        'if_load_obj_mesh': True, # set to False to not load meshes for objs (furniture) to save time
        'if_load_emitter_mesh': True,  # default True: to load emitter meshes, because not too many emitters
        },
    emitter_params_dict={
        },
)

'''
Mitsuba/Blender 2D renderer
'''
if opt.render_2d:
    assert opt.renderer in ['mi', 'blender']
    modality_list = [
        'im', # both hdr and sdr
        # 'poses', 
        # 'seg', 
        # 'albedo', 
        # 'roughness', 
        # 'depth', 'normal', 
        # 'lighting_envmap', 
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

'''
Matploblib 2D viewer
'''
if opt.vis_2d_plt:
    visualizer_2D = visualizer_scene_2D(
        mitsuba_scene, 
        modality_list_vis=[
            'im', 
            # 'layout', 
            # 'shapes', 
            # 'depth', 
            'mi_depth', 
            # 'normal', 
            'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            # 'lighting_SG', # convert to lighting_envmap and vis: images/demo_lighting_SG_envmap_2D_plt.png
            # 'lighting_envmap', 
            # 'seg_area', 'seg_env', 'seg_obj', 
            'mi_seg_area', 'mi_seg_env', 'mi_seg_obj', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
            ], 
        frame_idx_list=[0, 1, 2, 3, 4], 
        # frame_idx_list=[0], 
    )
    visualizer_2D.vis_2d_with_plt(
        lighting_params={
            'lighting_scale': 0.01, # rescaling the brightness of the envmap
            }, 
            )

'''
Evaluator for rad-MLP and inv-MLP
'''
eval_return_dict = {}
if opt.eval_rad:
    evaluator_rad = evaluator_scene_rad(
        host=host, 
        scene_object=mitsuba_scene, 
        INV_NERF_ROOT = INV_NERF_ROOT, 
        ckpt_path='kitchen/last.ckpt', # 110
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
    # eval_return_dict.update(
    #     evaluator_rad.sample_emitter(
    #         emitter_params={
    #             'max_plate': 32, 
    #             'radiance_scale': radiance_scale, 
    #             'emitter_type_index_list': emitter_type_index_list, 
    #             }))
    
    '''
    sample non-emitter locations along envmap (hemisphere) directions radiance from rad-MLP: images/demo_envmap_o3d_sampling.png
    '''
    eval_return_dict.update(
        evaluator_rad.sample_lighting(
            # sample_type='emission', # 'emission', 'incident'
            sample_type='incident', # 'emission', 'incident'
            subsample_rate_pts=500, 
        )
    )

'''
Open3D 3D viewer
'''
if opt.vis_3d_o3d:
    visualizer_3D_o3d = visualizer_scene_3D_o3d(
        mitsuba_scene, 
        modality_list_vis=[
            # 'dense_geo', # fused from 2D
            'cameras', 
            # 'lighting_SG', # images/demo_lighting_SG_o3d.png; arrows in blue
            # 'lighting_envmap', # images/demo_lighting_envmap_o3d.png; arrows in pink
            # 'layout', 
            'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters SHAPES)
            'emitters', # emitter PROPERTIES (e.g. SGs, half envmaps)
            'mi', # mitsuba sampled rays, pts
            ], 
        if_debug_info=opt.if_debug_info, 
    )

    lighting_params_vis={
        # 'subsample_lighting_pts_rate': 500, # change this according to how sparse the lighting arrows you would like to be (also according to num of frame_ids)
        # 'lighting_keep_ratio': 0.05, 
        # 'lighting_further_clip_ratio': 0.1, 
        'lighting_scale': 10, 
        # 'lighting_keep_ratio': 0.2, # - good for lighting_SG
        # 'lighting_further_clip_ratio': 0.3, 
        'lighting_keep_ratio': 0., # - good for lighting_envmap
        'lighting_further_clip_ratio': 0., 
        # 'lighting_keep_ratio': 0., # - debug
        # 'lighting_further_clip_ratio': 0., 
        'lighting_autoscale': False, 
        }

    if opt.if_add_rays_from_eval:
        if 'emitter_rays_list' in eval_return_dict:
            assert opt.eval_rad
            for (lpts, lpts_end) in eval_return_dict['emitter_rays_list']:
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


    visualizer_3D_o3d.run_o3d(
        if_shader=opt.if_shader, # set to False to disable faycny shaders 
        cam_params={
            'if_cam_axis_only': False, 
            }, 
        # dense_geo_params={
        #     'subsample_pcd_rate': 1, # change this according to how sparse the points you would like to be (also according to num of frame_ids)
        #     'if_ceiling': False, # [OPTIONAL] remove ceiling points to better see the furniture 
        #     'if_walls': False, # [OPTIONAL] remove wall points to better see the furniture 
        #     'if_normal': False, # [OPTIONAL] turn off normals to avoid clusters
        #     'subsample_normal_rate_x': 2, 
        #     'pcd_color_mode': opt.pcd_color_mode_dense_geo, 
        #     }, 
        lighting_params=lighting_params_vis, 
        shapes_params={
            'simply_ratio': 0.1, # simply num of triangles to #triangles * simply_ratio
            'if_meshes': False, # [OPTIONAL] if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': False, # [OPTIONAL] if show labels (False: only show bboxes)
            'if_voxel_volume': False, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png
            'if_ceiling': False, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            'if_walls': False, # [OPTIONAL] remove wall meshes to better see the furniture 
        },
        emitter_params={
            # 'if_half_envmap': False, # [OPTIONAL] if show half envmap as a hemisphere for window emitters (False: only show bboxes)
            # 'scale_SG_length': 2., 
            'if_sampling_emitter': True, 
            'radiance_scale': radiance_scale, 
            'max_plate': 32, 
        },
        mi_params={
            'if_pts': True, # if show pts sampled by mi; should close to backprojected pts from OptixRenderer depth maps
            'if_pts_colorize_rgb': True, 
            'pts_subsample': 10,
            # 'if_ceiling': False, # [OPTIONAL] remove ceiling points to better see the furniture 
            # 'if_walls': False, # [OPTIONAL] remove wall points to better see the furniture 

            'if_cam_rays': False, 
            'cam_rays_if_pts': True, # if cam rays end in surface intersections; set to False to visualize rays of unit length
            'cam_rays_subsample': 10, 
            
            'if_normal': False, 
            'normal_subsample': 50, 
            'normal_scale': 0.2, 

        }, 
    )

