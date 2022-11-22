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
from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)

from lib.class_mitsubaScene3D import mitsubaScene3D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d
from lib.class_visualizer_scene_2D import visualizer_scene_2D

from lib.utils_misc import str2bool
import argparse
parser = argparse.ArgumentParser()
# visualizers
# parser.add_argument('--vis_3d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to visualize 3D with plt for debugging')
parser.add_argument('--vis_3d_o3d', type=str2bool, nargs='?', const=True, default=True, help='whether to visualize in open3D')
# parser.add_argument('--vis_2d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to show projection onto one image with plt (e.g. layout, object bboxes')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=False, help='')
# options for visualizers
parser.add_argument('--pcd_color_mode_dense_geo', type=str, default='rgb', help='colormap for all points in fused geo')
parser.add_argument('--if_set_pcd_color_mi', type=str2bool, nargs='?', const=True, default=False, help='if create color map for all points of Mitsuba; required: input_colors_tuple')
# parser.add_argument('--if_add_rays_from_renderer', type=str2bool, nargs='?', const=True, default=False, help='if add camera rays and emitter sample rays from renderer')
# differential renderer
# parser.add_argument('--render_3d', type=str2bool, nargs='?', const=True, default=False, help='differentiable surface rendering')
# parser.add_argument('--renderer_option', type=str, default='PhySG', help='differentiable renderer option')
# debug
parser.add_argument('--if_debug_info', type=str2bool, nargs='?', const=True, default=False, help='if show debug info')
opt = parser.parse_args()

base_root = Path(PATH_HOME) / 'data/scenes'
xml_root = Path(PATH_HOME) / 'data/scenes'
intrinsics_path = Path(PATH_HOME) / 'data/scenes/intrinsic_mitsubaScene.txt'

'''
The kitchen scene: data/scenes/kitchen/scene_v3.xml
'''
xml_filename = 'scene_v3.xml'
scene_name = 'kitchen'

openrooms_scene = mitsubaScene3D(
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root, 'xml_scene_root': xml_root}, 
    scene_params_dict={
        'xml_filename': xml_filename, 
        'scene_name': scene_name, 
        'mitsuba_version': '3.0.0', 
        'intrinsics_path': intrinsics_path, 
        'up_axis': 'y+', 
        'pose_file': ('OpenRooms', 'cam.txt'), 
        # 'pose_file': ('Blender', 'train.npy'), 
        }, 
    mi_params_dict={
        'if_also_dump_xml_with_lit_lamps_only': True,  # True: to dump a second file containing lit-up lamps only
        'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_sample_poses': True, # True to generate camera poses following Zhengqin's method (i.e. walking along walls)
        'poses_num': 200, 
        'if_render_im': True, # True to render im with Mitsuba
        'if_get_segs': True, # True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
    # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
    modality_list = [
        'im_sdr', 
        # 'seg', 'im_hdr', 
        # 'albedo', 'roughness', 
        # 'depth', 'normal', 
        # 'lighting_SG', 
        # 'lighting_envmap', 
        'layout', 
        'shapes', # objs + emitters, geometry shapes + emitter properties
        ], 
    im_params_dict={
        # 'im_H_resize': 480, 'im_W_resize': 640, 
        'im_H_load': 320, 'im_W_load': 640, 
        'im_H_resize': 160, 'im_W_resize': 320, 
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
    # lighting_params_dict={
    # }, 
    shape_params_dict={
        'if_load_obj_mesh': True, # set to False to not load meshes for objs (furniture) to save time
        'if_load_emitter_mesh': True,  # default True: to load emitter meshes, because not too many emitters
        },
    emitter_params_dict={
        },
)

# '''
# Matploblib 2D viewer
# '''
# if opt.vis_2d_plt:
#     visualizer_2D = visualizer_scene_2D(
#         openrooms_scene, 
#         modality_list_vis=[
#             'im', 
#             'layout', 
#             # 'shapes', 
#             # 'depth', 'mi_depth', 
#             # 'normal', 'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
#             # 'lighting_SG', # convert to lighting_envmap and vis: images/demo_lighting_SG_envmap_2D_plt.png
#             # 'lighting_envmap', 
#             # 'seg_area', 'seg_env', 'seg_obj', 
#             # 'mi_seg_area', 'mi_seg_env', 'mi_seg_obj', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
#             ], 
#         # frame_idx_list=[0, 1, 2, 3, 4], 
#         frame_idx_list=[0], 
#     )
#     visualizer_2D.vis_2d_with_plt(
#         lighting_params={
#             'lighting_scale': 0.01, # rescaling the brightness of the envmap
#             }, 
#             )

# '''
# Matploblib 3D viewer
# '''
# if opt.vis_3d_plt:
#     visualizer_3D_plt = visualizer_openroomsScene_3D_plt(
#         openrooms_scene, 
#         modality_list_vis = [
#             'layout', 
#             'poses', # camera center + optical axis
#             # 'shapes', # boxes and labels (no meshes in plt visualization)
#             # 'emitters', # emitter properties
#             # 'emitter_envs', # emitter envmaps for (1) global envmap (2) half envmap & SG envmap of each window
#             ], 
#     )
#     visualizer_3D_plt.vis_3d_with_plt()

# '''
# Differential renderers
# '''
# if opt.render_3d:
#     renderer_3D = renderer_openroomsScene_3D(
#         openrooms_scene, 
#         renderer_option=opt.renderer_option, 
#         host=host, 
#         renderer_params={
#             'pts_from': 'mi', 
#         }
#     )
    
#     renderer_return_dict = renderer_3D.render(
#         frame_idx=0, 
#         if_show_rendering_plt=True, 
#         render_params={
#             'max_plate': 256, 
#             'emitter_type_index': ('lamp', 0), 
#         })
    
#     if opt.renderer_option == 'ZQ_emitter':
#         ts = np.median(renderer_return_dict['ts'], axis=1)
#         visibility = np.amax(renderer_return_dict['visibility'], axis=1)
#         print('visibility', visibility.shape, np.sum(visibility)/float(visibility.shape[0]))
#         # from scipy import stats
#         # visibility = stats.mode(renderer_return_dict['visibility'], axis=1)[0].flatten()

'''
Open3D 3D viewer
'''
if opt.vis_3d_o3d:
    visualizer_3D_o3d = visualizer_scene_3D_o3d(
        openrooms_scene, 
        modality_list_vis=[
            # 'dense_geo', 
            'cameras', 
            # 'lighting_SG', # images/demo_lighting_SG_o3d.png; arrows in blue
            # 'lighting_envmap', # images/demo_lighting_envmap_o3d.png; arrows in pink
            'layout', 
            'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters)
            # 'emitters', # emitter properties (e.g. SGs, half envmaps)
            'mi', # mitsuba sampled rays, pts
            ], 
        if_debug_info=opt.if_debug_info, 
    )

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
        # lighting_params={
        #     'subsample_lighting_pts_rate': 100, # change this according to how sparse the lighting arrows you would like to be (also according to num of frame_ids)
        #     # 'lighting_scale': 1., 
        #     # 'lighting_keep_ratio': 0.05, 
        #     # 'lighting_clip_ratio': 0.1, 
        #     'lighting_scale': 0.5, 
        #     # 'lighting_keep_ratio': 0.2, # - good for lighting_SG
        #     # 'lighting_clip_ratio': 0.3, 
        #     'lighting_keep_ratio': 0.1, # - good for lighting_envmap
        #     'lighting_clip_ratio': 0.2, 
        #     'lighting_autoscale': True, 
        #     }, 
        shapes_params={
            'simply_ratio': 0.1, # simply num of triangles to #triangles * simply_ratio
            'if_meshes': False, # [OPTIONAL] if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': False, # [OPTIONAL] if show labels (False: only show bboxes)
            'if_voxel_volume': False, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png
        },
        # emitters_params={
        #     'if_half_envmap': False, # [OPTIONAL] if show half envmap as a hemisphere for window emitters (False: only show bboxes)
        #     'scale_SG_length': 2., 
        # },
        mi_params={
            'if_pts': True, # if show pts sampled by mi; should close to backprojected pts from OptixRenderer depth maps
            'if_pts_colorize_rgb': True, 
            'pts_subsample': 1,
            'if_ceiling': True, # [OPTIONAL] remove ceiling points to better see the furniture 
            'if_walls': True, # [OPTIONAL] remove wall points to better see the furniture 

            'if_cam_rays': False, 
            'cam_rays_if_pts': True, # if cam rays end in surface intersections; set to False to visualize rays of unit length
            'cam_rays_subsample': 10, 
            
            'if_normal': False, 
            'normal_subsample': 50, 
            'normal_scale': 0.2, 

        }, 
    )

